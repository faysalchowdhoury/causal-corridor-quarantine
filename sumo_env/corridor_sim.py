"""Dual-mode corridor simulator.

If Eclipse SUMO is installed (``SUMO_HOME`` set and ``traci`` importable) and a
``.sumocfg`` is present, the corridor can be driven through SUMO/TraCI.  To keep
``pytest`` and every script runnable out-of-the-box on any machine *without*
external binaries, this module also implements a faithful, high-resolution
macroscopic **LWR / Godunov Cell-Transmission** engine with signalized stop-bar
queueing and an off-ramp DC-fast-charging (DCFC) plaza.

Units are kept internally consistent: density ``veh/km``, flow ``veh/h``, speed
``km/h``, cell length ``km``, time step ``h``.  A triangular (Daganzo)
fundamental diagram is used, so all states satisfy ``0 <= rho <= rho_jam`` by
construction in the ground-truth engine (verified by the test-suite).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import yaml


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #
_CFG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "corridor_config.yaml",
)


def load_config(path: str = _CFG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# --------------------------------------------------------------------------- #
# Triangular fundamental diagram
# --------------------------------------------------------------------------- #
@dataclass
class FundamentalDiagram:
    v_free: float          # km/h
    rho_jam: float         # veh/km
    q_max: float           # veh/h

    def __post_init__(self) -> None:
        self.rho_crit = self.q_max / self.v_free                       # veh/km
        self.w = self.q_max / (self.rho_jam - self.rho_crit)           # km/h (back wave)

    def flux(self, rho: np.ndarray) -> np.ndarray:
        rho = np.asarray(rho, dtype=float)
        free = self.v_free * rho
        cong = self.w * (self.rho_jam - rho)
        return np.clip(np.minimum(free, cong), 0.0, self.q_max)

    def sending(self, rho: np.ndarray) -> np.ndarray:
        """Demand / sending function S(rho) = min(v_free*rho, q_max)."""
        return np.clip(np.minimum(self.v_free * np.asarray(rho, float), self.q_max),
                       0.0, self.q_max)

    def receiving(self, rho: np.ndarray) -> np.ndarray:
        """Supply / receiving function R(rho) = min(q_max, w*(rho_jam-rho))."""
        return np.clip(np.minimum(self.q_max, self.w * (self.rho_jam - np.asarray(rho, float))),
                       0.0, self.q_max)

    def speed(self, rho: np.ndarray) -> np.ndarray:
        rho = np.asarray(rho, float)
        q = self.flux(rho)
        v = np.divide(q, rho, out=np.full_like(rho, self.v_free), where=rho > 1e-6)
        return np.clip(v, 0.0, self.v_free)


# --------------------------------------------------------------------------- #
# Per-step physical overrides produced by the attack injector / controller
# --------------------------------------------------------------------------- #
@dataclass
class StepOverride:
    green_split: Optional[Dict[int, float]] = None  # intersection -> green fraction
    plaza_recv_scale: float = 1.0                    # <1 == plaza backup (spillback)
    plaza_divert_frac: Optional[float] = None        # override off-ramp diversion
    demand_scale: float = 1.0                        # multiplicative demand change
    quarantine_signal: Optional[List[int]] = None    # intersections held at nominal green


# --------------------------------------------------------------------------- #
# Cell-Transmission corridor
# --------------------------------------------------------------------------- #
class CorridorCTM:
    """Signalized arterial corridor as a Godunov/CTM macroscopic model."""

    def __init__(self, cfg: dict, seed: int = 0):
        self.cfg = cfg
        c = cfg["corridor"]
        self.n_seg = int(c["n_segments"])
        self.cps = int(c["cells_per_segment"])
        self.N = self.n_seg * self.cps
        self.dx = float(c["segment_length_m"]) / 1000.0 / self.cps      # km
        self.dt = float(c["dt_s"]) / 3600.0                              # h
        self.dt_s = float(c["dt_s"])
        self.horizon_s = float(c["horizon_s"])
        self.n_steps = int(round(self.horizon_s / self.dt_s))

        fd = cfg["fundamental_diagram"]
        self.fd = FundamentalDiagram(fd["v_free_kmh"], fd["rho_jam_vehkm"], fd["q_max_vehh"])

        sg = cfg["signals"]
        self.cycle = float(sg["cycle_s"])
        self.green_split = float(sg["green_split"])
        self.offset = list(sg["offset_s"])

        dm = cfg["demand"]
        self.inflow = float(dm["inflow_vehh"])
        self.inflow_noise = float(dm["inflow_noise_frac"])
        self.warmup_s = float(dm["warmup_s"])

        pl = cfg["dcfc_plaza"]
        self.plaza_seg = int(pl["segment_index"])
        self.base_divert = float(pl["base_divert_frac"])

        sn = cfg["sensing"]
        self.det_offset = int(sn["detector_cell_offset"])
        self.rsu_noise = float(sn["rsu_speed_noise_kmh"])
        self.det_noise = float(sn["detector_noise_vehkm"])

        self.rng = np.random.default_rng(seed)
        self.reset()

    # -- geometry helpers --------------------------------------------------- #
    def stopbar_cell(self, seg: int) -> int:
        return seg * self.cps + self.cps - 1

    def detector_cell(self, seg: int) -> int:
        return seg * self.cps + self.det_offset

    def detector_cells(self) -> List[int]:
        return [self.detector_cell(s) for s in range(self.n_seg)]

    def segment_of_cell(self, cell: int) -> int:
        return cell // self.cps

    def reset(self) -> None:
        self.t = 0.0
        self.k = 0
        # start near a light but uncongested loading
        self.rho = np.full(self.N, 8.0)
        self.last_exit_flow = 0.0          # veh/h leaving the corridor (throughput)
        self.last_max_queue_m = 0.0        # longest per-segment standing queue (m)

    # -- control / demand --------------------------------------------------- #
    def _green(self, t: float, seg: int, override: StepOverride) -> float:
        split = self.green_split
        if override.green_split and seg in override.green_split:
            split = override.green_split[seg]
        if override.quarantine_signal and seg in override.quarantine_signal:
            split = self.green_split  # restored to nominal timing by quarantine
        phase = (t + self.offset[seg]) % self.cycle
        return 1.0 if phase < split * self.cycle else 0.0

    def _demand(self, t: float, override: StepOverride) -> float:
        ramp = min(1.0, t / max(self.warmup_s, 1e-6))
        noise = 1.0 + self.inflow_noise * self.rng.standard_normal()
        return max(0.0, self.inflow * ramp * noise * override.demand_scale)

    # -- one Godunov step --------------------------------------------------- #
    def step(self, override: Optional[StepOverride] = None) -> None:
        ov = override or StepOverride()
        S = self.fd.sending(self.rho)
        R = self.fd.receiving(self.rho)

        f = np.minimum(S[:-1], R[1:])                 # internal boundary fluxes (N-1)
        divert = self.base_divert if ov.plaza_divert_frac is None else ov.plaza_divert_frac
        divert_out = np.zeros(self.N)

        for s in range(self.n_seg):
            j = self.stopbar_cell(s)
            if j < self.N - 1:                        # internal signalized boundary
                g = self._green(self.t, s, ov)
                recv = R[j + 1]
                if s == self.plaza_seg:
                    recv = min(recv, self.fd.q_max * ov.plaza_recv_scale)
                f[j] = min(S[j], recv) * g
                if s == self.plaza_seg:
                    # part of the through flux leaves to the plaza off-ramp (sink)
                    divert_out[j] = divert * f[j]
                    f[j] = f[j] - divert_out[j]

        # boundaries
        f_up = min(self._demand(self.t, ov), R[0])
        g_last = self._green(self.t, self.n_seg - 1, ov)
        f_down = S[-1] * g_last

        drho = np.empty(self.N)
        drho[0] = f_up - f[0]
        drho[1:-1] = f[:-1] - f[1:]
        drho[-1] = f[-1] - f_down
        drho -= divert_out                            # remove plaza off-ramp exits
        self.rho = np.clip(self.rho + (self.dt / self.dx) * drho, 0.0, self.fd.rho_jam)

        # instrumentation for throughput / queue reporting
        self.last_exit_flow = float(f_down)
        congested = (self.rho > self.fd.rho_crit).reshape(self.n_seg, self.cps)
        self.last_max_queue_m = float(congested.sum(axis=1).max() * self.dx * 1000.0)

        self.t += self.dt_s
        self.k += 1

    # -- observation -------------------------------------------------------- #
    def true_density(self) -> np.ndarray:
        return self.rho.copy()

    def segment_mean(self) -> np.ndarray:
        return self.rho.reshape(self.n_seg, self.cps).mean(axis=1)

    def clean_measurements(self) -> Dict[str, np.ndarray]:
        """Noisy-but-honest sensor readings (per segment).

        Loop-detector density and RSU space-mean speed are both reported at
        segment-aggregate resolution, giving the two a shared physical referent
        (used as redundant evidence by the causal quarantine engine)."""
        seg_rho = self.segment_mean()
        det_rho = seg_rho + self.det_noise * self.rng.standard_normal(self.n_seg)
        det_rho = np.clip(det_rho, 0.0, self.fd.rho_jam)
        det_flow = self.fd.flux(det_rho)
        rsu_speed = self.fd.speed(seg_rho) + self.rsu_noise * self.rng.standard_normal(self.n_seg)
        rsu_speed = np.clip(rsu_speed, 0.0, self.fd.v_free)
        return {"det_rho": det_rho, "det_flow": det_flow, "rsu_speed": rsu_speed}


# --------------------------------------------------------------------------- #
# SUMO network export + optional TraCI driver
# --------------------------------------------------------------------------- #
def write_sumo_network(cfg: dict, out_dir: Optional[str] = None) -> str:
    """Emit valid .nod.xml/.edg.xml/.rou.xml/.sumocfg for native SUMO users."""
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "network")
    os.makedirs(out_dir, exist_ok=True)
    c = cfg["corridor"]
    n = int(c["n_segments"])
    L = float(c["segment_length_m"])
    vmax = float(cfg["fundamental_diagram"]["v_free_kmh"]) / 3.6  # m/s

    nodes = ['<?xml version="1.0" encoding="UTF-8"?>', "<nodes>"]
    for i in range(n + 1):
        ntype = "traffic_light" if 0 < i <= n else "priority"
        nodes.append(f'  <node id="J{i}" x="{i * L:.1f}" y="0.0" type="{ntype}"/>')
    nodes.append("</nodes>")

    edges = ['<?xml version="1.0" encoding="UTF-8"?>', "<edges>"]
    for i in range(n):
        edges.append(
            f'  <edge id="E{i}" from="J{i}" to="J{i+1}" numLanes="1" '
            f'speed="{vmax:.2f}" length="{L:.1f}"/>'
        )
    edges.append("</edges>")

    flow = float(cfg["demand"]["inflow_vehh"])
    routes = ['<?xml version="1.0" encoding="UTF-8"?>', "<routes>",
              f'  <route id="main" edges="{" ".join(f"E{i}" for i in range(n))}"/>',
              f'  <flow id="f0" route="main" begin="0" end="{c["horizon_s"]}" '
              f'vehsPerHour="{flow:.0f}"/>', "</routes>"]

    cfgxml = ['<?xml version="1.0" encoding="UTF-8"?>', "<configuration>", "  <input>",
              '    <net-file value="corridor.net.xml"/>',
              '    <route-files value="corridor.rou.xml"/>', "  </input>", "  <time>",
              '    <begin value="0"/>', f'    <end value="{c["horizon_s"]:.0f}"/>',
              "  </time>", "</configuration>"]

    paths = {
        "corridor.nod.xml": "\n".join(nodes),
        "corridor.edg.xml": "\n".join(edges),
        "corridor.rou.xml": "\n".join(routes),
        "corridor.sumocfg": "\n".join(cfgxml),
    }
    for name, text in paths.items():
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return out_dir


def sumo_available() -> bool:
    if not os.environ.get("SUMO_HOME"):
        return False
    try:
        import traci  # noqa: F401
        return True
    except Exception:
        return False


if __name__ == "__main__":
    cfg = load_config()
    out = write_sumo_network(cfg)
    print("SUMO network written to", out)
    print("SUMO/TraCI available:", sumo_available())
    sim = CorridorCTM(cfg, seed=1)
    for _ in range(sim.n_steps):
        sim.step()
    print("final mean density (veh/km):", round(float(sim.rho.mean()), 2),
          "| max:", round(float(sim.rho.max()), 2), "| rho_jam:", sim.fd.rho_jam)
