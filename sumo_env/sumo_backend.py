"""Real Eclipse SUMO backend (compiles the network and drives it via libsumo).

This exercises the SUMO path end-to-end when SUMO is installed (e.g. the
``eclipse-sumo`` + ``libsumo`` pip wheels, or a system install exposing
``SUMO_HOME``), and extracts per-segment macroscopic density/flow so the
built-in LWR/Godunov engine can be validated against microsimulation.

If SUMO is not available, :func:`sumo_ready` returns ``False`` and callers fall
back to the analytic CTM.
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional

import numpy as np

from sumo_env.corridor_sim import write_sumo_network


def _sumo_home() -> Optional[str]:
    if os.environ.get("SUMO_HOME"):
        return os.environ["SUMO_HOME"]
    try:
        import sumo
        return getattr(sumo, "SUMO_HOME", os.path.dirname(sumo.__file__))
    except Exception:
        return None


def _bin(name: str) -> Optional[str]:
    home = _sumo_home()
    if not home:
        return None
    for cand in (os.path.join(home, "bin", name + ".exe"),
                 os.path.join(home, "bin", name)):
        if os.path.exists(cand):
            return cand
    return None


def sumo_ready() -> bool:
    if _sumo_home() is None:
        return False
    try:
        home = _sumo_home()
        binp = os.path.join(home, "bin")
        if os.path.isdir(binp) and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(binp)
        import libsumo  # noqa: F401
        return True
    except Exception:
        try:
            import traci  # noqa: F401
            return _bin("sumo") is not None
        except Exception:
            return False


def build_network(cfg: dict, net_dir: Optional[str] = None) -> str:
    """Emit node/edge/route XML and compile corridor.net.xml with netconvert."""
    if net_dir is None:
        net_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "network")
    write_sumo_network(cfg, net_dir)
    netconvert = _bin("netconvert")
    if netconvert is None:
        raise RuntimeError("netconvert not found; is SUMO installed?")
    net_out = os.path.join(net_dir, "corridor.net.xml")
    cmd = [netconvert,
           "--node-files", os.path.join(net_dir, "corridor.nod.xml"),
           "--edge-files", os.path.join(net_dir, "corridor.edg.xml"),
           "--output-file", net_out,
           "--tls.guess", "true", "--tls.default-type", "static",
           "--no-turnarounds", "true", "--no-internal-links", "true"]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return net_out


def _match_signal_timing(tc, cfg: dict) -> None:
    """Override every traffic light with a fixed program matching the CTM
    (same cycle and green split) for a like-for-like macroscopic comparison."""
    cycle = float(cfg["signals"]["cycle_s"])
    split = float(cfg["signals"]["green_split"])
    yellow = 3.0
    green = max(1.0, split * cycle - yellow)
    red = max(1.0, cycle - green - yellow)
    for tl in tc.trafficlight.getIDList():
        links = tc.trafficlight.getControlledLinks(tl)
        m = len(links)
        phases = [tc.trafficlight.Phase(green, "G" * m),
                  tc.trafficlight.Phase(yellow, "y" * m),
                  tc.trafficlight.Phase(red, "r" * m)]
        logic = tc.trafficlight.Logic("match", 0, 0, phases)
        tc.trafficlight.setProgramLogic(tl, logic)
        tc.trafficlight.setProgram(tl, "match")


def run_sumo_corridor(cfg: dict, seed: int = 0, net_dir: Optional[str] = None):
    """Run SUMO and return per-segment (edge) density [T, n_seg] (veh/km),
    space-mean speed [T, n_seg] (km/h), and sample times (s)."""
    home = _sumo_home()
    binp = os.path.join(home, "bin")
    if os.path.isdir(binp) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(binp)
    try:
        import libsumo as tc
    except Exception:
        import traci as tc

    if net_dir is None:
        net_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "network")
    net = os.path.join(net_dir, "corridor.net.xml")
    if not os.path.exists(net):
        build_network(cfg, net_dir)

    n = int(cfg["corridor"]["n_segments"])
    L_km = float(cfg["corridor"]["segment_length_m"]) / 1000.0
    horizon = float(cfg["corridor"]["horizon_s"])
    edges = [f"E{i}" for i in range(n)]
    sumo_bin = _bin("sumo")

    tc.start([sumo_bin, "-c", os.path.join(net_dir, "corridor.sumocfg"),
              "--step-length", "1.0", "--seed", str(int(seed)),
              "--no-step-log", "true", "--no-warnings", "true"])
    _match_signal_timing(tc, cfg)
    dens, spd, times = [], [], []
    try:
        step = 0
        while step < int(horizon):
            tc.simulationStep()
            d_row, v_row = [], []
            for e in edges:
                nveh = tc.edge.getLastStepVehicleNumber(e)
                v = tc.edge.getLastStepMeanSpeed(e)                 # m/s
                d_row.append(nveh / L_km)                           # veh/km
                v_row.append(max(0.0, v) * 3.6)                     # km/h
            dens.append(d_row); spd.append(v_row); times.append(step)
            step += 1
    finally:
        tc.close()
    return np.asarray(dens), np.asarray(spd), np.asarray(times)


if __name__ == "__main__":
    from sumo_env.corridor_sim import load_config
    cfg = load_config()
    print("SUMO ready:", sumo_ready())
    if sumo_ready():
        net = build_network(cfg)
        print("compiled", net)
        d, v, t = run_sumo_corridor(cfg, seed=1)
        print("SUMO density shape", d.shape, "| mean", round(float(d.mean()), 1),
              "| max", round(float(d.max()), 1), "veh/km")
