"""Causal counterfactual quarantine.

The corridor is represented as a Structural Causal Model (SCM) DAG

    Cyber / V2X inputs  ->  Perceived state  ->  Signal / advisory actuation
                        ->  Segment density rho_s(t)  ->  Queue spillback L_s(t)

Structural equations for the segment-level dynamics are *fitted* on nominal
(attack-free) rollouts.  Given an anomaly, the engine:

1. scores each attack-surface node with physically grounded cross-sensor /
   temporal-residual evidence (which disambiguates, e.g., a spoofed RSU that
   reports free-flow over a congested detector from an under-reporting
   detector);
2. runs ``do(quarantine = S)`` counterfactual rollouts on the mutilated SCM
   (the quarantined node's corrupted influence is cut and restored to nominal);
3. selects the **minimum-regret** quarantine subset -- the one minimising
   predicted downstream spillback plus a parsimony term -- while refusing to
   isolate the signal carrying mainline emergency progression.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from sklearn.linear_model import LinearRegression

from sumo_env.corridor_sim import FundamentalDiagram
from cyber_attacks.attack_injector import det_node, rsu_node, sig_node, PLAZA_NODE


@dataclass
class CorridorSCM:
    cfg: dict
    emergency_signal: int = 0          # intersection 0 carries mainline emergency progression

    def __post_init__(self) -> None:
        c = self.cfg["corridor"]
        self.n_seg = int(c["n_segments"])
        self.cps = int(c["cells_per_segment"])
        fd = self.cfg["fundamental_diagram"]
        self.fd = FundamentalDiagram(fd["v_free_kmh"], fd["rho_jam_vehkm"], fd["q_max_vehh"])
        self.rho_crit = self.fd.rho_crit
        self.plaza_seg = int(self.cfg["dcfc_plaza"]["segment_index"])
        sg = self.cfg["signals"]
        self.green_split = float(sg["green_split"])
        self.dt = float(c["dt_s"]) / 3600.0                 # h
        self.L_seg = float(c["segment_length_m"]) / 1000.0  # km (segment-level continuity)
        self.inflow = float(self.cfg["demand"]["inflow_vehh"])
        self.graph = self.build_graph()
        self.cap_ = np.full(self.n_seg, self.fd.q_max)      # per-segment discharge capacity
        self.inflow_eff = self.inflow * self.green_split
        self._fitted = False

    # -- DAG ---------------------------------------------------------------- #
    def build_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        for s in range(self.n_seg):
            g.add_nodes_from([rsu_node(s), det_node(s), sig_node(s),
                              f"PERC{s}", f"ACT{s}", f"RHO{s}", f"L{s}"])
            g.add_edge(rsu_node(s), f"PERC{s}")
            g.add_edge(det_node(s), f"PERC{s}")
            g.add_edge(f"PERC{s}", f"ACT{s}")
            g.add_edge(sig_node(s), f"ACT{s}")
            g.add_edge(f"ACT{s}", f"RHO{s}")
            g.add_edge(f"RHO{s}", f"L{s}")
            if s > 0:
                g.add_edge(f"RHO{s-1}", f"RHO{s}")      # upstream -> downstream coupling
        g.add_node(PLAZA_NODE)
        g.add_edge(PLAZA_NODE, f"RHO{self.plaza_seg}")
        return g

    # -- structural-equation fit ------------------------------------------- #
    def _discrepancy(self, meas: dict) -> np.ndarray:
        """Per-segment redundant-sensor discrepancy: RSU-derived minus detector density."""
        rsu = np.asarray(meas["rsu_speed"], float)
        det = np.asarray(meas["det_rho"], float)
        rho_v = self.fd.w * self.fd.rho_jam / (rsu + self.fd.w)
        return rho_v - det

    def fit(self, seg_density: Sequence[np.ndarray], green: Sequence[np.ndarray],
            meas: Optional[Sequence[Sequence[dict]]] = None) -> "CorridorSCM":
        """Calibrate the structural causal mechanism from nominal data.

        The structural assignment for each segment is the macroscopic
        conservation equation (density balance with a signal-gated discharge);
        here we *estimate its parameters* -- the per-segment effective discharge
        capacity ``cap_s`` (slope of discharge flow on the green indicator,
        regressed through the origin) -- and calibrate the nominal cross-sensor
        discrepancy distribution used for sensor attribution."""
        for s in range(self.n_seg):
            g, q = [], []
            for r_ep, g_ep in zip(seg_density, green):
                g.extend(g_ep[:, s])
                q.extend(self.fd.flux(r_ep[:, s]) * g_ep[:, s])
            reg = LinearRegression(fit_intercept=False).fit(
                np.asarray(g).reshape(-1, 1), np.asarray(q))
            self.cap_[s] = float(np.clip(reg.coef_[0], 0.3 * self.fd.q_max, self.fd.q_max))
        if meas is not None:
            disc = np.array([self._discrepancy(m) for ep in meas for m in ep])
            self.disc_mu = disc.mean(axis=0)
            self.disc_sd = disc.std(axis=0) + 1e-6
        else:
            self.disc_mu = np.zeros(self.n_seg)
            self.disc_sd = np.ones(self.n_seg)
        self._fitted = True
        return self

    # -- causal sensor attribution (redundancy / leave-one-out consistency) - #
    def attribute_sensors(self, meas_window: List[dict], z_thresh: float = 4.0) -> List[str]:
        """Flag the measurement node whose reading is causally inconsistent with
        its redundant partner, standardised against the nominal discrepancy.

        A positive anomaly (RSU-derived >> detector) implicates an
        under-reporting **detector**; a negative anomaly (detector >> RSU-derived)
        implicates a speed-spoofed **RSU** -- both hallmarks of congestion-hiding
        false-data injection."""
        disc = np.mean([self._discrepancy(m) for m in meas_window], axis=0)
        z = (disc - self.disc_mu) / self.disc_sd
        flagged: List[str] = []
        for s in range(self.n_seg):
            if z[s] > z_thresh:
                flagged.append(det_node(s))
            elif z[s] < -z_thresh:
                flagged.append(rsu_node(s))
        return flagged

    # -- counterfactual rollout -------------------------------------------- #
    def _rollout(self, r0: np.ndarray, green_plan: np.ndarray,
                 disturb: Dict[int, float], horizon: int) -> np.ndarray:
        """Roll the structural conservation mechanism forward under a green plan;
        returns the queue trajectory L [horizon, n_seg]. Signal green gates the
        discharge, so ``do(quarantine)`` that restores a starved green correctly
        drains the queue."""
        r = r0.astype(float).copy()
        lam = self.dt / self.L_seg
        L = np.zeros((horizon, self.n_seg))
        for t in range(horizon):
            S, R = self.fd.sending(r), self.fd.receiving(r)
            q_out = np.empty(self.n_seg)
            for s in range(self.n_seg):
                recv = R[s + 1] if s < self.n_seg - 1 else self.fd.q_max   # free exit
                q_out[s] = min(green_plan[s] * self.cap_[s], S[s], recv)
            q_in = np.empty(self.n_seg)
            q_in[0] = min(self.inflow_eff, R[0])
            q_in[1:] = q_out[:-1]
            dist = np.array([disturb.get(s, 0.0) for s in range(self.n_seg)])
            r = np.clip(r + lam * (q_in - q_out) + dist, 0.0, self.fd.rho_jam)
            L[t] = np.maximum(0.0, r - self.rho_crit)
        return L

    def blast_radius(self, r0, green_plan, disturb, horizon: int) -> float:
        L = self._rollout(r0, green_plan, disturb, horizon)
        # downstream-weighted spillback integral (further downstream == larger radius)
        w = 1.0 + np.arange(self.n_seg) / self.n_seg
        return float((L * w).sum())

    # -- node-level anomaly attribution (cross-sensor + temporal) ---------- #
    def anomaly_scores(self, meas_window: List[dict], forecast_window: List[np.ndarray],
                       green_attacked: np.ndarray, green_nominal: np.ndarray
                       ) -> Dict[str, float]:
        """Aggregate evidence over the attack window into per-node anomaly scores."""
        scores = {n: 0.0 for n in
                  [rsu_node(s) for s in range(self.n_seg)]
                  + [det_node(s) for s in range(self.n_seg)]
                  + [sig_node(s) for s in range(self.n_seg)]
                  + [PLAZA_NODE]}
        for meas, fc in zip(meas_window, forecast_window):
            det_rho = np.asarray(meas["det_rho"], float)
            rsu_speed = np.asarray(meas["rsu_speed"], float)
            for s in range(self.n_seg):
                rho_from_v = self.fd.w * self.fd.rho_jam / (rsu_speed[s] + self.fd.w)
                mismatch = rho_from_v - det_rho[s]
                congested_speed = rsu_speed[s] < self.fd.v_free - 5.0
                # RSU says free-flow but detector/forecast say congested -> RSU spoof
                if (not congested_speed) and (det_rho[s] > self.rho_crit or fc[s] > self.rho_crit):
                    scores[rsu_node(s)] += max(0.0, max(det_rho[s], fc[s]) - rho_from_v)
                # detector reads low while speed indicates congestion -> detector FDI
                if congested_speed and mismatch > 3.0:
                    scores[det_node(s)] += mismatch
                # persistent positive residual vs nominal forecast under starved green
                resid = det_rho[s] - fc[s]
                if resid > 3.0 and green_attacked[s] < green_nominal[s] - 1e-3:
                    scores[sig_node(s)] += resid
                if resid > 3.0 and s == self.plaza_seg:
                    scores[PLAZA_NODE] += resid
        return scores

    # -- minimum-regret selection ------------------------------------------ #
    def select_quarantine(self, r0: np.ndarray, green_attacked: np.ndarray,
                          green_nominal: np.ndarray, scores: Dict[str, float],
                          disturb_est: Dict[int, float], horizon: int = 60,
                          score_thresh: float = 40.0, parsimony: float = 5.0
                          ) -> Tuple[List[str], Dict[str, float]]:
        """Pick the quarantine subset minimising predicted spillback + parsimony."""
        candidates = [n for n, v in scores.items() if v >= score_thresh]
        node_seg = {**{rsu_node(s): s for s in range(self.n_seg)},
                    **{det_node(s): s for s in range(self.n_seg)},
                    **{sig_node(s): s for s in range(self.n_seg)},
                    PLAZA_NODE: self.plaza_seg}

        def plan_and_disturb(subset):
            g = green_attacked.copy()
            d = dict(disturb_est)
            for n in subset:
                s = node_seg[n]
                if n.startswith("SIG"):
                    if s == self.emergency_signal:       # preserve emergency progression
                        continue
                    g[s] = green_nominal[s]               # restore starved green
                else:
                    d[s] = 0.0                            # cut sensor/plaza disturbance
            return g, d

        best, best_cost = [], None
        info = {}
        for k in range(len(candidates) + 1):
            for subset in combinations(candidates, k):
                g, d = plan_and_disturb(subset)
                br = self.blast_radius(r0, g, d, horizon)
                cost = br + parsimony * len(subset)
                info[",".join(subset) or "(none)"] = br
                if best_cost is None or cost < best_cost:
                    best_cost, best = cost, list(subset)
        return best, info


if __name__ == "__main__":
    import numpy as np
    from sumo_env.corridor_sim import load_config
    cfg = load_config()
    scm = CorridorSCM(cfg)
    print("SCM nodes:", scm.graph.number_of_nodes(), "edges:", scm.graph.number_of_edges())
