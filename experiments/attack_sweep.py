"""Attack-strength sweep and sensor-attribution ROC.

Addresses a reviewer concern: results should not hinge on one hand-picked attack
magnitude.  We (a) sweep a normalised attack-strength scalar ``s in [0,1]`` and
record reconstruction RMSE for every method under the compound attack, and
(b) at a fixed moderate strength, sweep the attribution z-threshold to trace the
ROC of the causal cross-sensor quarantine (true/false sensor-flag rates), with
its AUC.

Outputs ``results/attack_sweep.csv`` and ``results/roc.csv``.
"""
from __future__ import annotations

import copy
import os
import sys
from typing import List

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sumo_env.corridor_sim import load_config
from cyber_attacks.attack_injector import AttackInjector, rsu_node, det_node
from estimators.reconstructors import (CTMProcessModel, ExtendedKalmanReconstructor,
                                       UnconstrainedNeuralReconstructor,
                                       GraphNeuralReconstructor,
                                       ConservationConstrainedReconstructor)
from causal_engine.scm_quarantine import CorridorSCM
import experiments.run_benchmarks as rb

RESULTS = os.path.join(ROOT, "results")


def scaled_config(cfg: dict, s: float) -> dict:
    """Map strength s in [0,1] onto every compound-attack magnitude."""
    c = copy.deepcopy(cfg)
    a = c["attacks"]
    vfree = c["fundamental_diagram"]["v_free_kmh"]
    a["scenario_A_rsu_spoof"]["spoof_speed_kmh"] = float(vfree * (0.70 + 0.28 * s))
    a["scenario_B_green_starvation"]["green_split_attacked"] = float(0.50 - 0.40 * s)
    a["scenario_C_plaza_surge"]["spoofed_advisory_gain"] = float(0.05 + 0.50 * s)
    a["scenario_C_plaza_surge"]["surge_divert_frac"] = float(0.10 + 0.40 * s)
    return c


def build_estimators(cfg: dict):
    train_seeds = [cfg["experiment"]["base_seed"] - 1 - i
                   for i in range(cfg["experiment"]["train_episodes"])]
    X, F, Y, seg, grn, meas = rb.collect_training(cfg, train_seeds)
    neural = UnconstrainedNeuralReconstructor(cfg); neural.fit(X, Y)
    gnn = GraphNeuralReconstructor(cfg); gnn.fit(F, Y)
    scm = CorridorSCM(cfg); scm.fit(seg, grn, meas)
    return neural, gnn, scm, CTMProcessModel(cfg), ExtendedKalmanReconstructor(cfg), \
        ConservationConstrainedReconstructor(cfg)


def main(n_seeds: int = 12) -> None:
    cfg = load_config()
    dt_s = cfg["corridor"]["dt_s"]
    rho_jam = cfg["fundamental_diagram"]["rho_jam_vehkm"]
    n_seg = cfg["corridor"]["n_segments"]
    onset_k = int(cfg["attacks"]["onset_s"] / dt_s)
    dur_k = int(cfg["attacks"]["duration_s"] / dt_s)
    base = cfg["experiment"]["base_seed"]
    neural, gnn, scm, pm, ekf, cons = build_estimators(cfg)   # fit once on nominal

    # ---- (a) RMSE vs attack strength ------------------------------------- #
    print("[1/2] attack-strength sweep (RMSE vs s) ...")
    rows = []
    strengths = np.linspace(0.0, 1.0, 6)
    for s in strengths:
        cs = scaled_config(cfg, s)
        for ep in range(n_seeds):
            inj = AttackInjector(cs, "compound")
            d = rb.run_truth(cs, base + ep, inj)
            mask, _ = rb.decide_quarantine(cs, scm, pm, d)
            for rec, use_mask in ((ekf, False), (neural, False), (gnn, False), (cons, True)):
                rec.reset(d["rho"][0]); est = np.zeros_like(d["rho"])
                for k in range(d["T"]):
                    qm = mask if (use_mask and k >= onset_k) else None
                    est[k] = rec.reconstruct(d["meas"][k], k * dt_s, qm)
                rows.append({"strength": round(float(s), 3), "method": rec.name,
                             "rmse_vehkm": rb.rmse(est, d["rho"]),
                             "violation_pct": rb.violation_rate(est, rho_jam)})
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "attack_sweep.csv"), index=False)

    # ---- (b) sensor-attribution ROC at fixed strength -------------------- #
    print("[2/2] sensor-attribution ROC (per-sensor detection) ...")
    from sklearn.metrics import roc_curve, roc_auc_score
    scores: List[float] = []
    labels: List[int] = []
    for s_lvl in (0.15, 0.30, 0.45, 0.60):     # pool weak..moderate attacks
        cs = scaled_config(cfg, s_lvl)
        for ep in range(n_seeds):
            inj = AttackInjector(cs, "compound")
            d = rb.run_truth(cs, base + ep, inj)
            win = [d["meas"][k] for k in range(onset_k, min(onset_k + dur_k, d["T"]))]
            truth = set(inj.compromised_nodes_any())
            z = scm.sensor_zscores(win)
            for s in range(n_seg):             # two candidate sensors per segment
                scores.append(float(max(0.0, -z[s]))); labels.append(int(rsu_node(s) in truth))
                scores.append(float(max(0.0, z[s]))); labels.append(int(det_node(s) in truth))
    fpr, tpr, thr = roc_curve(labels, scores)
    auc = float(roc_auc_score(labels, scores))
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(os.path.join(RESULTS, "roc.csv"), index=False)
    with open(os.path.join(RESULTS, "roc_auc.txt"), "w") as fh:
        fh.write(f"{auc:.4f}\n")
    print(f"      sensor-attribution ROC AUC = {auc:.3f}")

    print("\nRMSE vs strength (mean):")
    piv = pd.read_csv(os.path.join(RESULTS, "attack_sweep.csv"))
    print(piv.groupby(["strength", "method"]).rmse_vehkm.mean().round(2).unstack().round(2))
    print(f"\nWrote results to {RESULTS}")


if __name__ == "__main__":
    main()
