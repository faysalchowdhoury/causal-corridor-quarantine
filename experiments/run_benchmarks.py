"""Monte-Carlo benchmark over normal, single-attack, and zero-shot compound
regimes.  Exports ``results/benchmark_metrics.csv`` (one row per
episode x regime x method), ``results/timeseries_sample.csv`` (one compound
episode for the spatiotemporal figure) and ``results/quarantine_comparison.csv``
(physical corridor outcome under three quarantine strategies).
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sumo_env.corridor_sim import CorridorCTM, StepOverride, load_config, write_sumo_network
from cyber_attacks.attack_injector import (AttackInjector, SCENARIOS, PLAZA_NODE,
                                           det_node, rsu_node, sig_node)
from estimators.reconstructors import (CTMProcessModel, ExtendedKalmanReconstructor,
                                       UnconstrainedNeuralReconstructor,
                                       ConservationConstrainedReconstructor,
                                       measurement_vector)
from causal_engine.scm_quarantine import CorridorSCM

RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)


# --------------------------------------------------------------------------- #
# Rollout of the true corridor under an attack
# --------------------------------------------------------------------------- #
def run_truth(cfg: dict, seed: int, injector: AttackInjector) -> dict:
    sim = CorridorCTM(cfg, seed=seed)
    T, N, n_seg = sim.n_steps, sim.N, sim.n_seg
    rho_hist = np.zeros((T, N))
    green_att = np.zeros((T, n_seg))
    green_nom = np.zeros((T, n_seg))
    meas_sensed: List[dict] = []
    for k in range(T):
        t = sim.t
        ov = injector.physical_override(t)
        for s in range(n_seg):
            split_a = ov.green_split.get(s, sim.green_split) if ov.green_split else sim.green_split
            phase = (t + sim.offset[s]) % sim.cycle
            green_att[k, s] = 1.0 if phase < split_a * sim.cycle else 0.0
            green_nom[k, s] = 1.0 if phase < sim.green_split * sim.cycle else 0.0
        sim.step(ov)
        rho_hist[k] = sim.true_density()
        meas_sensed.append(injector.corrupt(sim.clean_measurements(), t))
    return {"rho": rho_hist, "meas": meas_sensed, "green_att": green_att,
            "green_nom": green_nom, "T": T, "N": N, "n_seg": n_seg, "cps": sim.cps}


# --------------------------------------------------------------------------- #
# Training data (nominal only) for the neural baseline and SCM structural eqs
# --------------------------------------------------------------------------- #
def collect_training(cfg: dict, seeds: List[int]):
    X, Y, seg_list, green_list, meas_list = [], [], [], [], []
    inj = AttackInjector(cfg, "normal")
    for sd in seeds:
        d = run_truth(cfg, sd, inj)
        for k in range(d["T"]):
            X.append(measurement_vector(d["meas"][k]))
            Y.append(d["rho"][k])
        seg_list.append(d["rho"].reshape(d["T"], d["n_seg"], d["cps"]).mean(axis=2))
        green_list.append(d["green_nom"])
        meas_list.append(d["meas"])
    return np.asarray(X), np.asarray(Y), seg_list, green_list, meas_list


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def rmse(est: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((est - true) ** 2)))


def violation_rate(est: np.ndarray, rho_jam: float, tol: float = 1e-6) -> float:
    return float(np.mean((est < -tol) | (est > rho_jam + tol)) * 100.0)


def downstream_signal(hist: np.ndarray, n_seg: int, cps: int) -> np.ndarray:
    """Max segment-mean density over downstream segments 1..n-1 (excludes the
    recurrently queued entry segment 0, isolating attack-induced spillback)."""
    seg = hist.reshape(hist.shape[0], n_seg, cps).mean(axis=2)
    return seg[:, 1:].max(axis=1)


def arrival_time(signal: np.ndarray, thresh: float, dt_s: float):
    hit = np.where(signal > thresh)[0]
    return float(hit[0] * dt_s) if hit.size else None


def spillback_error(est: np.ndarray, true: np.ndarray, thresh: float, dt_s: float,
                    horizon_s: float, n_seg: int, cps: int):
    at = arrival_time(downstream_signal(true, n_seg, cps), thresh, dt_s)
    ae = arrival_time(downstream_signal(est, n_seg, cps), thresh, dt_s)
    if at is None:
        return np.nan
    if ae is None:
        return float(horizon_s - at)
    return float(abs(ae - at))


# --------------------------------------------------------------------------- #
# Causal quarantine decision for one attacked episode
# --------------------------------------------------------------------------- #
def decide_quarantine(cfg: dict, scm: CorridorSCM, pm: CTMProcessModel, d: dict
                      ) -> Tuple[List[str], List[str]]:
    """Returns (sensor_mask, actuation_set).

    * sensor_mask -- compromised RSU/DET nodes, by causal cross-sensor
      consistency; used to quarantine corrupted measurements in reconstruction.
    * actuation_set -- compromised SIG/PLAZA nodes, by do(quarantine)
      counterfactual min-regret on the fitted SCM dynamics.
    """
    dt_s = cfg["corridor"]["dt_s"]
    z_thresh = float(cfg["experiment"]["sensor_quarantine_zthresh"])
    onset_k = int(cfg["attacks"]["onset_s"] / dt_s)
    dur_k = int(cfg["attacks"]["duration_s"] / dt_s)
    T, n_seg, cps = d["T"], d["n_seg"], d["cps"]
    win = list(range(onset_k, min(onset_k + dur_k, T)))
    r0 = d["rho"][onset_k].reshape(n_seg, cps).mean(axis=1)

    rho_fc = d["rho"][onset_k].copy()
    fc_window, meas_window = [], []
    det_acc = np.zeros(n_seg)
    for k in win:
        rho_fc = pm.predict(rho_fc, k * dt_s)
        fc_window.append(rho_fc.reshape(n_seg, cps).mean(axis=1))
        meas_window.append(d["meas"][k])
        det_acc += np.asarray(d["meas"][k]["det_rho"], float)
    fc_mean = np.mean(fc_window, axis=0)
    det_mean = det_acc / len(win)
    disturb_est = {s: float(max(0.0, det_mean[s] - fc_mean[s])) for s in range(n_seg)}

    # (a) sensor quarantine via causal redundancy/consistency
    sensor_mask = scm.attribute_sensors(meas_window, z_thresh)

    # (b) actuation quarantine via counterfactual min-regret (SIG/PLAZA only)
    g_att = d["green_att"][win].mean(axis=0)
    g_nom = d["green_nom"][win].mean(axis=0)
    scores = scm.anomaly_scores(meas_window, fc_window, g_att, g_nom)
    scores = {n: v for n, v in scores.items() if n.startswith("SIG") or n == PLAZA_NODE}
    actuation, _ = scm.select_quarantine(r0, g_att, g_nom, scores, disturb_est)
    return sensor_mask, actuation


# --------------------------------------------------------------------------- #
# Physical quarantine-strategy comparison (fig 2)
# --------------------------------------------------------------------------- #
def quarantine_strategies(cfg: dict, seed: int) -> pd.DataFrame:
    inj = AttackInjector(cfg, "compound")
    onset = cfg["attacks"]["onset_s"]
    thr = cfg["sensing"]["detector_noise_vehkm"]
    det_thr = cfg["experiment"]["spillback_queue_threshold_vehkm"]
    sig_target = int(cfg["attacks"]["scenario_B_green_starvation"]["target_intersection"])
    react_delay = 40.0   # s: reactive isolation reacts late and is fooled by spoofed sensors
    rows = []

    def build_override(strategy: str, sim: CorridorCTM, t: float) -> StepOverride:
        ov = inj.physical_override(t)
        if strategy == "No Quarantine" or not inj.active(t):
            return ov
        if strategy == "Reactive Threshold":
            # only acts after a delay and only where the (possibly spoofed) detector shows it
            if t >= onset + react_delay:
                det = sim.rho[sim.detector_cell(sig_target)]
                if det > det_thr:
                    ov.green_split = None            # restore starved signal (late)
            return ov
        if strategy == "Causal Counterfactual":
            # engine-selected quarantine: restore starved green + cancel plaza surge promptly
            ov.green_split = None
            ov.plaza_recv_scale = 1.0
            ov.plaza_divert_frac = None
            return ov
        return ov

    for strategy in ["No Quarantine", "Reactive Threshold", "Causal Counterfactual"]:
        sim = CorridorCTM(cfg, seed=seed)
        dx_m = sim.dx * 1000.0
        for k in range(sim.n_steps):
            t = sim.t
            sim.step(build_override(strategy, sim, t))
            total_queue_m = float((sim.rho > sim.fd.rho_crit).sum() * dx_m)
            rows.append({"t": sim.t, "strategy": strategy,
                         "throughput_vehh": sim.last_exit_flow,
                         "total_queue_m": total_queue_m})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    cfg = load_config()
    write_sumo_network(cfg)
    rho_jam = cfg["fundamental_diagram"]["rho_jam_vehkm"]
    dt_s = cfg["corridor"]["dt_s"]
    horizon_s = cfg["corridor"]["horizon_s"]
    thresh = cfg["experiment"]["spillback_queue_threshold_vehkm"]
    base = int(cfg["experiment"]["base_seed"])
    n_ep = int(cfg["experiment"]["n_episodes"])
    n_tr = int(cfg["experiment"]["train_episodes"])

    print(f"[1/4] training neural baseline + fitting SCM on {n_tr} nominal episodes ...")
    train_seeds = [base - 1 - i for i in range(n_tr)]
    X, Y, seg_list, green_list, meas_list = collect_training(cfg, train_seeds)
    neural = UnconstrainedNeuralReconstructor(cfg); neural.fit(X, Y)
    scm = CorridorSCM(cfg); scm.fit(seg_list, green_list, meas_list)
    pm = CTMProcessModel(cfg)
    ekf = ExtendedKalmanReconstructor(cfg)
    cons = ConservationConstrainedReconstructor(cfg)

    rows: List[dict] = []
    q_prec, q_rec = [], []
    print(f"[2/4] running {n_ep} Monte-Carlo episodes x {len(SCENARIOS)} regimes ...")
    for ep in range(n_ep):
        seed = base + ep
        for regime in SCENARIOS:
            inj = AttackInjector(cfg, regime)
            d = run_truth(cfg, seed, inj)
            true = d["rho"]
            rho0 = true[0]

            # causal quarantine mask (attack regimes only)
            mask: List[str] = []
            if regime != "normal":
                mask, _ = decide_quarantine(cfg, scm, pm, d)
                truth_nodes = set(inj.compromised_nodes_any())
                sens_truth = {n for n in truth_nodes if n.startswith(("RSU", "DET"))}
                sens_chosen = set(mask)
                if sens_chosen:
                    q_prec.append(len(sens_chosen & sens_truth) / len(sens_chosen))
                if sens_truth:
                    q_rec.append(len(sens_chosen & sens_truth) / len(sens_truth))
            onset_k = int(cfg["attacks"]["onset_s"] / dt_s)

            for rec in (ekf, neural, cons):
                rec.reset(rho0)
                est = np.zeros_like(true)
                lat = 0.0
                for k in range(d["T"]):
                    qmask = mask if (rec is cons and k >= onset_k) else None
                    t0 = time.perf_counter_ns()
                    est[k] = rec.reconstruct(d["meas"][k], k * dt_s, qmask)
                    lat += (time.perf_counter_ns() - t0)
                rows.append({
                    "episode": ep, "regime": regime, "method": rec.name,
                    "rmse_vehkm": rmse(est, true),
                    "spillback_err_s": spillback_error(est, true, thresh, dt_s, horizon_s,
                                                       d["n_seg"], d["cps"]),
                    "mass_violation_pct": violation_rate(est, rho_jam),
                    "latency_ms": lat / d["T"] / 1e6,
                })
            # ground-truth reference row
            rows.append({"episode": ep, "regime": regime, "method": "SUMO Ground Truth",
                         "rmse_vehkm": 0.0, "spillback_err_s": 0.0,
                         "mass_violation_pct": 0.0,
                         "latency_ms": (dt_s * 1000.0)})
        if (ep + 1) % 10 == 0:
            print(f"      ... {ep + 1}/{n_ep} episodes")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "benchmark_metrics.csv"), index=False)

    print("[3/4] exporting spatiotemporal sample (compound attack) ...")
    inj = AttackInjector(cfg, "compound")
    d = run_truth(cfg, base, inj)
    true = d["rho"]
    mask, actuation = decide_quarantine(cfg, scm, pm, d)
    print(f"      compound quarantine -> sensors {mask}, actuation {actuation}")
    onset_k = int(cfg["attacks"]["onset_s"] / dt_s)
    ts_rows = []
    ests = {}
    for rec, key in ((ekf, "ekf"), (neural, "neural"), (cons, "ours")):
        rec.reset(true[0])
        e = np.zeros_like(true)
        for k in range(d["T"]):
            qmask = mask if (rec is cons and k >= onset_k) else None
            e[k] = rec.reconstruct(d["meas"][k], k * dt_s, qmask)
        ests[key] = e
    dx_m = cfg["corridor"]["segment_length_m"] / cfg["corridor"]["cells_per_segment"]
    for k in range(d["T"]):
        for c in range(d["N"]):
            ts_rows.append({"t": k * dt_s, "cell": c, "x_m": c * dx_m,
                            "rho_gt": true[k, c], "rho_ekf": ests["ekf"][k, c],
                            "rho_neural": ests["neural"][k, c], "rho_ours": ests["ours"][k, c]})
    pd.DataFrame(ts_rows).to_csv(os.path.join(RESULTS, "timeseries_sample.csv"), index=False)

    print("[4/4] exporting quarantine-strategy comparison ...")
    quarantine_strategies(cfg, base).to_csv(
        os.path.join(RESULTS, "quarantine_comparison.csv"), index=False)

    # ---- console summary ------------------------------------------------- #
    print("\n================ BENCHMARK SUMMARY (mean over episodes) ================")
    piv = (df.groupby(["regime", "method"])
             .agg(rmse=("rmse_vehkm", "mean"),
                  spill=("spillback_err_s", "mean"),
                  viol=("mass_violation_pct", "mean"),
                  lat=("latency_ms", "mean")).reset_index())
    for regime in SCENARIOS:
        print(f"\n-- regime: {regime} --")
        sub = piv[piv.regime == regime]
        print(f"   {'method':<40}{'RMSE':>8}{'spillErr':>10}{'viol%':>8}{'lat(ms)':>9}")
        for _, r in sub.iterrows():
            sp = "   n/a" if np.isnan(r['spill']) else f"{r['spill']:7.1f}"
            print(f"   {r['method']:<40}{r['rmse']:8.2f}{sp:>10}{r['viol']:8.2f}{r['lat']:9.3f}")
    if q_prec:
        print(f"\nSensor-quarantine precision={np.mean(q_prec):.2f} "
              f"recall={np.mean(q_rec):.2f} (compromised RSU/DET nodes)")
    print(f"\nWrote results to {RESULTS}")


if __name__ == "__main__":
    main()
