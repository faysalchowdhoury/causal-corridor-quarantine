"""End-to-end test-suite: physical bounds, attack injection, counterfactual
quarantine selection, reconstructor feasibility, and figure/CSV export."""
import copy
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sumo_env.corridor_sim import (CorridorCTM, FundamentalDiagram, load_config,
                                   write_sumo_network)
from cyber_attacks.attack_injector import (AttackInjector, SCENARIOS, rsu_node,
                                           det_node, sig_node, PLAZA_NODE)
from estimators.reconstructors import (CTMProcessModel, ExtendedKalmanReconstructor,
                                       UnconstrainedNeuralReconstructor,
                                       ConservationConstrainedReconstructor)
from causal_engine.scm_quarantine import CorridorSCM
import experiments.run_benchmarks as rb
import visualization.make_pub_figures as mpf

CFG = load_config()
RHO_JAM = CFG["fundamental_diagram"]["rho_jam_vehkm"]


# --------------------------------------------------------------------------- #
def test_fundamental_diagram_shape():
    fd = FundamentalDiagram(60.0, 140.0, 1800.0)
    assert abs(fd.flux(np.array([fd.rho_crit]))[0] - 1800.0) < 1e-6   # capacity at rho_crit
    assert fd.flux(np.array([0.0]))[0] == 0.0
    assert abs(fd.flux(np.array([fd.rho_jam]))[0]) < 1e-6             # zero flow at jam
    rho = np.linspace(0, 140, 50)
    assert np.all(fd.flux(rho) <= 1800.0 + 1e-6)


def test_ground_truth_respects_physical_bounds():
    for regime in SCENARIOS:
        inj = AttackInjector(CFG, regime)
        sim = CorridorCTM(CFG, seed=7)
        for _ in range(sim.n_steps):
            sim.step(inj.physical_override(sim.t))
            assert sim.rho.min() >= -1e-9
            assert sim.rho.max() <= RHO_JAM + 1e-9


def test_determinism():
    a = CorridorCTM(CFG, seed=3)
    b = CorridorCTM(CFG, seed=3)
    for _ in range(60):
        a.step(); b.step()
    assert np.allclose(a.rho, b.rho)


def test_attack_injection_changes_measurements():
    inj = AttackInjector(CFG, "compound")
    sim = CorridorCTM(CFG, seed=5)
    onset = CFG["attacks"]["onset_s"]
    for _ in range(int(onset / sim.dt_s) + 5):
        sim.step(inj.physical_override(sim.t))
    clean = sim.clean_measurements()
    sensed = inj.corrupt(clean, onset + 1.0)
    seg_a = CFG["attacks"]["scenario_A_rsu_spoof"]["target_segment"]
    seg_c = CFG["attacks"]["scenario_C_plaza_surge"]["target_segment"]
    assert sensed["rsu_speed"][seg_a] != clean["rsu_speed"][seg_a]      # spoofed speed
    assert sensed["det_rho"][seg_c] < clean["det_rho"][seg_c] + 1e-9    # under-reported
    # outside the attack window nothing is corrupted
    assert np.allclose(inj.corrupt(clean, 0.0)["rsu_speed"], clean["rsu_speed"])


def test_sumo_network_export(tmp_path):
    out = write_sumo_network(CFG, out_dir=str(tmp_path))
    for f in ("corridor.nod.xml", "corridor.edg.xml", "corridor.rou.xml", "corridor.sumocfg"):
        assert os.path.exists(os.path.join(out, f))


# --------------------------------------------------------------------------- #
def _fit_scm():
    X, Y, seg, grn, meas = rb.collect_training(CFG, [101, 102, 103])
    return CorridorSCM(CFG).fit(seg, grn, meas)


def test_causal_sensor_attribution():
    scm = _fit_scm()
    dt_s = CFG["corridor"]["dt_s"]
    onset_k = int(CFG["attacks"]["onset_s"] / dt_s)
    dur_k = int(CFG["attacks"]["duration_s"] / dt_s)
    z = float(CFG["experiment"]["sensor_quarantine_zthresh"])
    seg_a = CFG["attacks"]["scenario_A_rsu_spoof"]["target_segment"]
    seg_c = CFG["attacks"]["scenario_C_plaza_surge"]["target_segment"]
    # scenario A -> flags the spoofed RSU and not the detector
    d = rb.run_truth(CFG, 55, AttackInjector(CFG, "A"))
    win = [d["meas"][k] for k in range(onset_k, min(onset_k + dur_k, d["T"]))]
    flagged = scm.attribute_sensors(win, z)
    assert rsu_node(seg_a) in flagged and det_node(seg_a) not in flagged
    # scenario C -> flags the under-reporting detector
    d = rb.run_truth(CFG, 55, AttackInjector(CFG, "C"))
    win = [d["meas"][k] for k in range(onset_k, min(onset_k + dur_k, d["T"]))]
    assert det_node(seg_c) in scm.attribute_sensors(win, z)
    # no false positives under benign traffic
    d = rb.run_truth(CFG, 55, AttackInjector(CFG, "normal"))
    win = [d["meas"][k] for k in range(onset_k, min(onset_k + dur_k, d["T"]))]
    assert scm.attribute_sensors(win, z) == []


def test_counterfactual_min_regret_reduces_spillback():
    scm = _fit_scm()
    n = scm.n_seg
    r0 = np.full(n, 20.0)                 # light, uncongested corridor
    green_nom = np.full(n, scm.green_split)
    green_att = green_nom.copy()
    sig_t = CFG["attacks"]["scenario_B_green_starvation"]["target_intersection"]
    green_att[sig_t] = 0.18              # starved signal is the cause of the spillback
    disturb = {s: 0.0 for s in range(n)}
    br_none = scm.blast_radius(r0, green_att, disturb, horizon=60)
    br_quar = scm.blast_radius(r0, green_nom, disturb, horizon=60)   # do(quarantine SIG)
    assert br_quar < br_none             # restoring the starved signal relieves spillback


# --------------------------------------------------------------------------- #
def test_reconstructors_feasibility():
    scm = _fit_scm()
    pm = CTMProcessModel(CFG)
    X, Y, *_ = rb.collect_training(CFG, [201, 202])
    neural = UnconstrainedNeuralReconstructor(CFG); neural.fit(X, Y)
    cons = ConservationConstrainedReconstructor(CFG)
    ekf = ExtendedKalmanReconstructor(CFG)
    d = rb.run_truth(CFG, 77, AttackInjector(CFG, "compound"))
    mask, _ = rb.decide_quarantine(CFG, scm, pm, d)
    onset_k = int(CFG["attacks"]["onset_s"] / CFG["corridor"]["dt_s"])
    cons.reset(d["rho"][0]); ekf.reset(d["rho"][0]); neural.reset(d["rho"][0])
    cmin = cmax = None
    for k in range(d["T"]):
        qm = mask if k >= onset_k else None
        ec = cons.reconstruct(d["meas"][k], k * 2.0, qm)
        ee = ekf.reconstruct(d["meas"][k], k * 2.0)
        _ = neural.reconstruct(d["meas"][k], k * 2.0)
        cmin = ec.min() if cmin is None else min(cmin, ec.min())
        cmax = ec.max() if cmax is None else max(cmax, ec.max())
        assert -1e-6 <= ee.min() and ee.max() <= RHO_JAM + 1e-6      # EKF projects
    assert cmin >= -1e-6 and cmax <= RHO_JAM + 1e-6                  # ours stays feasible


def test_end_to_end_benchmark_and_figures(monkeypatch, tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["experiment"]["n_episodes"] = 1
    cfg["experiment"]["train_episodes"] = 2
    cfg["estimators"]["neural_epochs"] = 8
    monkeypatch.setattr(rb, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(mpf, "load_config", lambda *a, **k: cfg)
    rb.main()
    for f in ("benchmark_metrics.csv", "timeseries_sample.csv", "quarantine_comparison.csv"):
        assert os.path.exists(os.path.join(rb.RESULTS, f))
    mpf.main()
    for name in ("fig1_spatiotemporal_spillback", "fig2_quarantine_resilience",
                 "fig3_pareto_latency_accuracy", "fig4_conservation_residuals"):
        for ext in ("svg", "pdf", "jpg"):
            assert os.path.exists(os.path.join(mpf.FIGDIR, f"{name}.{ext}"))
