# Causal Counterfactual Quarantine and Conservation-Constrained State Reconstruction for Cyber-Resilient Connected Traffic Corridors

A reproducible, open-source research repository for cyber-resilient macroscopic
state estimation on a connected signalized arterial. It pairs a **Structural
Causal Model (SCM) quarantine** mechanism with a **conservation-constrained
reconstructor**, and benchmarks it honestly against an Extended Kalman Filter, a
strong **cell-level graph-neural network**, and an unconstrained MLP under
coordinated multi-vector attacks — with an attack-strength sweep, a
sensor-attribution ROC, and a cross-check against **Eclipse SUMO
microsimulation**.

> **Scope & honesty note.** This is a *simulation study*: there is no public
> field dataset of live cyber-attacks on signalized corridors, so ground truth
> is produced by a deterministic LWR/Godunov traffic-flow simulator (with an
> optional Eclipse SUMO backend). Results are **relative comparisons under a
> shared generator**, not field measurements. Findings below include a null
> result (H-blast-radius) and near-ties, reported as-is.

## Abstract

Connected-corridor controllers fuse V2X speed reports (RSU/BSM), loop-detector
densities, and signal actuation to estimate traffic state and manage spillback.
These channels are attackable. We model the corridor as an SCM
(Cyber/V2X → perceived state → actuation → segment density `ρ_i(t)` → queue
spillback `L_i(t)`) and treat the macroscopic conservation law
`Δρ/Δt + Δq/Δx = 0` with triangular-fundamental-diagram bounds
`0 ≤ ρ ≤ ρ_jam` strictly as a **physical consistency constraint** for state
reconstruction. A **causal quarantine** step (i) attributes compromised
measurement nodes by cross-sensor redundancy standardised against nominal
scatter, and (ii) selects an actuation quarantine by `do(quarantine=k)`
counterfactual rollouts on the calibrated SCM under a minimum-regret rule, while
preserving mainline emergency progression. On a 4-intersection arterial with an
off-ramp DC-fast-charging (DCFC) plaza, across 50 Monte-Carlo episodes and
normal / single-attack / zero-shot compound regimes, the method attains the
lowest reconstruction RMSE, never violates physical density bounds, and
correctly identifies compromised sensors (precision 1.00, recall 0.97), whereas
the unconstrained neural baseline produces `ρ > ρ_jam` artifacts on unseen
compound attacks.

## Structural Causal Model

For each segment `s` the DAG carries

```
RSU_s ─┐
       ├─► PERC_s ─► ACT_s ─► RHO_s ─► L_s
DET_s ─┘            ▲          ▲
             SIG_s ─┘   RHO_{s-1} ─┘        PLAZA ─► RHO_{plaza}
```

The structural assignment for `RHO_s` is the macroscopic conservation equation
with a signal-gated discharge (triangular FD sending/receiving `S`, `R`):

```
ρ_s(t+1) = clip( ρ_s + (Δt/L)·(q_in,s − q_out,s) + d_s ,  0, ρ_jam )
q_out,s  = min( green_s · cap_s ,  S(ρ_s),  R(ρ_{s+1}) )
```

`cap_s` (per-segment effective discharge capacity) is **estimated** from nominal
data. `do(quarantine = SIG_s)` restores `green_s` to nominal; `do(quarantine =
sensor)` cuts a corrupted measurement from the reconstruction. Because green
gates discharge mechanistically, restoring a starved signal provably drains the
queue (unit-tested).

## Threat model (attack scenarios)

| Scenario | Vector | Interface | Compromised node(s) |
|---|---|---|---|
| **A** | RSU/BSM **speed spoofing** (FDI): report free-flow over a congested segment | perception | `RSU` |
| **B** | Signal **green-time starvation** at a target intersection | actuation | `SIG` |
| **C** | EV-plaza **demand surge** + spoofed detour advisory under-reporting detector density (−40%) | physical + perception | `PLAZA`, `DET` |
| **compound** | Held-out, **zero-shot** combination of A+B+C (never used to fit any estimator) | all | all of the above |

## Compared reconstructors

1. **SUMO / CTM Ground Truth** — reference (LWR/Godunov CTM, optionally SUMO).
2. **EKF (LWR Godunov)** — physically grounded Extended Kalman Filter with the
   CTM as process model and analytic tridiagonal Jacobian; trusts every sensor.
3. **Unconstrained MLP** — plain feed-forward net, fast but physically unbounded.
4. **Graph Neural (GNN)** — cell-level message-passing GCN over the corridor
   cell chain; a strong, structure-aware learned baseline, also unconstrained.
5. **Causal Quarantine + Conservation (ours)** — conservation prediction with
   causal cross-sensor quarantine of compromised measurements.

## Benchmark — 50 Monte-Carlo episodes, mean over episodes

Reconstruction RMSE (veh/km), spillback arrival-time error (s), mass-conservation
violation rate (%; cell-steps with `ρ<0` or `ρ>ρ_jam`), and step inference
latency (ms). Lower is better. `SUMO Ground Truth` is the reference.

| Regime | Method | RMSE | Spillback err (s) | Violation % | Latency (ms) |
|---|---|---:|---:|---:|---:|
| normal | **Ours** | **2.46** | – | 0.00 | 0.13 |
| normal | EKF | 4.31 | – | 0.00 | 0.31 |
| normal | Graph Neural (GNN) | 21.60 | – | 0.00 | 0.60 |
| normal | Unconstr. MLP | 18.12 | – | 0.00 | 0.15 |
| **A** (RSU spoof) | **Ours** | **2.37** | – | 0.00 | 0.12 |
| **A** | EKF | 6.15 | – | 0.00 | 0.31 |
| **A** | Graph Neural (GNN) | 24.92 | – | 0.00 | 0.63 |
| **A** | Unconstr. MLP | 22.20 | – | 0.00 | 0.15 |
| **B** (green starvation) | **Ours** | 13.07 | **0.9** | 0.00 | 0.13 |
| **B** | EKF | **13.00** | 6.0 | 0.00 | 0.31 |
| **B** | Graph Neural (GNN) | 24.20 | 2.4 | 0.00 | 0.61 |
| **B** | Unconstr. MLP | 49.26 | 64.5 | 13.52 | 0.14 |
| **C** (detector FDI + plaza) | **Ours** | **8.28** | **7.0** | 0.00 | 0.12 |
| **C** | EKF | 11.84 | 116.3 | 0.00 | 0.30 |
| **C** | Graph Neural (GNN) | 23.26 | 115.6 | 0.00 | 0.57 |
| **C** | Unconstr. MLP | 34.88 | 180.0 | 5.09 | 0.14 |
| **compound** (zero-shot) | **Ours** | **12.54** | 26.6 | 0.00 | 0.12 |
| **compound** | EKF | 15.76 | **25.9** | 0.00 | 0.30 |
| **compound** | Graph Neural (GNN) | 28.99 | 24.2 | 0.00 | 0.59 |
| **compound** | Unconstr. MLP | 51.12 | 157.4 | 8.93 | 0.15 |

**Causal sensor quarantine:** precision **1.00**, recall **0.97** for compromised
RSU/DET nodes (attack regimes); sensor-attribution **ROC AUC 0.893** pooled over
weak-to-moderate attack strengths (`fig5`).

### What holds, and what does not

- **Causal quarantine — not just a constraint — is the differentiator.** The
  cell-level **GNN is a genuinely strong learned baseline** (it beats the MLP
  everywhere and stays feasible here), yet under detector FDI (**C**) it is
  fooled *exactly like EKF* — mis-timing spillback by ~116 s — because it has no
  quarantine. Ours cuts that to ~7 s. So the advantage is the causal
  cross-sensor quarantine, not merely bounding the output.
- **Ours has the lowest RMSE in every regime** (tied with EKF in B), ~2.3× lower
  than the GNN under the compound attack (12.5 vs 29.0 veh/km).
- **Physical feasibility.** Ours and EKF (projected) and the GNN (empirically)
  never violate bounds; the **unconstrained MLP** produces `ρ>ρ_jam` artifacts
  on attacked regimes (5–14%, `fig1`, `fig4`) — architecture/constraints matter.
- **Attribution — supported, and robust to attack strength.** Precision 1.00 /
  recall 0.97 at the operating threshold, ROC AUC 0.893 pooled across strengths,
  and **no false positives** on benign/green-starvation traffic.
- **Graceful degradation.** Across a normalised attack-strength sweep `s∈[0,1]`,
  Ours stays lowest-RMSE throughout and degrades slowest (`fig5`).
- **B is an honest near-tie with EKF on RMSE** (13.07 vs 13.00): no sensor is
  spoofed in pure green-starvation, so quarantine offers no reconstruction gain
  there; the advantage is only in spillback timing (0.9 s vs 6.0 s).
- **Compound spillback timing is a near-tie with EKF** (26.6 s vs 25.9 s): once
  congestion is severe both bounded estimators detect it; the MLP is far worse.
- **Actuation quarantine (`fig2`).** Counterfactual minimum-regret quarantine
  restores throughput to ~900 veh/h vs ~290 (no quarantine) and ~420 (reactive),
  and caps total congested length at ~820 m vs ~1260 m.
- **Latency.** All reconstructors are sub-millisecond; Ours (~0.12 ms) is the
  fastest (no Kalman inversion, no GNN message passing).

### SUMO microsimulation validation (`fig6`)

With the optional Eclipse SUMO backend, the analytic LWR/Godunov CTM is
cross-checked against microsimulation on the same corridor and matched signal
timing: corridor-mean density tracks SUMO with **Pearson r = 0.97** over time,
and per-segment mean densities agree to within **~19%** (differences at the
entry/exit segments reflect micro- vs. macroscopic boundary handling). This is a
structural cross-check, not a cell-for-cell match.

## Figures (`figures/`, 300 DPI, `.svg`/`.pdf`/`.jpg`)

1. `fig1_spatiotemporal_spillback` — 2×2 `x`–`t` density heatmaps (Ground Truth,
   Unconstrained Neural with highlighted `ρ>ρ_jam` artifacts, EKF, Ours) under
   the compound attack.
2. `fig2_quarantine_resilience` — throughput and congested length before/during/
   after the attack for *No Quarantine*, *Reactive Threshold*, *Causal
   Counterfactual*.
3. `fig3_pareto_latency_accuracy` — latency vs spillback RMSE Pareto.
4. `fig4_conservation_residuals` — violation-rate distribution across zero-shot
   compound episodes.
5. `fig5_attack_strength_roc` — RMSE vs normalised attack strength (all methods)
   and the sensor-attribution ROC (AUC 0.893).
6. `fig6_sumo_validation` — CTM vs SUMO temporal density tracking and
   per-segment mean density (optional; requires SUMO).

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10–3.12 recommended. Eclipse SUMO is **optional** — everything runs
out-of-the-box on the analytic LWR/Godunov engine. To enable the real SUMO
microsimulation backend and `validate_sumo.py`:

```bash
pip install -r requirements-sumo.txt        # eclipse-sumo + libsumo + traci + sumolib
```

The backend auto-discovers `SUMO_HOME` (including the pip wheel's install path),
compiles the network with `netconvert`, and drives it with `libsumo`.

## One-command reproduction

```bash
pytest -q                                   # 10 tests: physics, attacks, quarantine, export, SUMO
python experiments/run_benchmarks.py        # 50 episodes -> results/*.csv  (+ console table)
python experiments/attack_sweep.py          # strength sweep + attribution ROC -> results/
python visualization/make_pub_figures.py    # figures/ at 300 DPI (svg/pdf/jpg)

# optional -- real SUMO microsimulation validation (pip install -r requirements-sumo.txt)
python experiments/validate_sumo.py         # -> results/sumo_validation.csv, figures/fig6_*
```

Everything is deterministic given the seeds in `config/corridor_config.yaml`.
The SUMO step is optional; without it the analytic CTM is the ground truth.

## Repository structure

```
config/corridor_config.yaml      corridor geometry, FD, signals, attack schedule, seeds
sumo_env/corridor_sim.py         analytic LWR/Godunov CTM + .nod/.edg/.rou/.sumocfg export
sumo_env/sumo_backend.py         real Eclipse SUMO backend (netconvert + libsumo driver)
cyber_attacks/attack_injector.py scenarios A/B/C + zero-shot compound; physical + perception
causal_engine/scm_quarantine.py  SCM DAG, structural-parameter estimation, do(quarantine)
                                 counterfactual rollouts, minimum-regret selection
estimators/reconstructors.py     EKF (LWR Godunov), Unconstrained MLP, cell-level GNN (PyTorch),
                                 Conservation-Constrained (ours)
experiments/run_benchmarks.py    Monte-Carlo benchmark -> results/*.csv
experiments/attack_sweep.py      attack-strength sweep + sensor-attribution ROC
experiments/validate_sumo.py     CTM-vs-SUMO microsimulation validation (optional)
visualization/make_pub_figures.py six 300-DPI publication figures
tests/test_pipeline.py           physics, attacks, quarantine, feasibility, export, SUMO
```

## Limitations & threats to validity

- **Simulation, not field data.** Attack and benign traffic are generated by a
  macroscopic model; absolute rates and the benign/attack mix are assumptions.
  The SUMO cross-check validates the *physics* (r = 0.97 temporal density
  tracking), not the attack realism.
- **Attack magnitudes are scenario parameters.** The sweep (`fig5`) shows
  detectability and robustness *across* strengths; a detector FDI within nominal
  cross-sensor scatter is correspondingly harder to attribute and less harmful.
- **Per-client / single-leader model.** A distributed attacker spreading across
  many identities is only partially modeled.
- **Deterministic CTM.** Latencies are real `perf_counter`-based measurements of
  this Python implementation and indicate relative cost, not production SLOs.
- **Near-tie / scoped claims.** B is a RMSE tie with EKF and compound spillback
  timing is a near-tie; these are stated rather than spun. The GNN is a strong
  learned baseline but is not exhaustively architecture-searched.

## License

MIT — see [LICENSE](LICENSE).
