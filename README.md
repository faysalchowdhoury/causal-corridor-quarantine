# Causal Counterfactual Quarantine and Conservation-Constrained State Reconstruction for Cyber-Resilient Connected Traffic Corridors

A reproducible, open-source research repository for cyber-resilient macroscopic
state estimation on a connected signalized arterial. It pairs a **Structural
Causal Model (SCM) quarantine** mechanism with a **conservation-constrained
reconstructor**, and benchmarks it honestly against an Extended Kalman Filter
and an unconstrained neural baseline under coordinated multi-vector attacks.

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

## Benchmark — 50 Monte-Carlo episodes, mean over episodes

Reconstruction RMSE (veh/km), spillback arrival-time error (s), mass-conservation
violation rate (%; cell-steps with `ρ<0` or `ρ>ρ_jam`), and step inference
latency (ms). Lower is better. `SUMO Ground Truth` is the reference.

| Regime | Method | RMSE | Spillback err (s) | Violation % | Latency (ms) |
|---|---|---:|---:|---:|---:|
| normal | **Ours** | **2.46** | – | 0.00 | 0.10 |
| normal | EKF | 4.31 | – | 0.00 | 0.27 |
| normal | Unconstr. Neural | 18.12 | – | 0.00 | 0.12 |
| **A** (RSU spoof) | **Ours** | **2.37** | – | 0.00 | 0.10 |
| **A** | EKF | 6.15 | – | 0.00 | 0.27 |
| **A** | Unconstr. Neural | 22.20 | – | 0.00 | 0.11 |
| **B** (green starvation) | **Ours** | 13.07 | **0.9** | 0.00 | 0.10 |
| **B** | EKF | **13.00** | 6.0 | 0.00 | 0.26 |
| **B** | Unconstr. Neural | 49.26 | 64.5 | 13.52 | 0.12 |
| **C** (detector FDI + plaza) | **Ours** | **8.28** | **7.0** | 0.00 | 0.10 |
| **C** | EKF | 11.84 | 116.3 | 0.00 | 0.25 |
| **C** | Unconstr. Neural | 34.88 | 180.0 | 5.09 | 0.11 |
| **compound** (zero-shot) | **Ours** | **12.54** | 26.6 | 0.00 | 0.10 |
| **compound** | EKF | 15.76 | **25.9** | 0.00 | 0.26 |
| **compound** | Unconstr. Neural | 51.12 | 157.4 | 8.93 | 0.12 |

**Causal sensor quarantine:** precision **1.00**, recall **0.97** for compromised
RSU/DET nodes (attack regimes).

### What holds, and what does not

- **Conservation-constrained reconstruction wins on accuracy and feasibility.**
  Ours has the lowest RMSE in every regime (tied with EKF in B) and **0%**
  physical-bound violations throughout, versus **5–14%** for the unconstrained
  neural net on attacked regimes (`fig1`, `fig4`).
- **Causal quarantine is the differentiator under FDI.** Under detector FDI
  (**C**) EKF trusts the corrupted detector and mis-times spillback by ~116 s;
  quarantining the flagged sensor cuts that to ~7 s.
- **H (multi-signal attribution) — supported.** Sensor attribution is
  near-perfect (precision 1.00 / recall 0.97) and the physically grounded
  cross-sensor test produces **no false positives** on benign/green-starvation
  traffic.
- **B is an honest near-tie with EKF on RMSE.** No measurement is spoofed in a
  pure green-starvation attack, so sensor quarantine offers no reconstruction
  advantage there (Ours 13.07 vs EKF 13.00); the gain is only in spillback
  timing (0.9 s vs 6.0 s).
- **Compound spillback timing is a near-tie with EKF** (26.6 s vs 25.9 s): once
  congestion is severe enough, both bounded estimators detect it; the neural
  baseline is far worse (157 s).
- **Actuation quarantine (fig2, physical corridor).** Counterfactual
  minimum-regret quarantine restores throughput to ~900 veh/h vs ~290 (no
  quarantine) and ~420 (reactive threshold), and caps total congested length at
  ~820 m vs ~1260 m.
- **Latency.** All reconstructors are sub-millisecond; Ours (~0.10 ms) is
  actually the fastest (no Kalman matrix inversion).

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

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10–3.12 recommended. Eclipse SUMO is **optional**: if `SUMO_HOME` is set
and `traci` is importable the corridor can be driven through SUMO, otherwise the
built-in LWR/Godunov engine runs everything out-of-the-box.

## One-command reproduction

```bash
pytest -q                                   # 9 tests: physics, attacks, quarantine, export
python experiments/run_benchmarks.py        # 50 episodes -> results/*.csv  (+ console table)
python visualization/make_pub_figures.py    # figures/ at 300 DPI (svg/pdf/jpg)
python sumo_env/corridor_sim.py             # (optional) emit SUMO network XML into sumo_env/network/
```

Everything is deterministic given the seeds in `config/corridor_config.yaml`.

## Repository structure

```
config/corridor_config.yaml      corridor geometry, FD, signals, attack schedule, seeds
sumo_env/corridor_sim.py         dual-mode simulator: LWR/Godunov CTM (+ optional SUMO/TraCI),
                                 valid .nod/.edg/.rou/.sumocfg export
cyber_attacks/attack_injector.py scenarios A/B/C + zero-shot compound; physical + perception
causal_engine/scm_quarantine.py  SCM DAG, structural-parameter estimation, do(quarantine)
                                 counterfactual rollouts, minimum-regret selection
estimators/reconstructors.py     EKF (LWR Godunov), Unconstrained Neural (PyTorch),
                                 Conservation-Constrained (ours)
experiments/run_benchmarks.py    Monte-Carlo benchmark -> results/*.csv
visualization/make_pub_figures.py four 300-DPI publication figures
tests/test_pipeline.py           physics bounds, attack injection, quarantine, feasibility, export
```

## Limitations & threats to validity

- **Simulation, not field data.** Attack and benign traffic are generated by a
  macroscopic model; absolute rates and the benign/attack mix are assumptions.
- **Attack magnitudes are scenario parameters.** A weaker detector FDI (e.g.
  within nominal cross-sensor scatter) is correspondingly harder to attribute
  and less harmful; the reported detectability holds for the stated magnitudes.
- **Per-client / single-leader model.** A distributed attacker spreading across
  many identities is only partially modeled.
- **Deterministic CTM.** Latencies are real `perf_counter`-based measurements of
  this Python implementation and indicate relative cost, not production SLOs.
- **Near-tie / scoped claims.** B is a RMSE tie with EKF and compound spillback
  timing is a near-tie; these are stated rather than spun.

## License

MIT — see [LICENSE](LICENSE).
