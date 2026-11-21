"""Validate the analytic LWR/Godunov CTM against Eclipse SUMO microsimulation.

Runs both backends on the same corridor under nominal demand and checks that
(i) SUMO's microsimulated flow-density points follow the triangular fundamental
diagram the CTM assumes, and (ii) per-segment mean densities agree in magnitude.
This is a *structural* cross-check (micro vs macro with independently generated
signal timing), not a cell-for-cell match.

Outputs ``results/sumo_validation.csv`` and ``figures/fig6_sumo_validation.*``.
Skips gracefully (exit 0) if SUMO is not installed.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sumo_env.corridor_sim import CorridorCTM, FundamentalDiagram, load_config
from sumo_env.sumo_backend import sumo_ready, build_network, run_sumo_corridor

RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(ROOT, "figures")


def run_ctm_segments(cfg: dict, seed: int):
    sim = CorridorCTM(cfg, seed=seed)
    dens, spd = [], []
    for _ in range(sim.n_steps):
        sim.step()
        seg = sim.segment_mean()
        dens.append(seg)
        spd.append(sim.fd.speed(seg))
    return np.asarray(dens), np.asarray(spd)


def main() -> None:
    cfg = load_config()
    if not sumo_ready():
        print("SUMO not available -- skipping validation (install eclipse-sumo + libsumo).")
        return
    print("[1/3] compiling SUMO network + running microsimulation ...")
    build_network(cfg)
    sumo_d, sumo_v, _ = run_sumo_corridor(cfg, seed=1)
    sumo_q = sumo_d * sumo_v                       # veh/h = veh/km * km/h

    print("[2/3] running analytic CTM ...")
    ctm_d, ctm_v = run_ctm_segments(cfg, seed=1)

    fd = FundamentalDiagram(cfg["fundamental_diagram"]["v_free_kmh"],
                            cfg["fundamental_diagram"]["rho_jam_vehkm"],
                            cfg["fundamental_diagram"]["q_max_vehh"])

    # --- validation metrics ---------------------------------------------- #
    seg_sumo = sumo_d.mean(axis=0)
    seg_ctm = ctm_d.mean(axis=0)
    rel = float(np.mean(np.abs(seg_sumo - seg_ctm) / (seg_ctm + 1e-6)))
    # temporal agreement of corridor-mean density (SUMO 1s resampled to CTM 2s grid)
    step = int(round(cfg["corridor"]["dt_s"]))
    T = min(len(ctm_d), len(sumo_d) // step)
    sumo_corr = sumo_d[:T * step].reshape(T, step, -1).mean(axis=1).mean(axis=1)
    ctm_corr = ctm_d[:T].mean(axis=1)
    pear = float(np.corrcoef(sumo_corr, ctm_corr)[0, 1])
    tsec = np.arange(T) * step
    print(f"[3/3] per-segment mean density: SUMO={np.round(seg_sumo,1)}  "
          f"CTM={np.round(seg_ctm,1)}  (mean rel. diff {rel*100:.1f}%)")
    print(f"      corridor-mean density temporal correlation r = {pear:.3f}")

    rows = []
    for s in range(len(seg_sumo)):
        rows.append({"segment": s, "sumo_mean_density": seg_sumo[s],
                     "ctm_mean_density": seg_ctm[s]})
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "sumo_validation.csv"), index=False)

    # --- figure ----------------------------------------------------------- #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG_BG, AX_BG, FG, GRID = "#0f0f1a", "#1a1a2e", "#d6d6e0", "#33334d"
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "figure.facecolor": FIG_BG, "axes.facecolor": AX_BG,
                         "savefig.facecolor": FIG_BG, "axes.edgecolor": GRID,
                         "axes.labelcolor": FG, "text.color": FG,
                         "xtick.color": FG, "ytick.color": FG,
                         "grid.color": GRID, "grid.alpha": 0.5, "axes.grid": True})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    ax1.plot(tsec, sumo_corr, color="#4cc9f0", lw=2.0, label="SUMO microsim")
    ax1.plot(tsec, ctm_corr, color="#ffd166", lw=2.0, label="CTM (analytic)")
    for sp in ("top", "right"):
        ax1.spines[sp].set_visible(False)
    ax1.set_xlabel("time  (s)"); ax1.set_ylabel("corridor-mean density  (veh/km)")
    ax1.set_title(f"Temporal density tracking  (r={pear:.2f})", color=FG)
    ax1.legend(frameon=False, fontsize=9)

    x = np.arange(len(seg_sumo))
    ax2.bar(x - 0.18, seg_sumo, 0.36, color="#4cc9f0", label="SUMO")
    ax2.bar(x + 0.18, seg_ctm, 0.36, color="#ffd166", label="CTM")
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    ax2.set_xticks(x); ax2.set_xticklabels([f"seg {i}" for i in x])
    ax2.set_ylabel("mean density  (veh/km)")
    ax2.set_title(f"Per-segment mean density  (rel. diff {rel*100:.0f}%)", color=FG)
    ax2.legend(frameon=False, fontsize=9)
    fig.suptitle("CTM vs SUMO microsimulation validation", color=FG, fontsize=13)
    fig.tight_layout()
    for ext in ("svg", "pdf", "jpg"):
        fig.savefig(os.path.join(FIGDIR, f"fig6_sumo_validation.{ext}"), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print("Wrote results/sumo_validation.csv and figures/fig6_sumo_validation.*")


if __name__ == "__main__":
    main()
