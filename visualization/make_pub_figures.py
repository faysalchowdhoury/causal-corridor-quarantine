"""Publication-grade figures (dark theme, Times New Roman, 300 DPI).

Generates four figures in ``.svg``, ``.pdf`` and ``.jpg`` inside ``figures/``
from the CSVs written by ``experiments/run_benchmarks.py``.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

try:
    import seaborn as sns
    _SNS = True
except Exception:
    _SNS = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sumo_env.corridor_sim import load_config  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(ROOT, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# --- theme ----------------------------------------------------------------- #
FIG_BG, AX_BG = "#0f0f1a", "#1a1a2e"
FG, GRID = "#d6d6e0", "#33334d"
C_GT, C_EKF, C_OURS, C_NEURAL, C_GNN = "#c9c9d6", "#4cc9f0", "#ffd166", "#ff6b6b", "#b983ff"
METHOD_COLOR = {
    "SUMO Ground Truth": C_GT,
    "EKF (LWR Godunov)": C_EKF,
    "Unconstrained Neural": C_NEURAL,
    "Graph Neural (GNN)": C_GNN,
    "Causal Quarantine + Conservation (ours)": C_OURS,
}
SHORT = {
    "SUMO Ground Truth": "Ground Truth",
    "EKF (LWR Godunov)": "EKF",
    "Unconstrained Neural": "Unconstr. Neural",
    "Graph Neural (GNN)": "GNN",
    "Causal Quarantine + Conservation (ours)": "Ours",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "figure.facecolor": FIG_BG, "axes.facecolor": AX_BG,
    "savefig.facecolor": FIG_BG, "axes.edgecolor": GRID,
    "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": FG, "ytick.color": FG,
    "grid.color": GRID, "grid.alpha": 0.5, "grid.linewidth": 0.6,
    "axes.grid": True, "axes.axisbelow": True, "font.size": 11,
})


def _save(fig, name: str) -> None:
    for ext in ("svg", "pdf", "jpg"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name + ".{svg,pdf,jpg}")


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return ax


# --------------------------------------------------------------------------- #
def fig1_spatiotemporal(cfg):
    df = pd.read_csv(os.path.join(RESULTS, "timeseries_sample.csv"))
    rho_jam = cfg["fundamental_diagram"]["rho_jam_vehkm"]
    cols = [("rho_gt", "Ground Truth", C_GT),
            ("rho_neural", "Unconstrained Neural", C_NEURAL),
            ("rho_ekf", "EKF (LWR Godunov)", C_EKF),
            ("rho_ours", "Causal Quarantine + Conservation (ours)", C_OURS)]
    cmap = LinearSegmentedColormap.from_list("rho", ["#10213a", "#2a6f97", "#ffd166", "#e63946"])
    cmap = cmap.with_extremes(over="#ff00e6")   # infeasible rho > rho_jam highlighted in magenta
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True, sharey=True)
    for ax, (col, title, c) in zip(axs.ravel(), cols):
        grid = df.pivot(index="t", columns="x_m", values=col)
        im = ax.pcolormesh(grid.columns.values, grid.index.values, grid.values,
                           cmap=cmap, vmin=0, vmax=rho_jam, shading="auto", rasterized=True)
        over = 100.0 * np.mean(grid.values > rho_jam)
        tag = f"  (>{rho_jam:.0f}: {over:.1f}%)" if over > 0.05 else ""
        ax.set_title(title.replace(" + ", "\n+ ") + tag, color=c, fontsize=11, pad=6)
        ax.set_facecolor(AX_BG)
    for ax in axs[1]:
        ax.set_xlabel("corridor position  x  (m)")
    for ax in axs[:, 0]:
        ax.set_ylabel("time  t  (s)")
    cb = fig.colorbar(im, ax=axs, fraction=0.046, pad=0.03, extend="max")
    cb.set_label(r"density  $\rho$  (veh/km)")
    cb.outline.set_edgecolor(GRID)
    fig.suptitle("Spatiotemporal density under the zero-shot compound attack",
                 color=FG, fontsize=13, y=0.98)
    _save(fig, "fig1_spatiotemporal_spillback")


def fig2_quarantine_resilience(cfg):
    df = pd.read_csv(os.path.join(RESULTS, "quarantine_comparison.csv"))
    onset = cfg["attacks"]["onset_s"]
    styles = {"No Quarantine": ("#ff6b6b", "-"),
              "Reactive Threshold": ("#ffd166", "--"),
              "Causal Counterfactual": ("#4cc9f0", "-")}
    win = max(1, int(cfg["signals"]["cycle_s"] / cfg["corridor"]["dt_s"]))  # 1-cycle smoothing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for strat, (c, ls) in styles.items():
        s = df[df.strategy == strat].sort_values("t")
        thru = s.throughput_vehh.rolling(win, center=True, min_periods=1).mean()
        queue = s.total_queue_m.rolling(win, center=True, min_periods=1).mean()
        lw = 2.6 if strat == "Causal Counterfactual" else 1.8
        ax1.plot(s.t, thru, color=c, ls=ls, lw=lw, label=strat)
        ax2.plot(s.t, queue, color=c, ls=ls, lw=lw, label=strat)
    for ax in (ax1, ax2):
        ax.axvline(onset, color=FG, ls=":", lw=1.2, alpha=0.7)
        _style(ax)
        ax.set_xlabel("time  t  (s)")
        ax.text(onset + 6, ax.get_ylim()[1] * 0.05, "attack onset", color=FG,
                fontsize=9, alpha=0.8)
    ax1.set_ylabel("corridor throughput  (veh/h, 1-cycle mean)")
    ax2.set_ylabel("total congested length  (m)")
    ax1.set_title("Throughput resilience", color=FG)
    ax2.set_title("Queue containment", color=FG)
    ax1.legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("Corridor resilience under compound attack: quarantine strategy",
                 color=FG, fontsize=13)
    _save(fig, "fig2_quarantine_resilience")


def fig3_pareto(cfg):
    df = pd.read_csv(os.path.join(RESULTS, "benchmark_metrics.csv"))
    sub = df[df.regime == "compound"]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for method, g in sub.groupby("method"):
        if method == "SUMO Ground Truth":
            continue
        x, y = g.latency_ms.mean(), g.rmse_vehkm.mean()
        xe, ye = g.latency_ms.std(), g.rmse_vehkm.std()
        c = METHOD_COLOR[method]
        ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="o", ms=11, color=c, ecolor=c,
                    elinewidth=1.3, capsize=4, mec="white", mew=0.7, label=SHORT[method])
        ax.annotate(SHORT[method], (x, y), textcoords="offset points",
                    xytext=(10, 8), color=c, fontsize=10)
    ax.set_xscale("log")
    _style(ax)
    ax.set_xlabel("step inference latency  (ms, log scale)")
    ax.set_ylabel("spillback RMSE  (veh/km)")
    ax.set_title("Accuracy-latency Pareto under compound attack\n(lower-left is better)",
                 color=FG)
    ax.annotate("Pareto-preferred", xy=(0.04, 0.10), xycoords="axes fraction",
                color=C_OURS, fontsize=10)
    _save(fig, "fig3_pareto_latency_accuracy")


def fig4_conservation(cfg):
    df = pd.read_csv(os.path.join(RESULTS, "benchmark_metrics.csv"))
    sub = df[(df.regime == "compound") & (df.method != "SUMO Ground Truth")].copy()
    sub["m"] = sub.method.map(SHORT)
    order = ["EKF", "Unconstr. Neural", "GNN", "Ours"]
    colors = [C_EKF, C_NEURAL, C_GNN, C_OURS]
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    data = [sub[sub.m == o].mass_violation_pct.values for o in order]
    if _SNS:
        sns.violinplot(data=data, ax=ax, palette=colors, inner="box", cut=0, linewidth=1.0)
    else:
        parts = ax.violinplot(data, showmeans=False, showmedians=True)
        for pc, c in zip(parts["bodies"], colors):
            pc.set_facecolor(c); pc.set_alpha(0.65)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    for i, d in enumerate(data):
        ax.scatter(np.full_like(d, i) + np.random.uniform(-0.05, 0.05, len(d)), d,
                   s=12, color="white", alpha=0.35, zorder=3)
    _style(ax)
    ax.set_ylabel("mass-conservation violation rate  (%)")
    ax.set_title("Physical feasibility under zero-shot compound attacks\n"
                 r"(% of cell-steps with $\rho<0$ or $\rho>\rho_{jam}$)", color=FG)
    _save(fig, "fig4_conservation_residuals")


def fig5_attack_sweep_roc(cfg):
    sweep_p = os.path.join(RESULTS, "attack_sweep.csv")
    roc_p = os.path.join(RESULTS, "roc.csv")
    if not (os.path.exists(sweep_p) and os.path.exists(roc_p)):
        print("  (skipping fig5: run experiments/attack_sweep.py first)")
        return
    sweep = pd.read_csv(sweep_p)
    roc = pd.read_csv(roc_p)
    auc_p = os.path.join(RESULTS, "roc_auc.txt")
    auc = float(open(auc_p).read().strip()) if os.path.exists(auc_p) else None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for method, g in sweep.groupby("method"):
        stat = g.groupby("strength").rmse_vehkm.agg(["mean", "std"]).reset_index()
        c = METHOD_COLOR.get(method, FG)
        ax1.plot(stat.strength, stat["mean"], "-o", ms=5, color=c, lw=2, label=SHORT.get(method, method))
        ax1.fill_between(stat.strength, stat["mean"] - stat["std"], stat["mean"] + stat["std"],
                         color=c, alpha=0.15)
    _style(ax1)
    ax1.set_xlabel("normalised attack strength  s")
    ax1.set_ylabel("reconstruction RMSE  (veh/km)")
    ax1.set_title("Robustness vs attack strength", color=FG)
    ax1.legend(frameon=False, fontsize=9, loc="upper left")

    ax2.plot([0, 1], [0, 1], ls=":", color=GRID, lw=1.2)
    ax2.plot(roc.fpr, roc.tpr, "-", color=C_OURS, lw=2.4,
             label=(f"causal quarantine (AUC={auc:.3f})" if auc else "causal quarantine"))
    ax2.scatter(roc.fpr, roc.tpr, s=16, color=C_OURS, zorder=3)
    _style(ax2)
    ax2.set_xlabel("false-positive rate  (benign sensors flagged)")
    ax2.set_ylabel("true-positive rate  (attacks caught)")
    ax2.set_title("Sensor-attribution ROC\n(pooled over attack strengths)", color=FG)
    ax2.legend(frameon=False, fontsize=9, loc="lower right")
    ax2.set_xlim(-0.02, 1.02); ax2.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    _save(fig, "fig5_attack_strength_roc")


def main():
    cfg = load_config()
    print("Generating publication figures (300 DPI, svg/pdf/jpg) ...")
    fig1_spatiotemporal(cfg)
    fig2_quarantine_resilience(cfg)
    fig3_pareto(cfg)
    fig4_conservation(cfg)
    fig5_attack_sweep_roc(cfg)
    print("Figures written to", FIGDIR)


if __name__ == "__main__":
    main()
