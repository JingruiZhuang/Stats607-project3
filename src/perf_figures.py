"""Figures for the performance study: complexity, speedup and scaling.

Shares the palette and styling of ``src.figures`` so that the performance
figures read as part of the same report, and draws only from the CSVs written
by ``src.complexity``, ``src.benchmark`` and ``src.stability`` -- no figure
here re-runs the simulation, so a plot can be redrawn without a fresh sweep.
"""

import argparse
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import bench_config, complexity, config
from .figures import GRID, INK, INK_SOFT, SURFACE, style

logger = logging.getLogger(__name__)

# Implementation identity gets its own two colours, chosen from the same
# validated categorical palette as src.figures but kept distinct from the
# estimator-family colours so the two sets of figures cannot be confused.
TARGET_COLOUR = {"baseline": "#8a6a4a", "optimized": "#1baf7a"}
TARGET_LABEL = {"baseline": "baseline (Unit 2)", "optimized": "optimized"}

# Components are told apart by line style rather than by colour, so that colour
# stays free to mean the implementation.
COMPONENT_STYLE = {
    "total": ("-", 1.9, "total"),
    "method": ((0, (5, 2)), 1.4, "estimators"),
    "dgp": ((0, (1, 2)), 1.4, "DGP"),
}


def _frame(path, what):
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found; run `make {what}` first")
    return pd.read_csv(path)


def plot_complexity(timings, fits, out_path, at_n=None):
    """Runtime per replicate against n, log-log, with fitted exponents.

    Log-log because a power law is a straight line there, so the slope reads
    off directly as the empirical complexity.  A dashed vertical rule marks the
    study's own n, which is the point the whole exercise is about: if the
    curves are still flat there, the runtime at the design point is overhead
    rather than arithmetic, and no amount of faster arithmetic will help.
    """
    style()
    scenarios = list(dict.fromkeys(timings["scenario"]))
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5.4 * len(scenarios), 4.6),
                             sharey=True)
    axes = np.atleast_1d(axes)
    at_n = at_n if at_n is not None else config.N

    columns = [("seconds_per_replicate", "total"),
               ("method_per_replicate", "method"),
               ("dgp_per_replicate", "dgp")]

    for ax, scenario in zip(axes, scenarios):
        panel = timings[timings["scenario"] == scenario]
        for target, group in panel.groupby("target", sort=False):
            group = group.sort_values("n")
            for column, component in columns:
                if column not in group or group[column].isna().all():
                    continue
                dash, width, _ = COMPONENT_STYLE[component]
                ax.plot(group["n"], group[column],
                        color=TARGET_COLOUR.get(target, INK), linestyle=dash,
                        linewidth=width, marker="o" if component == "total" else None,
                        markersize=3.4, markeredgecolor="none", zorder=3)

                # Annotate the slope over the upper half of the grid, which is
                # where the asymptotic rate is visible.
                row = fits[(fits.target == target) & (fits.scenario == scenario)
                           & (fits.component == component)
                           & (fits.min_n == fits.min_n.max())]
                if component == "total" and not row.empty:
                    ax.annotate(f"p = {row.exponent.iloc[0]:.2f}",
                                xy=(group["n"].iloc[-1], group[column].iloc[-1]),
                                xytext=(4, -2), textcoords="offset points",
                                color=TARGET_COLOUR.get(target, INK), fontsize=8,
                                va="top")

        ax.axvline(at_n, color=INK, linewidth=1.0, linestyle=(0, (4, 3)), zorder=0)
        ax.annotate(f"study n = {at_n}", xy=(at_n, 0.02),
                    xycoords=("data", "axes fraction"), xytext=(4, 0),
                    textcoords="offset points", color=INK_SOFT, fontsize=8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="major", color=GRID, linewidth=0.6, zorder=0)
        ax.grid(True, which="minor", color=GRID, linewidth=0.3, alpha=0.6, zorder=0)
        ax.set_title(f"scenario {scenario}: "
                     f"{config.SCENARIOS[scenario]['label']}", loc="left")
        ax.set_xlabel("sample size n")
    axes[0].set_ylabel("seconds per replicate")

    handles = [plt.Line2D([], [], color=TARGET_COLOUR[t], linewidth=2.0,
                          label=TARGET_LABEL[t])
               for t in dict.fromkeys(timings["target"]) if t in TARGET_COLOUR]
    handles += [plt.Line2D([], [], color=INK_SOFT, linestyle=d, linewidth=w,
                           label=label)
                for d, w, label in COMPONENT_STYLE.values()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Cost of one replicate against sample size, log-log; "
                 "the slope is the empirical complexity",
                 x=0.011, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_path)


def plot_overhead(timings, out_path, at_n=None):
    """The fixed cost of a replicate, as a share of its cost at each n.

    The affine fit t = c + a n splits a replicate into a part that scales with
    the data and a part that does not.  Plotted as a share, it answers the
    question an optimizer needs answered before choosing a strategy: at the n
    this study actually runs at, is there more to gain from faster arithmetic
    or from less Python?
    """
    style()
    at_n = at_n if at_n is not None else config.N
    scenarios = list(dict.fromkeys(timings["scenario"]))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    grid = np.geomspace(timings["n"].min(), timings["n"].max(), 200)
    for scenario in scenarios:
        panel = timings[(timings["scenario"] == scenario)
                        & (timings["target"] == "baseline")].sort_values("n")
        fit = complexity.fit_affine(panel["n"], panel["seconds_per_replicate"], at_n)
        fixed = fit["fixed_seconds"]
        share = 100.0 * fixed / (fixed + fit["seconds_per_observation"] * grid)
        line, = ax.plot(grid, share, linewidth=1.9,
                        label=f"scenario {scenario}")
        ax.plot([at_n], [100.0 * fit["fixed_share_at_n"]], marker="o",
                markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.2,
                color=line.get_color(), zorder=4)
        ax.annotate(f"{100 * fit['fixed_share_at_n']:.0f}% fixed at n = {at_n}",
                    xy=(at_n, 100.0 * fit["fixed_share_at_n"]),
                    xytext=(8, 6), textcoords="offset points", fontsize=8,
                    color=line.get_color())

    ax.axvline(at_n, color=INK, linewidth=1.0, linestyle=(0, (4, 3)), zorder=0)
    ax.set_xscale("log")
    ax.set_ylim(0, 100)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_xlabel("sample size n")
    ax.set_ylabel("share of a replicate that does not scale with n (%)")
    ax.legend(frameon=False, loc="lower left")
    fig.suptitle("Most of a replicate at the study's n is fixed overhead, "
                 "not arithmetic", x=0.011, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_path)


def plot_stability(summary, out_path):
    """What the estimators are exposed to numerically, against n.

    Left: the largest inverse-probability weight in a replicate -- the median
    over replicates, with the worst replicate shaded above it, because a
    worst-case read off 200 replicates is itself noisy.  Right: the effective
    sample size the weighting leaves behind, as a share of n.

    Together they locate this study's numerical risk.  Nothing in the baseline
    ever warns, yet in scenario B with the correct propensity model a typical
    replicate contains a weight of order 20 and the weighting discards some
    85% of the sample -- so a single draw can move a Monte Carlo mean, and no
    diagnostic in the submitted code would say so.
    """
    style()
    panel = summary[summary["target"] == "baseline"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    colour = {"A": "#2a78d6", "B": "#eb6834"}
    for (scenario, ps_model), group in panel.groupby(["scenario", "ps_model"],
                                                     sort=False):
        group = group.sort_values("n")
        correct = ps_model == "ps_true"
        style_kw = dict(color=colour[scenario],
                        linestyle="-" if correct else (0, (4, 2)),
                        linewidth=1.9, marker="o", markersize=3.4,
                        markeredgecolor="none", zorder=3)
        label = (f"scenario {scenario}, "
                 f"{'correct' if correct else 'false'} propensity model")

        axes[0].plot(group["n"], group["median_max_weight"], label=label, **style_kw)
        axes[0].fill_between(group["n"], group["median_max_weight"],
                             group["max_weight"], color=colour[scenario],
                             alpha=0.10, linewidth=0, zorder=1)
        axes[1].plot(group["n"], 100.0 * group["min_effective_n"] / group["n"],
                     label=label, **style_kw)

    for ax, ylabel, title in (
            (axes[0], "largest weight in a replicate",
             "Inverse-probability weights, median replicate\n"
             "(shading reaches the worst replicate seen)"),
            (axes[1], "effective sample size, worst replicate (% of n)",
             "Sample the weighting effectively leaves behind")):
        ax.axvline(config.N, color=INK, linewidth=1.0, linestyle=(0, (4, 3)),
                   zorder=0)
        ax.annotate(f"study n = {config.N}", xy=(config.N, 0.02),
                    xycoords=("data", "axes fraction"), xytext=(4, 0),
                    textcoords="offset points", color=INK_SOFT, fontsize=8)
        ax.set_xscale("log")
        ax.grid(True, which="major", color=GRID, linewidth=0.6, zorder=0)
        ax.set_xlabel("sample size n")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=9.5)
    axes[0].set_yscale("log")
    axes[1].set_ylim(0, 100)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Numerical exposure of the baseline, across conditions "
                 "in which it raises no warning at all",
                 x=0.011, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-dir", default=bench_config.BENCHMARK_DIR)
    parser.add_argument("--outdir", default=config.FIGURE_DIR)
    parser.add_argument("--skip-missing", action="store_true",
                        help="draw whatever CSVs exist rather than failing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    os.makedirs(args.outdir, exist_ok=True)

    def path(name):
        return os.path.join(args.bench_dir, name)

    def out(name):
        return os.path.join(args.outdir, name)

    drawn = 0
    if os.path.exists(path("complexity.csv")) or not args.skip_missing:
        timings = _frame(path("complexity.csv"), "complexity")
        fits = _frame(path("complexity_fits.csv"), "complexity")
        plot_complexity(timings, fits, out("complexity.png"))
        plot_overhead(timings, out("overhead.png"))
        drawn += 2

    stability = path("stability_baseline_summary.csv")
    if os.path.exists(stability):
        plot_stability(pd.read_csv(stability), out("stability_weights.png"))
        drawn += 1
    elif not args.skip_missing:
        _frame(stability, "stability-check")

    benchmark = path("benchmark.csv")
    if os.path.exists(benchmark):
        from .benchmark import plot_comparison, plot_speedup
        plot_comparison(pd.read_csv(benchmark), out("benchmark.png"))
        plot_speedup(pd.read_csv(benchmark), out("speedup.png"))
        drawn += 2

    logger.info("drew %d performance figures into %s", drawn, args.outdir)


if __name__ == "__main__":
    main()
