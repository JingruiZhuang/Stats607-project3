"""Figures for the performance study: complexity, speedup and scaling.

Shares the palette and styling of ``src.figures`` so that the performance
figures read as part of the same report, and draws only from the CSVs written
by ``src.complexity``, ``src.benchmark`` and ``src.stability`` -- no figure
here re-runs the simulation, so a plot can be redrawn without a fresh sweep.

The figures are the three comparisons the unit asks for -- empirical
complexity, overall timing, and parallel speedup -- plus the budget figure of
docs/EXTENSIONS.md.
"""

import argparse
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import bench_config, budget, complexity, config
from .figures import FAMILY_COLOUR, GRID, INK, INK_SOFT, SURFACE, style

logger = logging.getLogger(__name__)

# Implementation identity gets its own two colours, chosen from the same
# validated categorical palette as src.figures but kept distinct from the
# estimator-family colours so the two sets of figures cannot be confused.
TARGET_COLOUR = {"baseline": "#8a6a4a", "optimized": "#1baf7a"}
TARGET_LABEL = {"baseline": "baseline (Unit 2)", "optimized": "optimized"}

# Scenario identity, for the budget figure, where the comparison is between
# the two scenarios rather than between two implementations.
SCENARIO_COLOUR = {"A": "#2a78d6", "B": "#eb6834"}

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


def plot_budget(dispersion, fixed, out_path, at_n=None):
    """The three relationships a simulation budget is decided by.

    Left: how fast each cell's sampling variance falls with n, which is what
    makes replicates at large n redundant.  Middle: the replicate count that
    holds one precision target as n grows, against the 1/n line that theory
    predicts.  Right: what the two together do to the cost of the study -- the
    uniform design pays in proportion to n, the budgeted one pays almost
    nothing extra.
    """
    style()
    at_n = at_n if at_n is not None else config.N
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.8))
    ax_var, ax_count, ax_cost = axes

    # --- variance against n, one line per cell
    sound = dispersion[dispersion["tail_share"] < budget.TAIL_SHARE_LIMIT]
    excluded = {}
    for (scenario, estimator), cell in dispersion.groupby(["scenario", "estimator"],
                                                          sort=False):
        cell = cell.sort_values("n")
        family = estimator.split(".")[0]
        keep = cell["tail_share"] < budget.TAIL_SHARE_LIMIT
        ax_var.plot(cell.loc[keep, "n"], cell.loc[keep, "variance"],
                    color=FAMILY_COLOUR.get(family, INK), linewidth=1.2,
                    alpha=0.75, marker="o", markersize=2.6,
                    markeredgecolor="none",
                    linestyle="-" if scenario == "A" else (0, (4, 2)), zorder=3)
        excluded[family] = excluded.get(family, []) + list(cell.loc[~keep, "n"])

    # The excluded cells sit twenty orders of magnitude above the rest, so they
    # are drawn against the top of the panel rather than allowed to flatten
    # everything else into a line.
    top = float(sound["variance"].max()) * 6.0
    bottom = float(sound["variance"].min()) / 2.0
    for family, sizes in excluded.items():
        ax_var.plot(sizes, [top / 1.8] * len(sizes), linestyle="none",
                    marker="^", markersize=6.5, markerfacecolor="none",
                    markeredgewidth=1.2, zorder=5,
                    color=FAMILY_COLOUR.get(family, INK))

    reference = float(sound[sound["n"] == sound["n"].min()]["variance"].max())
    grid = np.geomspace(dispersion["n"].min(), dispersion["n"].max(), 100)
    ax_var.plot(grid, reference * dispersion["n"].min() / grid, color=INK,
                linewidth=1.0, linestyle=(0, (1, 2)), zorder=2, label="1/n")
    ax_var.set_yscale("log")
    ax_var.set_ylim(bottom, top)
    ax_var.set_ylabel("sampling variance of the estimator")
    ax_var.set_title("Every well-behaved cell's variance falls as 1/n\n"
                     "(carets: cells off the scale, where one replicate "
                     f"carries over {100 * budget.TAIL_SHARE_LIMIT:.0f}% of it)",
                     loc="left", fontsize=9.5)
    ax_var.legend(frameon=False, loc="lower left", fontsize=8)

    # --- replicate count holding one target, against 1/n
    for scenario, plan in fixed.groupby("scenario", sort=False):
        plan = plan.sort_values("n")
        ax_count.plot(plan["n"], plan["n_sim_allocated"], linewidth=1.9,
                      marker="o", markersize=4, markeredgecolor="none",
                      color=SCENARIO_COLOUR[scenario], zorder=3,
                      label=f"scenario {scenario}")
    anchor = fixed[fixed["n"] == at_n]
    ax_count.plot(grid, float(anchor["n_sim_allocated"].max()) * at_n / grid,
                  color=INK, linewidth=1.0, linestyle=(0, (1, 2)), zorder=2,
                  label="1/n")
    ax_count.axhline(config.N_SIM, color=INK_SOFT, linewidth=1.0,
                     linestyle=(0, (4, 3)), zorder=1)
    ax_count.annotate(f"the study's uniform {config.N_SIM}",
                      xy=(0.02, config.N_SIM), xycoords=("axes fraction", "data"),
                      xytext=(0, 4), textcoords="offset points",
                      color=INK_SOFT, fontsize=8)
    ax_count.set_yscale("log")
    ax_count.set_ylabel("replicates needed for the same precision")
    ax_count.set_title("Holding the precision the published design reaches at "
                       f"n = {at_n},\nthe replicate count needed falls like 1/n",
                       loc="left", fontsize=9.5)
    ax_count.legend(frameon=False, loc="upper right", fontsize=8)

    # --- cost of the study, uniform against allocated
    totals = fixed.groupby("n", as_index=False)[["cost_uniform",
                                                 "cost_allocated"]].sum()
    for column, colour, label in (
            ("cost_uniform", "#8a6a4a", f"uniform {config.N_SIM} replicates"),
            ("cost_allocated", "#1baf7a", "allocated to the target")):
        ax_cost.plot(totals["n"], totals[column], linewidth=1.9, marker="o",
                     markersize=4, markeredgecolor="none", color=colour,
                     zorder=3, label=label)
    ax_cost.set_yscale("log")
    ax_cost.set_ylabel("seconds for the whole simulation")
    ax_cost.set_yticks([0.3, 0.5, 1, 2, 3, 5])
    ax_cost.set_title("So the cost of the study stops growing with n",
                      loc="left", fontsize=9.5)
    ax_cost.legend(frameon=False, loc="upper left", fontsize=8)

    sizes = sorted(dispersion["n"].unique())
    for ax in (ax_var, ax_count, ax_cost):
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(int(size)) for size in sizes])
        ax.minorticks_off()
        ax.axvline(at_n, color=INK, linewidth=1.0, linestyle=(0, (4, 3)), zorder=0)
        ax.annotate(f"study n = {at_n}", xy=(at_n, 0.02),
                    xycoords=("data", "axes fraction"), xytext=(4, 0),
                    textcoords="offset points", color=INK_SOFT, fontsize=8)
        ax.set_xlabel("sample size n")
    for ax in (ax_count, ax_cost):
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    for ax in axes.ravel():
        ax.grid(True, which="major", color=GRID, linewidth=0.6, zorder=0)

    fig.suptitle("A replicate count chosen cell by cell: the study's precision "
                 "for a fraction of its cost",
                 x=0.011, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
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
        drawn += 1

    benchmark, ladder = path("benchmark.csv"), path("benchmark_ladder.csv")
    if os.path.exists(benchmark) and os.path.exists(ladder):
        from .benchmark import plot_comparison
        plot_comparison(pd.read_csv(benchmark), pd.read_csv(ladder),
                        out("benchmark.png"))
        drawn += 1
    elif not args.skip_missing:
        _frame(benchmark, "benchmark")

    dispersion = path("budget_dispersion.csv")
    if os.path.exists(dispersion):
        plot_budget(pd.read_csv(dispersion),
                    pd.read_csv(path("budget_fixed_target.csv")),
                    out("budget.png"))
        drawn += 1

    cores = path("benchmark_cores.csv")
    if os.path.exists(cores):
        from .benchmark import plot_speedup
        plot_speedup(pd.read_csv(cores), out("speedup.png"))
        drawn += 1

    logger.info("drew %d performance figures into %s", drawn, args.outdir)


if __name__ == "__main__":
    main()
