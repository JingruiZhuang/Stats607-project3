"""Honest wall-clock comparison of the baseline against the optimized code.

Three measurements, each written to its own CSV and each plotted by
``src.perf_figures``:

* **implementations** -- the whole study, baseline against optimized, across a
  range of simulation sizes, which is the headline speedup.
* **ladder** -- the same study with the optimizations switched on one at a
  time, so each one's own contribution can be read off rather than inferred.
  The rungs use knobs the production code already has, so nothing is measured
  that is not also shipped.
* **cores** -- the parallel path against the number of worker processes, which
  is where the limits of parallelising this particular study show up.

Timings are the smallest of several runs after a discarded warm-up, for the
reason given in ``src.complexity``.
"""

import argparse
import importlib
import logging
import os
import time
import warnings

import numpy as np
import pandas as pd

from . import bench_config, config
from .complexity import _best_of

logger = logging.getLogger(__name__)

# Simulation sizes for the headline comparison.  The study's own n_sim is in
# the middle, with a smaller size to show the fixed costs and a larger one to
# show that the gain is not an artefact of the particular size.
COMPARISON_N_SIM = [100, 250, 1000, 2500]

# The ladder, as (label, implementation, keyword arguments).  Each rung adds
# one optimization to the one above it.
#
# Rung 2 runs the optimized code with a batch of one: the caching, the array
# design matrices and the hand-written solver are all active, but replicates
# are still handled one at a time, so the difference from rung 1 is what those
# three buy on their own and the difference from rung 3 is what batching buys.
LADDER = [
    ("baseline (Unit 2)", "baseline", {}),
    ("+ cached fits, array designs, own solver", "optimized",
     {"batch_size": 1, "jobs": 1}),
    ("+ batched replicates", "optimized", {"batch_size": 250, "jobs": 1}),
    ("+ parallel workers", "optimized", {"batch_size": 250, "jobs": -1}),
]

CORE_COUNTS = [1, 2, 4, 6, 8]


def _module(target):
    return importlib.import_module(bench_config.IMPLEMENTATIONS[target])


def _time_run(target, scenarios, n, n_sim, seed, out_path, repeats, **kwargs):
    module = _module(target)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _best_of(repeats,
                        lambda: module.run(scenarios, n, n_sim, seed, out_path,
                                           **kwargs))


def compare_implementations(scenarios, n, sizes, seed, out_dir, repeats=3):
    """Time the whole study, baseline against optimized, across study sizes."""
    rows = []
    for n_sim in sizes:
        for target in ("baseline", "optimized"):
            path = os.path.join(out_dir, f"{target}_benchmark_estimates.csv")
            seconds = _time_run(target, scenarios, n, n_sim, seed, path, repeats)
            rows.append({"target": target, "n": n, "n_sim": n_sim,
                         "seconds": seconds, "seconds_per_replicate":
                         seconds / (n_sim * len(scenarios))})
            logger.info("%-9s n_sim=%-5d %.2fs", target, n_sim, seconds)

    table = pd.DataFrame(rows)
    wide = table.pivot(index="n_sim", columns="target", values="seconds")
    wide["speedup"] = wide["baseline"] / wide["optimized"]
    return table, wide.reset_index()


def compare_ladder(scenarios, n, n_sim, seed, out_dir, repeats=3):
    """Each optimization's own contribution, added one rung at a time."""
    rows = []
    previous = None
    for position, (label, target, kwargs) in enumerate(LADDER):
        path = os.path.join(out_dir, f"ladder_{position}_estimates.csv")
        seconds = _time_run(target, scenarios, n, n_sim, seed, path, repeats,
                            **kwargs)
        rows.append({"rung": position, "label": label, "target": target,
                     "seconds": seconds,
                     "speedup_vs_baseline": rows[0]["seconds"] / seconds
                     if rows else 1.0,
                     "speedup_vs_previous": previous / seconds
                     if previous else 1.0})
        previous = seconds
        logger.info("rung %d  %-42s %.2fs", position, label, seconds)
    return pd.DataFrame(rows)


def compare_cores(scenarios, n, n_sim, seed, out_dir, counts=None, repeats=3):
    """Runtime and speedup against the number of worker processes."""
    counts = counts or CORE_COUNTS
    available = os.cpu_count() or 1
    counts = [c for c in counts if c <= available]

    rows = []
    serial = None
    for jobs in counts:
        path = os.path.join(out_dir, f"cores_{jobs}_estimates.csv")
        seconds = _time_run("optimized", scenarios, n, n_sim, seed, path,
                            repeats, jobs=jobs)
        serial = serial if serial is not None else seconds
        rows.append({"jobs": jobs, "seconds": seconds,
                     "speedup": serial / seconds,
                     "efficiency": (serial / seconds) / jobs})
        logger.info("jobs=%-2d %.2fs  speedup %.2fx", jobs, seconds,
                    serial / seconds)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures

def plot_comparison(comparison, ladder, out_path):
    """Runtime against simulation size, and the ladder that explains it."""
    import matplotlib.pyplot as plt
    from .figures import GRID, INK, INK_SOFT
    from .perf_figures import TARGET_COLOUR, TARGET_LABEL
    from .figures import style

    style()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5),
                             gridspec_kw={"width_ratios": [1, 1.35]})

    for target, group in comparison.groupby("target", sort=False):
        group = group.sort_values("n_sim")
        axes[0].plot(group["n_sim"], group["seconds"],
                     color=TARGET_COLOUR[target], linewidth=2.0, marker="o",
                     markersize=4, markeredgecolor="none",
                     label=TARGET_LABEL[target])
    study = comparison[comparison["n_sim"] == config.N_SIM]
    if not study.empty:
        fast = study[study.target == "optimized"]["seconds"].iloc[0]
        slow = study[study.target == "baseline"]["seconds"].iloc[0]
        axes[0].annotate(f"{slow / fast:.0f}x at the study's\n"
                         f"n_sim = {config.N_SIM}",
                         xy=(config.N_SIM, fast), xytext=(-6, 26),
                         textcoords="offset points", fontsize=8, color=INK_SOFT,
                         ha="right")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].grid(True, which="major", color=GRID, linewidth=0.6, zorder=0)
    axes[0].set_xlabel("replicates per scenario")
    axes[0].set_ylabel("seconds for the whole study")
    axes[0].set_title("Whole study, both implementations", loc="left",
                      fontsize=9.5)
    axes[0].legend(frameon=False, loc="upper left")

    rungs = ladder.sort_values("rung")
    positions = np.arange(len(rungs))
    colours = [TARGET_COLOUR["baseline"]] + [TARGET_COLOUR["optimized"]] * (
        len(rungs) - 1)
    axes[1].barh(positions, rungs["seconds"], color=colours, height=0.62,
                 zorder=3)
    for y, (seconds, speedup) in enumerate(zip(rungs["seconds"],
                                               rungs["speedup_vs_baseline"])):
        axes[1].annotate(f"{seconds:.2f}s" + (f"   {speedup:.0f}x" if y else ""),
                         xy=(seconds, y), xytext=(5, 0),
                         textcoords="offset points", va="center", fontsize=8,
                         color=INK)
    axes[1].set_yticks(positions)
    axes[1].set_yticklabels(rungs["label"], fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, rungs["seconds"].max() * 1.28)
    axes[1].grid(True, axis="x", color=GRID, linewidth=0.6, zorder=0)
    axes[1].set_xlabel("seconds for the whole study")
    axes[1].set_title(f"One optimization at a time "
                      f"(n = {config.N}, n_sim = {config.N_SIM})",
                      loc="left", fontsize=9.5)

    fig.suptitle("Runtime of the simulation study, before and after",
                 x=0.011, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_path)


def plot_speedup(cores, out_path):
    """Speedup against worker count, against the linear ideal."""
    import matplotlib.pyplot as plt
    from .figures import GRID, INK, INK_SOFT, style
    from .perf_figures import TARGET_COLOUR

    style()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    jobs = cores["jobs"].to_numpy()

    ax.plot(jobs, jobs, color=INK_SOFT, linewidth=1.0, linestyle=(0, (4, 3)),
            zorder=2, label="linear speedup")
    ax.plot(jobs, cores["speedup"], color=TARGET_COLOUR["optimized"],
            linewidth=2.0, marker="o", markersize=4.5, markeredgecolor="none",
            zorder=3, label="measured")
    for x, y in zip(jobs, cores["speedup"]):
        ax.annotate(f"{y:.1f}x", xy=(x, y), xytext=(0, -14),
                    textcoords="offset points", ha="center", fontsize=8,
                    color=INK)

    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_xlabel("worker processes")
    ax.set_ylabel("speedup over one worker")
    ax.set_xticks(jobs)
    ax.legend(frameon=False, loc="upper left")
    fig.suptitle("Parallel speedup of the optimized study, "
                 f"n_sim = {config.N_SIM}", x=0.011, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_path)


def main():
    """Run the benchmarks and write their timings to disk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=config.N)
    parser.add_argument("--n-sim", type=int, default=config.N_SIM)
    parser.add_argument("--seed", type=int, default=config.MASTER_SEED)
    parser.add_argument("--scenario", action="append", dest="scenarios")
    parser.add_argument("--out-dir", default=bench_config.BENCHMARK_DIR)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--only", action="append", dest="sections",
                        choices=["implementations", "ladder", "cores"],
                        help="run one section at a time; each writes its own "
                             "CSV, so the three can be measured separately on "
                             "an otherwise idle machine")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    scenarios = args.scenarios or list(config.SCENARIOS)
    os.makedirs(args.out_dir, exist_ok=True)

    def path(name):
        return os.path.join(args.out_dir, name)

    sections = args.sections or ["implementations", "ladder", "cores"]
    show = lambda frame: frame.to_string(index=False,
                                         float_format=lambda x: f"{x: .2f}")

    if "implementations" in sections:
        comparison, speedups = compare_implementations(
            scenarios, args.n, COMPARISON_N_SIM, args.seed, args.out_dir,
            args.repeats)
        comparison.to_csv(path("benchmark.csv"), index=False)
        speedups.to_csv(path("benchmark_speedup.csv"), index=False)
        print("\nWhole study, baseline against optimized")
        print(show(speedups), flush=True)

    if "ladder" in sections:
        ladder = compare_ladder(scenarios, args.n, args.n_sim, args.seed,
                                args.out_dir, args.repeats)
        ladder.to_csv(path("benchmark_ladder.csv"), index=False)
        print("\nOne optimization at a time")
        print(show(ladder), flush=True)

    if "cores" in sections:
        cores = compare_cores(scenarios, args.n, args.n_sim, args.seed,
                              args.out_dir, repeats=args.repeats)
        cores.to_csv(path("benchmark_cores.csv"), index=False)
        print("\nParallel scaling")
        print(show(cores), flush=True)


if __name__ == "__main__":
    main()
