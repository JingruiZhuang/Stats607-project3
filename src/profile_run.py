"""Profile a representative simulation run and report the main bottlenecks.

Writes a ``.prof`` file for interactive inspection (``snakeviz``) and a text
table of the functions holding the most cumulative and most internal time.
``--target baseline`` profiles the frozen Unit 2 code, ``--target optimized``
the current ``src``.
"""

import argparse
import cProfile
import importlib
import io
import logging
import os
import pstats
import time
import warnings

from . import bench_config, config

logger = logging.getLogger(__name__)

# Pipeline stages, keyed by the "file:lineno(function)" that pstats reports.
# Reported with cumulative time, so the stages NEST: `generate` and the
# estimators both call `design_matrix`, and `fit_propensity` contains the whole
# statsmodels IRLS.  The table says where to look, it is not a partition.
STAGES = [
    # the baseline's names
    ("estimate: all estimators",          "methods.py", "estimate"),
    ("  fit_propensity",                  "methods.py", "fit_propensity"),
    ("    statsmodels IRLS",              "generalized_linear_model.py", "_fit_irls"),
    ("  OLS solve (lstsq wrapper)",       "methods.py", "_ols"),
    ("DGP: generate one dataset",         "dgps.py", "generate"),
    # the optimized code's names
    ("estimate_batch: all estimators",    "methods.py", "estimate_batch"),
    ("  logistic fit (own IRLS)",         "methods.py", "fit_logistic"),
    ("  least-squares solve",             "methods.py", "solve_ols"),
    ("    linear solve / Cholesky",       "methods.py", "_solve"),
    ("DGP: generate a batch",             "dgps.py", "generate_columns"),
    # shared
    ("design_matrix (DGP + estimators)",  "dgps.py", "design_matrix"),
    ("with_treatment (DataFrame copy)",   "dgps.py", "with_treatment"),
]

# Where the time is physically spent.  Internal (`tottime`) time grouped by the
# package the frame belongs to, which IS a partition of the run: every frame
# counts once and the shares sum to 100%.
# Built-in frames carry no path, so they are matched on the function name
# instead: NumPy's C routines (`c_einsum`, `matmul`, the ufunc methods) would
# otherwise be charged to the standard library, which is where the first
# profile of the optimized code hid two thirds of its runtime.
PACKAGES = [
    ("pandas", "/pandas/", ()),
    ("statsmodels", "/statsmodels/", ()),
    ("numpy", "/numpy/", ("numpy", "ufunc", "multiarray", "matmul", "dot")),
    ("this study's own code", "/project 3/", ()),
    ("python stdlib / builtins", None, ()),
]

# Call counts worth reporting per replicate, with the number of *distinct*
# results each replicate actually needs.  Where the two differ, the gap is
# redundant computation that caching or restructuring can remove outright.
# The distinct counts are properties of the design in `config`, computed below.
COUNTED = [
    ("logistic fits performed", "methods.py", "fit_logistic"),
    ("propensity fits requested", "methods.py", "fit_propensity"),
    ("design matrices built", "dgps.py", "design_matrix"),
    ("DataFrame copies (with_treatment)", "dgps.py", "with_treatment"),
    ("OLS solves", "methods.py", "_ols"),
    ("eval() of a model term string", None, "builtins.eval"),
    ("pandas column reads", "/pandas/core/frame.py", "__getitem__"),
]


def distinct_propensity_fits(scenarios):
    """Propensity models actually needed per replicate, summed over scenarios.

    Every estimator refits the propensity model named in its own row of
    `config.ESTIMATORS`, but a scenario names only two distinct models, so the
    rest of the fits recompute a result the replicate already has.
    """
    import importlib
    simulation = importlib.import_module("baseline.simulation")
    total = 0
    for scenario in scenarios:
        names = {ps for _, _, ps, _ in simulation.cells(scenario) if ps}
        total += len(names)
    return total


def profile(target, scenarios, n, n_sim, seed, out_dir):
    """Run one simulation under cProfile and return the stats object."""
    module = importlib.import_module(bench_config.IMPLEMENTATIONS[target])
    os.makedirs(out_dir, exist_ok=True)
    prof_path = os.path.join(out_dir, f"{target}.prof")

    # The profiled run writes its estimates to the profile directory rather
    # than to results/raw, so that profiling never clobbers the study output.
    raw_path = os.path.join(out_dir, f"{target}_profile_estimates.csv")

    profiler = cProfile.Profile()
    started = time.perf_counter()
    with warnings.catch_warnings():
        # Warnings are the stability check's business; silencing them here
        # keeps their formatting cost out of the timings.
        warnings.simplefilter("ignore")
        profiler.enable()
        module.run(scenarios, n, n_sim, seed, raw_path)
        profiler.disable()
    elapsed = time.perf_counter() - started

    profiler.dump_stats(prof_path)
    logger.info("profiled %s: %.1fs under cProfile -> %s", target, elapsed, prof_path)
    return pstats.Stats(profiler), elapsed, prof_path


def stage_times(stats):
    """Cumulative seconds in each pipeline stage, in the order of STAGES."""
    rows = []
    for label, filename, function in STAGES:
        seconds = sum(
            entry[3] for key, entry in stats.stats.items()
            if key[0].endswith(filename) and key[2] == function
        )
        rows.append((label, seconds))
    return rows


def package_times(stats):
    """Group internal seconds by package, a true partition of the run."""
    totals = {label: 0.0 for label, *_ in PACKAGES}
    for key, entry in stats.stats.items():
        path, function, tottime = key[0], key[2], entry[2]
        for label, marker, builtins in PACKAGES:
            matched = (marker is None
                       or marker in path
                       or (path == "~" and any(b in function for b in builtins)))
            if matched:
                totals[label] += tottime
                break
    return [(label, totals[label]) for label, *_ in PACKAGES]


def _table(rows, total, header, note):
    out = io.StringIO()
    print(header, file=out)
    print(note, file=out)
    print(f"{'':<34}{'seconds':>10}{'share':>9}", file=out)
    for label, seconds in rows:
        if seconds <= 0.0:
            continue                      # not part of this implementation
        share = 100.0 * seconds / total if total else 0.0
        print(f"{label:<34}{seconds:>10.3f}{share:>8.1f}%", file=out)
    return out.getvalue()


def report(stats, elapsed, target, scenarios, n_replicates, out_dir, top=25):
    """Write and return the human-readable profile summary."""
    buffer = io.StringIO()
    stats.stream = buffer
    internal = sum(entry[2] for entry in stats.stats.values())

    print(f"Profile of the {target} implementation", file=buffer)
    print(f"Wall time under cProfile: {elapsed:.1f}s; {stats.total_calls:,} "
          f"function calls.", file=buffer)
    print("cProfile's per-call overhead inflates these seconds -- the shares "
          "are what to read.\nUse `make benchmark` for honest wall times.\n",
          file=buffer)

    print(_table(stage_times(stats), elapsed,
                 "Cumulative time by pipeline stage",
                 "(stages nest: indented rows are inside the row above; the last two\n are called from several stages and so overlap them)"),
          file=buffer)
    counts = []
    for label, filename, function in COUNTED:
        calls = sum(
            entry[1] for key, entry in stats.stats.items()
            if (function in key[2]
                and (filename is None or key[0].endswith(filename)))
        )
        counts.append((label, calls / n_replicates))
    print("Calls per replicate", file=buffer)
    print(f"{'':<34}{'per rep':>10}", file=buffer)
    for label, per in counts:
        print(f"{label:<34}{per:>10.3f}", file=buffer)
    needed = distinct_propensity_fits(scenarios) / len(scenarios)
    print(f"\nDistinct propensity models the design needs per replicate: "
          f"{needed:.1f}", file=buffer)
    # A batched implementation fits a whole block of replicates in one call, so
    # its calls-per-replicate is below one by design and the redundancy
    # arithmetic below does not apply to it.
    done = dict(counts).get("propensity fits requested", 0.0)
    if done >= needed:
        print(f"So {100 * (1 - needed / done):.0f}% of the propensity fits "
              f"recompute a result the replicate already holds.\n", file=buffer)
    else:
        batched = dict(counts).get("logistic fits performed", 0.0)
        print(f"Logistic fits are performed {batched:.3f} times per replicate: "
              f"each call fits a whole\nbatch of replicates at once, and each "
              f"model is fitted once per batch rather than\nonce per estimator "
              f"that asks for it.\n", file=buffer)

    print(_table(package_times(stats), internal,
                 "Internal time by package",
                 "(a partition of the run: each frame counted once)"),
          file=buffer)

    print(f"\nTop {top} functions by cumulative time", file=buffer)
    stats.sort_stats("cumulative").print_stats(top)
    print(f"\n\nTop {top} functions by internal time", file=buffer)
    stats.sort_stats("tottime").print_stats(top)

    text = buffer.getvalue()
    path = os.path.join(out_dir, f"{target}_profile.txt")
    with open(path, "w") as handle:
        handle.write(text)
    logger.info("wrote %s", path)
    return text


def main():
    """Profile one implementation and write the attribution to disk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="baseline",
                        choices=sorted(bench_config.IMPLEMENTATIONS))
    parser.add_argument("--n", type=int, default=bench_config.PROFILE_N)
    parser.add_argument("--n-sim", type=int, default=bench_config.PROFILE_N_SIM)
    parser.add_argument("--seed", type=int, default=config.MASTER_SEED)
    parser.add_argument("--scenario", action="append", dest="scenarios")
    parser.add_argument("--out-dir", default=bench_config.PROFILE_DIR)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    scenarios = args.scenarios or list(config.SCENARIOS)

    stats, elapsed, _ = profile(args.target, scenarios, args.n, args.n_sim,
                                args.seed, args.out_dir)
    text = report(stats, elapsed, args.target, scenarios,
                  len(scenarios) * args.n_sim, args.out_dir, args.top)
    # Only the component table and the cumulative-time head go to the console;
    # the file has the rest.
    print(text.split("\nTop")[0])


if __name__ == "__main__":
    main()
