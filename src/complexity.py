"""Empirical computational complexity: runtime against the sample size n.

Times each implementation over a geometric grid of n and fits a power law
``t = a * n^p`` by least squares on the logs, so the exponent p is read off a
log-log plot.  Total runtime is measured for whatever implementation is named;
where the implementation still runs one replicate at a time, the DGP and the
estimators are also timed separately, since they scale differently and the
profile says the estimators dominate.
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

logger = logging.getLogger(__name__)


def _package(target):
    """The package a named implementation lives in (``src`` or ``baseline``)."""
    return bench_config.IMPLEMENTATIONS[target].rsplit(".", 1)[0]


def _best_of(repeats, call):
    """Smallest of ``repeats`` timings of ``call``, after one discarded warm-up.

    The minimum rather than the mean: a timing can only be inflated by other
    work on the machine, never deflated, so the smallest run is the closest to
    the cost of the code itself.  The warm-up absorbs first-call costs --
    statsmodels builds its family objects lazily and the allocator has to grow
    into the working set -- which otherwise land entirely on the first n.
    """
    call()
    best = float("inf")
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        best = min(best, time.perf_counter() - started)
    return best


def time_total(module, scenario, n, n_sim, seed, tmp_path, repeats=3):
    """Seconds for a whole run of ``n_sim`` replicates at sample size ``n``."""
    return _best_of(repeats, lambda: module.run([scenario], n, n_sim, seed, tmp_path))


def time_components(target, scenario, n, n_sim, seed, repeats=3):
    """Seconds spent drawing the data and seconds spent estimating, separately.

    Each implementation is split at its own natural seam and timed through the
    entry point it actually uses, so the two components always sum to roughly
    the total plotted beside them: the baseline one replicate at a time, the
    optimized one batch at a time.  Timing the optimized code through its
    per-replicate API instead would measure a path the study never takes.
    """
    package = _package(target)
    simulation = importlib.import_module(f"{package}.simulation")
    dgps = importlib.import_module(f"{package}.dgps")
    methods = importlib.import_module(f"{package}.methods")

    cells = simulation.cells(scenario)
    seeds = np.random.SeedSequence(seed).spawn(n_sim)
    batched = hasattr(methods, "estimate_batch")
    timings = {"dgp": float("inf"), "method": float("inf")}

    def one_pass():
        dgp_seconds = 0.0
        method_seconds = 0.0
        if batched:
            generators = [np.random.default_rng(s) for s in seeds]
            started = time.perf_counter()
            columns = dgps.generate_columns(scenario, n, generators, n_sim=n_sim)
            dgp_seconds = time.perf_counter() - started

            started = time.perf_counter()
            methods.estimate_batch(columns, scenario, cells)
            method_seconds = time.perf_counter() - started
        else:
            for replicate in range(n_sim):
                rng = np.random.default_rng(seeds[replicate])
                started = time.perf_counter()
                data = dgps.generate(scenario, n, rng)
                dgp_seconds += time.perf_counter() - started

                started = time.perf_counter()
                for _, family, ps_name, or_name in cells:
                    methods.estimate(data, scenario, family, ps_name, or_name)
                method_seconds += time.perf_counter() - started
        timings["dgp"] = min(timings["dgp"], dgp_seconds)
        timings["method"] = min(timings["method"], method_seconds)

    one_pass()                      # warm-up, discarded
    timings.update(dgp=float("inf"), method=float("inf"))
    for _ in range(repeats):
        one_pass()
    return timings["dgp"], timings["method"]


def sweep(target, scenarios, sizes, n_sim, seed, out_dir, components=True,
          repeats=3):
    """Time ``target`` across the grid of sample sizes and return a tidy frame."""
    module = importlib.import_module(bench_config.IMPLEMENTATIONS[target])
    os.makedirs(out_dir, exist_ok=True)
    tmp_path = os.path.join(out_dir, f"{target}_complexity_estimates.csv")

    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n in sizes:
            for scenario in scenarios:
                total = time_total(module, scenario, n, n_sim, seed, tmp_path,
                                   repeats)
                row = {"target": target, "scenario": scenario, "n": n,
                       "n_sim": n_sim, "total_seconds": total,
                       "seconds_per_replicate": total / n_sim}
                if components:
                    split = time_components(target, scenario, n, n_sim, seed,
                                            repeats)
                    if split is not None:
                        dgp, method = split
                        row["dgp_seconds"] = dgp
                        row["method_seconds"] = method
                        row["dgp_per_replicate"] = dgp / n_sim
                        row["method_per_replicate"] = method / n_sim
                rows.append(row)
                logger.info("%s scenario %s n=%-5d %.3f s/rep",
                            target, scenario, n, total / n_sim)
    return pd.DataFrame(rows)


def fit_power_law(sizes, seconds):
    """Least-squares fit of ``log t = log a + p log n``; returns ``(p, a, r2)``.

    The exponent p is the empirical complexity: p = 1 is linear in n, p = 2
    quadratic.  R^2 is reported because a single exponent only describes the
    runtime if the points really do lie on a line in the logs -- at small n
    the per-replicate Python overhead is constant and flattens the curve.
    """
    log_n = np.log(np.asarray(sizes, dtype=float))
    log_t = np.log(np.asarray(seconds, dtype=float))
    slope, intercept = np.polyfit(log_n, log_t, 1)
    predicted = intercept + slope * log_n
    residual = np.sum((log_t - predicted) ** 2)
    total = np.sum((log_t - log_t.mean()) ** 2)
    r2 = 1.0 - residual / total if total > 0 else float("nan")
    return slope, float(np.exp(intercept)), r2


def fit_affine(sizes, seconds, at_n):
    """Fit ``t = c + a n`` and report how much of the cost at ``at_n`` is fixed.

    Theory says every stage of a replicate is O(n) -- the DGP draws n rows, the
    IRLS and the OLS solves are O(n k^2) in a fixed number of columns -- so an
    affine model is the right shape, with c the per-replicate interpreter and
    pandas overhead that no sample size can amortise.  The fixed share at the
    study's own n says how much of the runtime is available to an optimization
    that removes overhead rather than arithmetic.
    """
    sizes = np.asarray(sizes, dtype=float)
    seconds = np.asarray(seconds, dtype=float)
    slope, intercept = np.polyfit(sizes, seconds, 1)
    predicted = intercept + slope * at_n
    return {"fixed_seconds": float(intercept),
            "seconds_per_observation": float(slope),
            "predicted_at_n": float(predicted),
            "fixed_share_at_n": float(intercept / predicted) if predicted else float("nan")}


def overheads(timings, at_n):
    """Affine fit per target, scenario and component."""
    columns = [c for c in ("seconds_per_replicate", "dgp_per_replicate",
                           "method_per_replicate") if c in timings.columns]
    rows = []
    for (target, scenario), group in timings.groupby(["target", "scenario"], sort=False):
        for column in columns:
            values = group[column].dropna()
            if len(values) < 3:
                continue
            fit = fit_affine(group.loc[values.index, "n"], values, at_n)
            rows.append({"target": target, "scenario": scenario,
                         "component": column.replace("_per_replicate", "")
                                            .replace("seconds", "total"),
                         "at_n": at_n, **fit})
    return pd.DataFrame(rows)


def exponents(timings, min_n=0):
    """Fitted exponent per target, scenario and timed component."""
    data = timings[timings["n"] >= min_n]
    columns = [c for c in ("seconds_per_replicate", "dgp_per_replicate",
                           "method_per_replicate")
               if c in data.columns]
    rows = []
    for (target, scenario), group in data.groupby(["target", "scenario"], sort=False):
        for column in columns:
            values = group[column].dropna()
            if len(values) < 3:
                continue
            slope, scale, r2 = fit_power_law(group.loc[values.index, "n"], values)
            rows.append({"target": target, "scenario": scenario,
                         "component": column.replace("_per_replicate", "")
                                            .replace("seconds", "total"),
                         "exponent": slope, "scale": scale, "r2": r2,
                         "min_n": int(data["n"].min())})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", action="append", dest="targets")
    parser.add_argument("--n", action="append", type=int, dest="sizes")
    parser.add_argument("--n-sim", type=int, default=bench_config.COMPLEXITY_N_SIM)
    parser.add_argument("--seed", type=int, default=config.MASTER_SEED)
    parser.add_argument("--scenario", action="append", dest="scenarios")
    parser.add_argument("--out-dir", default=bench_config.BENCHMARK_DIR)
    parser.add_argument("--no-components", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    targets = args.targets or ["baseline"]
    sizes = args.sizes or bench_config.COMPLEXITY_N
    scenarios = args.scenarios or list(config.SCENARIOS)

    timings = pd.concat([
        sweep(target, scenarios, sizes, args.n_sim, args.seed, args.out_dir,
              components=not args.no_components, repeats=args.repeats)
        for target in targets
    ], ignore_index=True)

    os.makedirs(args.out_dir, exist_ok=True)
    timing_path = os.path.join(args.out_dir, "complexity.csv")
    timings.to_csv(timing_path, index=False)

    # Fitted on the upper half of the grid as well as on all of it: the small-n
    # points are dominated by fixed per-replicate overhead and bias the
    # exponent downwards.
    large = sorted(sizes)[len(sizes) // 2]
    fits = pd.concat([exponents(timings), exponents(timings, min_n=large)],
                     ignore_index=True)
    fit_path = os.path.join(args.out_dir, "complexity_fits.csv")
    fits.to_csv(fit_path, index=False)

    affine = overheads(timings, config.N)
    affine_path = os.path.join(args.out_dir, "complexity_overhead.csv")
    affine.to_csv(affine_path, index=False)
    logger.info("wrote %s, %s and %s", timing_path, fit_path, affine_path)

    print("\nSeconds per replicate")
    print(timings.to_string(index=False, float_format=lambda x: f"{x: .4f}"))
    print("\nFitted power law  t = a * n^p")
    print(fits.to_string(index=False, float_format=lambda x: f"{x: .3f}"))
    print(f"\nFitted affine model  t = c + a n, evaluated at the study's "
          f"n = {config.N}")
    print(affine.to_string(index=False, float_format=lambda x: f"{x: .6f}"))


if __name__ == "__main__":
    main()
