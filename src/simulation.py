"""Run the full simulation and write raw estimates to disk.

Replicates are drawn and estimated in batches: a batch holds its data as arrays
with a leading replicate axis, so the design matrices, the logistic fits and
the least-squares solves are each performed for the whole batch in one call.
The per-replicate Python that dominated the Unit 2 runtime is paid once per
batch instead of once per replicate.

Batches are also the unit of parallelism.  Seeds are derived from the master
seed by position, so which batch a replicate lands in and which process runs
that batch cannot affect its result: a parallel run, a serial run and a run
with a different batch size all produce the same estimates.
"""

import argparse
import logging
import os
import time

import numpy as np
import pandas as pd

from . import config, dgps, methods

logger = logging.getLogger(__name__)

# Replicates per batch.  Large enough that per-batch overhead is negligible,
# small enough that the stacked design matrices stay in cache and that a
# parallel run has several batches per worker to balance over.  The runtime is
# flat over a wide range around this; see docs/OPTIMIZATION.md.
BATCH_SIZE = 250


def cells(scenario):
    """Return the estimator configurations to run for one scenario."""
    extensions = (config.EXTENSION_ESTIMATORS
                  if scenario in config.EXTENSION_SCENARIOS else [])
    return config.ESTIMATORS + extensions


def scenario_seeds(scenario, n_sim, master_seed):
    """Return the seed of every replicate of a scenario, in order.

    Derived against the full scenario list rather than the subset being run, so
    that a single scenario rerun on its own draws the same data as it does in a
    full run, and a short run reproduces the leading replicates of a long one.
    """
    all_scenarios = list(config.SCENARIOS)
    parent = np.random.SeedSequence(master_seed).spawn(len(all_scenarios))
    return parent[all_scenarios.index(scenario)].spawn(n_sim)


def run_replicate(scenario, replicate, seed, n):
    """Draw one dataset and return every estimate computed on it.

    All estimators within a replicate see the same data, so differences between
    them are not confounded with the draw.  The seed is passed in rather than
    derived here, so a single replicate can be rerun on its own.
    """
    columns = dgps.generate_columns(scenario, n, np.random.default_rng(seed))
    estimates = methods.estimate_batch(columns, scenario, cells(scenario))
    return [
        {"scenario": scenario, "replicate": replicate, "estimator": name,
         "estimate": float(value)}
        for name, value in estimates.items()
    ]


def run_batch(scenario, n, seeds, first_replicate, report=None):
    """Draw and estimate a block of replicates at once.

    Returns the estimates as a dict of arrays rather than as records; the
    caller assembles the frame once for the whole run, which keeps pandas out
    of the inner loop entirely.
    """
    generators = [np.random.default_rng(seed) for seed in seeds]
    columns = dgps.generate_columns(scenario, n, generators, n_sim=len(seeds))
    estimates = methods.estimate_batch(columns, scenario, cells(scenario), report)
    return first_replicate, estimates


def _batches(n_sim, batch_size):
    """Index ranges covering ``n_sim`` replicates."""
    return [(start, min(start + batch_size, n_sim))
            for start in range(0, n_sim, batch_size)]


def _frame(scenario, blocks, n_sim):
    """Assemble one scenario's estimates into the long format on disk.

    Built column-wise from the stacked arrays and ordered by replicate and then
    by estimator, which is the order the per-replicate loop produced, so the
    file is comparable row for row with the Unit 2 output.
    """
    names = [name for name, *_ in cells(scenario)]
    estimates = {name: np.empty(n_sim) for name in names}
    for first, block in blocks:
        for name, values in block.items():
            estimates[name][first:first + len(values)] = values

    return pd.DataFrame({
        "scenario": scenario,
        "replicate": np.repeat(np.arange(n_sim), len(names)),
        "estimator": np.tile(names, n_sim),
        "estimate": np.column_stack([estimates[name] for name in names]).ravel(),
    })


def _run_scenario(scenario, n, n_sim, master_seed, batch_size, jobs, report):
    seeds = scenario_seeds(scenario, n_sim, master_seed)
    ranges = _batches(n_sim, batch_size)
    work = [(scenario, n, seeds[start:stop], start) for start, stop in ranges]

    if jobs == 1:
        return [run_batch(*arguments, report) for arguments in work]

    from joblib import Parallel, delayed
    # Each worker returns its own convergence counts, which are merged into the
    # caller's report: a warning raised inside a worker process would not reach
    # the parent, and a silent non-convergence is exactly what this study was
    # missing a way to see.
    results = Parallel(n_jobs=jobs, backend="loky")(
        delayed(_batch_with_report)(*arguments) for arguments in work)
    for block, counts in results:
        report.fits += counts.fits
        report.failures += counts.failures
        report.clipped_fits += counts.clipped_fits
        report.max_iterations = max(report.max_iterations, counts.max_iterations)
    return [block for block, _ in results]


def _batch_with_report(scenario, n, seeds, first_replicate):
    """``run_batch`` in a worker process, returning its convergence counts."""
    report = methods.ConvergenceReport()
    return run_batch(scenario, n, seeds, first_replicate, report), report


def run(scenarios, n, n_sim, master_seed, out_path, jobs=1,
        batch_size=BATCH_SIZE):
    """Run every scenario and write a long-format CSV of all estimates."""
    report = methods.ConvergenceReport()
    frames = []

    for scenario in scenarios:
        started = time.perf_counter()
        blocks = _run_scenario(scenario, n, n_sim, master_seed, batch_size,
                               jobs, report)
        frames.append(_frame(scenario, blocks, n_sim))
        logger.info("scenario %s: %d replicates in %.1fs",
                    scenario, n_sim, time.perf_counter() - started)

    raw = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    raw.to_csv(out_path, index=False)

    logger.info("%d logistic fits, %d did not converge, worst %d iterations, "
                "%d fits clipped at the bound",
                report.fits, report.failures, report.max_iterations,
                report.clipped_fits)
    report.warn_if_needed()
    logger.info("wrote %d estimates to %s", len(raw), out_path)
    return raw


def main():
    """Run the simulation from the command line and write the raw estimates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=config.RAW_PATH)
    parser.add_argument("--n", type=int, default=config.N)
    parser.add_argument("--n-sim", type=int, default=config.N_SIM)
    parser.add_argument("--seed", type=int, default=config.MASTER_SEED)
    parser.add_argument("--scenario", action="append", dest="scenarios")
    parser.add_argument("--jobs", type=int, default=1,
                        help="worker processes; -1 uses every core")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(args.scenarios or list(config.SCENARIOS),
        args.n, args.n_sim, args.seed, args.out, args.jobs, args.batch_size)


if __name__ == "__main__":
    main()
