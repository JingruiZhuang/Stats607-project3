"""Simulation budget: how many replicates each scenario actually needs.

The study of Unit 2 spends the same number of replicates on every cell, which
spends the same effort on the cells that need it least.  Two facts make that
wasteful.  Statistically, the Monte Carlo standard error of a performance
measure is set by the dispersion of the estimator it summarises, and that
dispersion differs by a factor of several across cells and falls with the
sample size n.  Computationally, a replicate costs time proportional to n, so
the cells whose estimates are most precise are also the ones whose replicates
are most expensive.

This module measures both sides -- dispersion against n from a pilot run, cost
per replicate against n from a timing sweep -- and derives the replicate counts
that reach a stated precision at the least cost.  Nothing here changes the
study's design: `config.py` is untouched, and the recommendation is reported
rather than applied.  See docs/EXTENSIONS.md.
"""

import argparse
import importlib
import logging
import os
import warnings

import numpy as np
import pandas as pd

from . import bench_config, config, metrics
from .complexity import _best_of, fit_power_law, time_components

logger = logging.getLogger(__name__)

# The four measures of Table 2, and how each one's Monte Carlo standard error
# depends on the number of replicates.  The mean and the variance have closed
# forms; the median and the interquartile range do not, and are bootstrapped.
CLOSED_FORM = ("bias_mean", "variance")
BOOTSTRAPPED = ("bias_median", "iqr")

# A cell in which the single most extreme replicate contributes more than this
# share of the total squared deviation has no usable variance: the sample
# variance is then a property of that one draw, and a replicate count derived
# from it is meaningless.  Such cells are excluded from the allocation and
# counted, never silently dropped.  See docs/EXTENSIONS.md section 4.
TAIL_SHARE_LIMIT = 0.05

# Resamples used for the quantile standard errors inside the pilot.  Smaller
# than metrics.N_BOOTSTRAP because what the pilot needs from them is a ratio
# accurate to a percent or two, not a published standard error.
PILOT_N_BOOTSTRAP = 500


# --------------------------------------------------------------- measurement

def _quantile_ses(values, n_bootstrap, rng):
    """Bootstrap standard errors of the median and the interquartile range."""
    n = values.size
    resamples = values[rng.integers(0, n, size=(n_bootstrap, n))]
    quartiles = np.percentile(resamples, [25, 75], axis=1)
    return (float(np.median(resamples, axis=1).std(ddof=1)),
            float(np.diff(quartiles, axis=0).std(ddof=1)))


def dispersion(raw, scenario, n, rng):
    """Per-cell dispersion and Monte Carlo standard errors of one pilot run.

    Every measure's standard error is reported in the scale-free form
    ``se * sqrt(n_sim) / sd``, because that is the quantity that transfers: it
    is a property of the shape of the estimator's sampling distribution, so it
    can be read off a pilot at one replicate count and used to size a study at
    another.
    """
    rows = []
    for estimator, group in raw.groupby("estimator", sort=False):
        values = group["estimate"].to_numpy(dtype=float)
        n_sim = values.size
        sd = values.std(ddof=1)
        median_se, iqr_se = _quantile_ses(values, PILOT_N_BOOTSTRAP, rng)

        # Two diagnostics for whether a variance is a meaningful summary of
        # this cell at all.  `tail_share` is the fraction of the total squared
        # deviation contributed by the single most extreme replicate: a cell
        # where one draw carries a tenth of the variance has no stable variance
        # to budget against.  `robust_sd` is the interquartile range rescaled
        # to the normal, which such a draw cannot move.
        squared = (values - values.mean()) ** 2
        quartiles = np.percentile(values, [25, 75])
        rows.append({
            "scenario": scenario, "n": n, "estimator": estimator,
            "n_sim_pilot": n_sim, "sd": float(sd), "variance": float(sd ** 2),
            "robust_sd": float((quartiles[1] - quartiles[0]) / 1.3489795),
            "tail_share": float(squared.max() / squared.sum()),
            # Closed forms, repeated here so that every measure is on one footing.
            "se_bias_mean": float(sd / np.sqrt(n_sim)),
            "se_variance": float(sd ** 2 * np.sqrt(2.0 / (n_sim - 1))),
            "se_bias_median": median_se,
            "se_iqr": iqr_se,
            # The transferable shape constants.
            "factor_bias_mean": 1.0,
            "factor_bias_median": float(median_se * np.sqrt(n_sim) / sd),
            "factor_iqr": float(iqr_se * np.sqrt(n_sim) / sd),
        })
    return pd.DataFrame(rows)


def pilot(target, scenarios, sizes, n_sim, seed, out_dir):
    """Run the pilot across the grid of sample sizes and summarise dispersion.

    Only the summary is kept.  The estimates themselves are written to
    ``tmp_path`` by each run and are not needed again: everything downstream is
    a function of each cell's dispersion.
    """
    simulation = importlib.import_module(bench_config.IMPLEMENTATIONS[target])
    os.makedirs(out_dir, exist_ok=True)
    tmp_path = os.path.join(out_dir, f"{target}_budget_estimates.csv")

    rng = np.random.default_rng(seed)
    frames = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n in sizes:
            for scenario in scenarios:
                raw = simulation.run([scenario], n, n_sim, seed, tmp_path)
                frames.append(dispersion(raw, scenario, n, rng))
                logger.info("pilot scenario %s n=%-5d %d replicates",
                            scenario, n, n_sim)
    return pd.concat(frames, ignore_index=True)


def costs(target, scenarios, sizes, n_sim, seed, out_dir, repeats=3):
    """Seconds per replicate for each scenario and sample size.

    Reported two ways, because a budget can be spent on either.  ``compute`` is
    the arithmetic alone -- drawing the data and running the estimators -- which
    is the part that scales with n.  ``total`` adds what a whole run of the
    study costs around it, chiefly formatting and writing the estimates, which
    is per replicate and so does not scale with n at all.  The allocation below
    uses the total, since that is the wall clock the budget is actually drawn
    against, but the two differ enough at small n to be worth separating.

    The minimum of several timings, after a discarded warm-up, for the reasons
    given in ``complexity._best_of``.
    """
    simulation = importlib.import_module(bench_config.IMPLEMENTATIONS[target])
    tmp_path = os.path.join(out_dir, f"{target}_budget_timing.csv")

    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n in sizes:
            for scenario in scenarios:
                seconds = _best_of(repeats, lambda: simulation.run(
                    [scenario], n, n_sim, seed, tmp_path))
                dgp, method = time_components(target, scenario, n, n_sim,
                                              seed, repeats)
                rows.append({
                    "scenario": scenario, "n": n,
                    "seconds_per_replicate": seconds / n_sim,
                    "compute_per_replicate": (dgp + method) / n_sim,
                    "output_per_replicate": (seconds - dgp - method) / n_sim})
                logger.info("cost scenario %s n=%-5d %.3g s/replicate "
                            "(%.3g compute)", scenario, n, seconds / n_sim,
                            (dgp + method) / n_sim)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- scaling

def variance_scaling(pilot_frame):
    """Fit ``sd^2 = a n^-q`` per cell.

    Standard asymptotics say the sampling variance of a root-n consistent
    estimator falls like 1/n, so q = 1 is the value to compare against; a cell
    that departs from it is one where the bias, not the variance, is what n is
    buying.  Fitted on the logs, where the power law is a line.
    """
    rows = []
    for (scenario, estimator), group in usable(pilot_frame).groupby(
            ["scenario", "estimator"], sort=False):
        if len(group) < 3:
            continue
        slope, scale, r2 = fit_power_law(group["n"], group["variance"])
        rows.append({"scenario": scenario, "estimator": estimator,
                     "points": int(len(group)), "exponent": float(-slope),
                     "scale": float(scale), "r2": float(r2)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- allocation

def required_replicates(frame, target, measure="bias_mean"):
    """Replicates needed to reach ``target`` Monte Carlo standard error.

    For the mean and the quantile measures the standard error is
    ``f * sd / sqrt(n_sim)``, so ``n_sim = (f * sd / target)^2``.  For the
    variance it is ``sd^2 * sqrt(2 / (n_sim - 1))``, which rearranges to
    ``n_sim = 1 + 2 (sd^2 / target)^2``.  The asymmetry matters: an absolute
    target on the variance is quadratic in the dispersion where the others are
    linear, so it is the measure that decides a budget whenever it is included.
    """
    if measure == "variance":
        return 1.0 + 2.0 * (frame["variance"] / target) ** 2
    factor = frame.get("factor_" + measure, 1.0)
    return (factor * frame["sd"] / target) ** 2


def usable(frame, limit=TAIL_SHARE_LIMIT):
    """Select the cells whose variance is not the artefact of a single draw."""
    return frame[frame["tail_share"] < limit]


def binding_cell(frame, target, measure="bias_mean"):
    """Return the replicate count a scenario needs, and the cell that sets it.

    Within a scenario every estimator is computed on the same replicates -- the
    design draws one dataset and hands it to all of them -- so a replicate
    count cannot be chosen per cell.  The scenario must satisfy its most
    demanding cell, and that cell is worth naming: it is the only one whose
    dispersion the budget depends on.
    """
    needed = required_replicates(frame, target, measure)
    worst = int(np.argmax(needed.to_numpy()))
    return {"n_sim_required": float(needed.iloc[worst]),
            "binding_estimator": frame["estimator"].iloc[worst],
            "binding_sd": float(frame["sd"].iloc[worst]),
            "binding_variance": float(frame["variance"].iloc[worst])}


def uniform_target(frame, n_sim, measure="bias_mean"):
    """Report what a uniform design of ``n_sim`` replicates actually delivers.

    Its worst cell, which is the only honest summary of a design that reports
    every cell in one table: a reader comparing two numbers from that table is
    limited by the noisier of them.  Using it as the common target is what
    makes the allocated design comparable to the uniform one -- the allocation
    is then required to be no worse anywhere.
    """
    if measure == "variance":
        return float((frame["variance"] * np.sqrt(2.0 / (n_sim - 1))).max())
    factor = frame.get("factor_" + measure, 1.0)
    return float((factor * frame["sd"] / np.sqrt(n_sim)).max())


def allocate(pilot_frame, cost_frame, n_sim_uniform, measure="bias_mean",
             target=None, cost_column="seconds_per_replicate"):
    """Replicate counts and cost, uniform against equal-precision, by n.

    The equal-precision allocation is also the one that minimises the worst
    Monte Carlo standard error for a given total cost: with cost linear in the
    replicate counts and each scenario's standard error falling like
    ``1/sqrt(n_sim)``, any budget that leaves two scenarios at different
    precisions can be improved by moving replicates from the precise one to the
    noisy one.  So the minimax-optimal design is the one that equalises
    precision, which is what ``binding_cell`` computes.
    """
    rows = []
    for n, by_n in pilot_frame.groupby("n", sort=True):
        sound = usable(by_n)
        common = (uniform_target(sound, n_sim_uniform, measure)
                  if target is None else target)
        for scenario, cell in sound.groupby("scenario", sort=False):
            row_cost = cost_frame.loc[
                (cost_frame["scenario"] == scenario) & (cost_frame["n"] == n)]
            seconds = float(row_cost[cost_column].iloc[0])
            needed = binding_cell(cell, common, measure)
            n_sim_allocated = int(np.ceil(needed["n_sim_required"]))
            total = int((by_n["scenario"] == scenario).sum())
            worst_tail = float(cell["tail_share"].max())
            rows.append({
                "n": int(n), "scenario": scenario, "measure": measure,
                "target_mcse": common,
                **needed,
                "n_sim_allocated": n_sim_allocated,
                "n_sim_uniform": int(n_sim_uniform),
                "worst_tail_share": worst_tail,
                "cells_used": int(len(cell)),
                "cells_excluded": total - int(len(cell)),
                "seconds_per_replicate": seconds,
                "cost_uniform": seconds * n_sim_uniform,
                "cost_allocated": seconds * n_sim_allocated,
            })
    return pd.DataFrame(rows)


def measure_comparison(pilot_frame, cost_frame, n, n_sim_uniform):
    """Compare what each reported measure asks of the replicate count.

    All four are reported from the same replicates, so the design has to serve
    the most demanding of them; a count chosen for the mean alone can leave the
    quantile measures far noisier than the table suggests.  The targets are not
    comparable across measures -- each is that measure's own worst cell under
    the uniform design -- so what the table compares is not the targets but
    whether a measure needs more or fewer replicates to hold its own.
    """
    frames = []
    for measure in CLOSED_FORM + BOOTSTRAPPED:
        allocation = allocate(pilot_frame[pilot_frame["n"] == n], cost_frame,
                              n_sim_uniform, measure)
        frames.append(allocation[["n", "scenario", "measure", "target_mcse",
                                  "binding_estimator", "n_sim_allocated",
                                  "cost_allocated"]])
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- validation

def validate(target, allocation, n, seed, out_dir):
    """Run the allocated design and check it meets the target everywhere.

    A derived replicate count is a prediction, and the prediction is testable:
    run the study at those counts and the realised standard errors should sit
    at the target, not above it.  Reported per cell rather than per scenario,
    because the claim being checked is that no cell was left noisier than the
    uniform design left its worst one.
    """
    simulation = importlib.import_module(bench_config.IMPLEMENTATIONS[target])
    tmp_path = os.path.join(out_dir, f"{target}_budget_validation.csv")
    plan = allocation[allocation["n"] == n]

    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _, row in plan.iterrows():
            scenario = row["scenario"]
            n_sim = int(row["n_sim_allocated"])
            raw = simulation.run([scenario], n, n_sim, seed, tmp_path)
            truth = config.SCENARIOS[scenario]["truth"]
            rng = np.random.default_rng(seed)
            for estimator, group in raw.groupby("estimator", sort=False):
                measured = metrics.performance(group["estimate"], truth, rng)
                rows.append({
                    "scenario": scenario, "estimator": estimator, "n": int(n),
                    "n_sim": n_sim, "target_mcse": float(row["target_mcse"]),
                    "achieved_mcse": measured["bias_mean_se"],
                    "ratio": measured["bias_mean_se"] / float(row["target_mcse"]),
                    "bias_mean": measured["bias_mean"],
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- entry

def main():
    """Size the simulation budget and write the budget tables to disk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="optimized")
    parser.add_argument("--n", action="append", type=int, dest="sizes")
    parser.add_argument("--n-sim", type=int, default=bench_config.BUDGET_N_SIM)
    parser.add_argument("--measure", default="bias_mean",
                        choices=list(CLOSED_FORM + BOOTSTRAPPED))
    parser.add_argument("--seed", type=int, default=config.MASTER_SEED)
    parser.add_argument("--out-dir", default=bench_config.BENCHMARK_DIR)
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    sizes = args.sizes or bench_config.BUDGET_N
    scenarios = list(config.SCENARIOS)
    # The sub-studies that look at one sample size use the study's own n where
    # the grid contains it, and otherwise the largest size that was run.
    at_n = config.N if config.N in sizes else max(sizes)
    os.makedirs(args.out_dir, exist_ok=True)

    def write(frame, name):
        path = os.path.join(args.out_dir, f"budget_{name}.csv")
        frame.to_csv(path, index=False)
        return path

    pilot_frame = pilot(args.target, scenarios, sizes, args.n_sim, args.seed,
                        args.out_dir)
    cost_frame = costs(args.target, scenarios, sizes,
                       bench_config.BUDGET_TIMING_N_SIM, args.seed, args.out_dir)
    scaling = variance_scaling(pilot_frame)
    # Two allocations, answering two different questions.  The first
    # recalibrates the precision target at every sample size, which is the
    # question of how to split a study at one n between its scenarios.  The
    # second holds the target fixed at what the published design achieves at
    # its own n, which is the question of what a larger n costs once the
    # replicate count is allowed to follow the precision it buys.
    allocation = allocate(pilot_frame, cost_frame, config.N_SIM, args.measure)
    study_target = uniform_target(
        usable(pilot_frame[pilot_frame["n"] == at_n]), config.N_SIM, args.measure)
    fixed = allocate(pilot_frame, cost_frame, config.N_SIM, args.measure,
                     target=study_target)
    measures = measure_comparison(pilot_frame, cost_frame, at_n, config.N_SIM)

    for frame, name in ((pilot_frame, "dispersion"), (cost_frame, "cost"),
                        (scaling, "scaling"), (allocation, "allocation"),
                        (fixed, "fixed_target"), (measures, "measures")):
        logger.info("wrote %s", write(frame, name))

    fmt = {"index": False, "float_format": lambda x: f"{x: .4f}"}
    print("\nDispersion of each cell, by sample size")
    print(pilot_frame[["scenario", "n", "estimator", "sd", "robust_sd",
                       "tail_share", "factor_bias_median",
                       "factor_iqr"]].to_string(**fmt))
    print("\nFitted variance scaling  sd^2 = a n^-q")
    print(scaling.to_string(**fmt))
    print("\nSeconds per replicate")
    print(cost_frame.to_string(index=False, float_format=lambda x: f"{x: .3e}"))
    print(f"\nReplicate counts for equal precision on {args.measure}, against "
          f"the uniform n_sim = {config.N_SIM}")
    print(allocation[["n", "scenario", "target_mcse", "binding_estimator",
                      "n_sim_uniform", "n_sim_allocated", "worst_tail_share",
                      "cost_uniform", "cost_allocated"]].to_string(**fmt))

    def totals(frame):
        table = frame.groupby("n", as_index=False)[["cost_uniform",
                                                    "cost_allocated"]].sum()
        table["saving"] = 1.0 - table["cost_allocated"] / table["cost_uniform"]
        return table

    print("\nTotal cost per sample size, uniform against allocated")
    print(totals(allocation).to_string(**fmt))

    print(f"\nHolding the target at the {args.measure} precision the uniform "
          f"design reaches at n = {at_n} ({study_target:.4f})")
    print(fixed[["n", "scenario", "binding_estimator", "n_sim_allocated",
                 "seconds_per_replicate", "cost_allocated"]].to_string(**fmt))
    print(totals(fixed).to_string(**fmt))

    print(f"\nReplicate counts each reported measure needs at n = {at_n}")
    print(measures.to_string(**fmt))

    if not args.no_validate:
        check = validate(args.target, allocation, at_n, args.seed, args.out_dir)
        logger.info("wrote %s", write(check, "validation"))
        print(f"\nAllocated design at n = {at_n}: achieved standard error "
              f"against the target")
        print(check[["scenario", "estimator", "n_sim", "target_mcse",
                     "achieved_mcse", "ratio"]].to_string(**fmt))
        print(f"\nWorst cell sits at {check['ratio'].max():.2f} times the target")


if __name__ == "__main__":
    main()
