"""Audit the baseline for numerical instability, without changing it.

Reruns replicates across a range of sample sizes with every warning turned
into a recorded event, and alongside that measures the quantities the estimators
are actually exposed to: the fitted propensities and the inverse-probability
weights built from them.  The Horvitz-Thompson and doubly robust estimators
divide by ``pi`` and by ``1 - pi`` with no trimming, so a fitted propensity
near either bound is the mechanism by which a single replicate can dominate a
Monte Carlo mean, whether or not Python emits a warning about it.
"""

import argparse
import importlib
import logging
import os
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from . import bench_config, config

logger = logging.getLogger(__name__)


def _package(target):
    return bench_config.IMPLEMENTATIONS[target].rsplit(".", 1)[0]


def fit_with_history(data, spec, dgps):
    """Refit a propensity model keeping the parts ``fit_propensity`` discards.

    ``methods.fit_propensity`` returns only the fitted probabilities, so the
    results object -- and with it ``converged`` and the iteration count -- is
    thrown away.  statsmodels sets ``converged = False`` and raises nothing
    when IRLS exhausts its 100 iterations, so a failure to converge is
    invisible to the study as submitted.  Refitting here recovers it.
    """
    import statsmodels.api as sm

    X = dgps.design_matrix(data, spec["terms"])
    fitted = sm.GLM(data["D"].to_numpy(dtype=float), X,
                    family=sm.families.Binomial()).fit()
    history = getattr(fitted, "fit_history", {}) or {}
    return fitted.predict(X), {
        "converged": bool(getattr(fitted, "converged", True)),
        "irls_iterations": int(history.get("iteration", -1)),
        "rank": int(np.linalg.matrix_rank(X)),
        "n_columns": int(X.shape[1]),
        "condition_number": float(np.linalg.cond(X)),
    }


def propensity_diagnostics(data, propensity, scenario):
    """Summarise one fitted propensity vector and the weights it produces."""
    treatment = data["D"].to_numpy(dtype=float)
    edge = min(propensity.min(), 1.0 - propensity.max())

    # Scenario A inverts pi only among the observed; scenario B inverts pi for
    # the treated and 1 - pi for the controls.  Weighting each unit by the
    # bound it actually meets avoids counting a small 1 - pi against an
    # estimator that never divides by it.
    if scenario == "A":
        weights = treatment / propensity
    else:
        weights = treatment / propensity + (1.0 - treatment) / (1.0 - propensity)

    return {
        "min_propensity": float(propensity.min()),
        "max_propensity": float(propensity.max()),
        "closest_to_bound": float(edge),
        "n_extreme": int(np.sum((propensity < bench_config.EXTREME_PROPENSITY) |
                                (propensity > 1.0 - bench_config.EXTREME_PROPENSITY))),
        "max_weight": float(np.max(weights)),
        "effective_sample_size": float(weights.sum() ** 2 / np.sum(weights ** 2)),
    }


def audit(target, scenarios, sizes, n_sim, seed):
    """Replay replicates, recording every warning and propensity diagnostic."""
    package = _package(target)
    simulation = importlib.import_module(f"{package}.simulation")
    dgps = importlib.import_module(f"{package}.dgps")
    methods = importlib.import_module(f"{package}.methods")

    warning_rows = []
    diagnostic_rows = []
    all_scenarios = list(config.SCENARIOS)

    for n in sizes:
        for scenario in scenarios:
            spec = config.SCENARIOS[scenario]
            cells = simulation.cells(scenario)
            ps_names = sorted({ps for _, _, ps, _ in cells if ps})

            # Same seed derivation as the study, so an event found here names a
            # replicate that really occurs in the submitted run.
            parent = np.random.SeedSequence(seed).spawn(len(all_scenarios))
            seeds = parent[all_scenarios.index(scenario)].spawn(n_sim)

            counts = Counter()
            for replicate in range(n_sim):
                rng = np.random.default_rng(seeds[replicate])
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    np.seterr(all="warn")

                    data = dgps.generate(scenario, n, rng)
                    for ps_name in ps_names:
                        propensity, fit = fit_with_history(data, spec[ps_name], dgps)
                        diagnostic_rows.append({
                            "target": target, "scenario": scenario, "n": n,
                            "replicate": replicate, "ps_model": ps_name,
                            **fit,
                            **propensity_diagnostics(data, propensity, scenario),
                        })
                    for _, family, ps_name, or_name in cells:
                        methods.estimate(data, scenario, family, ps_name, or_name)

                for warning in caught:
                    counts[(type(warning.message).__name__,
                            str(warning.message)[:90])] += 1

            for (kind, message), count in counts.items():
                warning_rows.append({"target": target, "scenario": scenario,
                                     "n": n, "n_sim": n_sim, "warning": kind,
                                     "message": message, "count": count})
            logger.info("%s scenario %s n=%-5d %d warnings over %d replicates",
                        target, scenario, n, sum(counts.values()), n_sim)

    return pd.DataFrame(warning_rows), pd.DataFrame(diagnostic_rows)


def summarise(diagnostics):
    """Per condition: how often the fit got close to a bound, and how close."""
    diagnostics = diagnostics.assign(
        rank_deficient=diagnostics["rank"] < diagnostics["n_columns"])
    grouped = diagnostics.groupby(["target", "scenario", "n", "ps_model"], sort=False)
    return grouped.agg(
        replicates=("replicate", "nunique"),
        worst_edge=("closest_to_bound", "min"),
        median_edge=("closest_to_bound", "median"),
        pct_replicates_with_extreme=("n_extreme", lambda s: 100.0 * (s > 0).mean()),
        pct_not_converged=("converged", lambda s: 100.0 * (~s.astype(bool)).mean()),
        max_irls_iterations=("irls_iterations", "max"),

        max_weight=("max_weight", "max"),
        median_max_weight=("max_weight", "median"),
        min_effective_n=("effective_sample_size", "min"),
        worst_condition_number=("condition_number", "max"),
        pct_rank_deficient=("rank_deficient", lambda s: 100.0 * s.mean()),
    ).reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="baseline",
                        choices=sorted(bench_config.IMPLEMENTATIONS))
    parser.add_argument("--n", action="append", type=int, dest="sizes")
    parser.add_argument("--n-sim", type=int, default=bench_config.STABILITY_N_SIM)
    parser.add_argument("--seed", type=int, default=config.MASTER_SEED)
    parser.add_argument("--scenario", action="append", dest="scenarios")
    parser.add_argument("--out-dir", default=bench_config.BENCHMARK_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    sizes = args.sizes or bench_config.STABILITY_N
    scenarios = args.scenarios or list(config.SCENARIOS)

    warning_table, diagnostics = audit(args.target, scenarios, sizes,
                                       args.n_sim, args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    prefix = os.path.join(args.out_dir, f"stability_{args.target}")
    warning_table.to_csv(f"{prefix}_warnings.csv", index=False)
    diagnostics.to_csv(f"{prefix}_diagnostics.csv", index=False)
    summary = summarise(diagnostics)
    summary.to_csv(f"{prefix}_summary.csv", index=False)
    logger.info("wrote %s_{warnings,diagnostics,summary}.csv", prefix)

    print("\nWarnings raised")
    if warning_table.empty:
        print("  none across any condition")
    else:
        print(warning_table.to_string(index=False))

    print("\nFitted propensities and inverse-probability weights")
    print(summary.to_string(index=False, float_format=lambda x: f"{x: .4g}"))


if __name__ == "__main__":
    main()
