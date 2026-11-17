"""The optimizations preserve the study's results.

The Unit 2 estimates are committed at ``results/baseline/estimates.csv`` and
are the reference throughout.  Three kinds of check, in increasing order of
how much they would tolerate:

1. **Exact.** The data-generating process is bit-identical, and the optimized
   code reproduces itself exactly under any batch size, worker count or subset
   of scenarios.  Nothing here is allowed to differ at all.
2. **Numerical.** Every one of the 20,000 estimates agrees with its Unit 2
   counterpart far inside the tolerance that matters, which is set by the
   Monte Carlo standard error rather than chosen for convenience.
3. **Statistical.** The sampling distribution of each cell is unchanged, and
   all sixteen cells still reproduce their published Table 2 values.

Why not demand exact equality of the estimates too: the optimized code solves
the logistic fit itself rather than calling statsmodels, and converges on a
tighter criterion, so it lands nearer the exact maximum likelihood estimate
than the baseline did.  The difference is a property of the solver, not of the
study.  ``docs/OPTIMIZATION.md`` sets out the argument in full.
"""

import os

import numpy as np
import pandas as pd
import pytest

from baseline import dgps as baseline_dgps
from src import analyze, config, dgps, methods, simulation

REFERENCE = "results/baseline/estimates.csv"

# An estimate may differ from its Unit 2 counterpart by at most this much.
# Chosen to be far below anything that could matter: the smallest Monte Carlo
# standard error in the study is about 0.006, so this is some seven orders of
# magnitude inside the noise the study is already reporting.  The measured
# worst case is about 5e-10; the margin is there so that a change of machine or
# BLAS does not fail the suite for a reason that is not a regression.
TOLERANCE = 1e-7

# Estimates must agree with Unit 2 to this many significant figures in the
# summary table, which is two more than the three decimal places reported.
SUMMARY_TOLERANCE = 1e-6


@pytest.fixture(scope="module")
def reference():
    """The committed Unit 2 estimates, rebuilt from ``baseline/`` if absent.

    Rebuilt rather than skipped.  These are the checks that the optimizations
    preserved the study's results, and a correctness test that quietly does
    not run is worse than one that takes twenty seconds: a skip looks like a
    pass in a summary line.
    """
    if not os.path.exists(REFERENCE):
        from baseline import simulation as baseline_simulation
        os.makedirs(os.path.dirname(REFERENCE), exist_ok=True)
        baseline_simulation.run(list(config.SCENARIOS), config.N, config.N_SIM,
                                config.MASTER_SEED, REFERENCE)
    return pd.read_csv(REFERENCE)


@pytest.fixture(scope="module")
def optimized(tmp_path_factory, reference):
    """The same study, rerun with the optimized code."""
    out = tmp_path_factory.mktemp("raw") / "estimates.csv"
    return simulation.run(list(config.SCENARIOS), config.N, config.N_SIM,
                          config.MASTER_SEED, str(out))


@pytest.fixture(scope="module")
def paired(reference, optimized):
    """Unit 2 and optimized estimates, matched replicate by replicate."""
    keys = ["scenario", "replicate", "estimator"]
    merged = reference.merge(optimized, on=keys, suffixes=("_old", "_new"))
    assert len(merged) == len(reference) == len(optimized)
    return merged


# ------------------------------------------------------------------- exact

def test_dgp_is_bit_identical():
    """The optimized DGP draws exactly the data the Unit 2 DGP drew.

    This is the foundation of every other comparison: if the data differed at
    all, a difference in the estimates could not be attributed to the solver.
    """
    for scenario in config.SCENARIOS:
        for seed in (1, 17, 2024):
            old = baseline_dgps.generate(scenario, 500, np.random.default_rng(seed))
            new = dgps.generate(scenario, 500, np.random.default_rng(seed))
            assert list(old.columns) == list(new.columns)
            for column in old.columns:
                assert np.array_equal(old[column].to_numpy(),
                                      new[column].to_numpy(), equal_nan=True), \
                    (scenario, seed, column)


def test_batched_dgp_matches_one_at_a_time():
    """A batch of replicates holds exactly the datasets drawn individually."""
    seeds = np.random.SeedSequence(99).spawn(6)
    batch = dgps.generate_columns(
        "B", 200, [np.random.default_rng(s) for s in seeds], n_sim=len(seeds))
    for i, seed in enumerate(seeds):
        one = dgps.generate("B", 200, np.random.default_rng(seed))
        for column in one.columns:
            assert np.array_equal(one[column].to_numpy(), batch[column][i],
                                  equal_nan=True), (i, column)


@pytest.mark.parametrize("batch_size", [1, 7, 250])
def test_batch_size_does_not_change_results(tmp_path, batch_size):
    """Results are a property of the replicate, not of how it was grouped.

    The batched logistic fit iterates until its slowest replicate converges,
    so a replicate that converged early must be frozen rather than carried
    along; otherwise its answer would depend on which others shared its batch.
    """
    def go(size):
        return simulation.run(["A", "B"], config.N, 24, config.MASTER_SEED,
                              str(tmp_path / f"{size}.csv"), batch_size=size)

    assert go(batch_size).equals(go(250))


def test_parallel_matches_serial(tmp_path):
    """Worker count changes nothing, including how batches are split."""
    serial = simulation.run(["A", "B"], config.N, 24, config.MASTER_SEED,
                            str(tmp_path / "serial.csv"), batch_size=7)
    parallel = simulation.run(["A", "B"], config.N, 24, config.MASTER_SEED,
                              str(tmp_path / "parallel.csv"), jobs=3,
                              batch_size=7)
    assert serial.equals(parallel)


def test_seed_semantics_preserved(tmp_path):
    """Unit 2's reproducibility guarantees still hold.

    A scenario run on its own draws the data it draws in a full run, and a
    short run reproduces the leading replicates of a long one.
    """
    def go(n_sim, scenarios):
        return simulation.run(scenarios, config.N, n_sim, config.MASTER_SEED,
                              str(tmp_path / f"{n_sim}-{''.join(scenarios)}.csv"))

    both = go(6, ["A", "B"])
    assert both[both.scenario == "B"].reset_index(drop=True).equals(go(6, ["B"]))
    assert both[both.replicate < 3].reset_index(drop=True).equals(go(3, ["A", "B"]))


def test_row_order_matches_unit_2(paired, reference, optimized):
    """The output file is comparable with Unit 2's row for row."""
    keys = ["scenario", "replicate", "estimator"]
    assert optimized[keys].equals(reference[keys])


# --------------------------------------------------------------- numerical

def test_every_estimate_matches(paired):
    """No estimate moved by more than the tolerance."""
    difference = (paired["estimate_new"] - paired["estimate_old"]).abs()
    worst = paired.loc[difference.idxmax(), ["scenario", "estimator"]].to_dict()
    assert difference.max() < TOLERANCE, (difference.max(), worst)


def test_differences_are_negligible_against_monte_carlo_error(paired, reference):
    """The disagreement is orders of magnitude below the study's own noise.

    The point of the tolerance is not that it is small in the abstract but that
    it is small compared with the uncertainty the study already reports.  A
    difference that is a millionth of a Monte Carlo standard error cannot
    change any conclusion drawn from these estimates.
    """
    difference = (paired["estimate_new"] - paired["estimate_old"]).abs()
    monte_carlo_se = (reference.groupby(["scenario", "estimator"])["estimate"]
                      .std(ddof=1) / np.sqrt(config.N_SIM))
    assert difference.max() < 1e-4 * monte_carlo_se.min()


def test_no_estimate_became_non_finite(optimized):
    """Nothing turned into a NaN or an infinity."""
    assert np.isfinite(optimized["estimate"].to_numpy()).all()


# -------------------------------------------------------------- statistical

def _two_sample_ks(a, b):
    """Two-sample Kolmogorov-Smirnov statistic and its asymptotic p-value.

    Written out rather than imported so that the test suite does not acquire a
    SciPy dependency for one statistic.
    """
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    grid = np.concatenate([a, b])
    statistic = np.max(np.abs(np.searchsorted(a, grid, side="right") / a.size
                              - np.searchsorted(b, grid, side="right") / b.size))

    effective = np.sqrt(a.size * b.size / (a.size + b.size))
    scaled = (effective + 0.12 + 0.11 / effective) * statistic
    terms = np.arange(1, 101)
    p_value = 2.0 * np.sum((-1.0) ** (terms - 1)
                           * np.exp(-2.0 * terms ** 2 * scaled ** 2))
    return statistic, float(np.clip(p_value, 0.0, 1.0))


def test_ks_detects_a_real_shift():
    """The KS test used below has power against a shift of the size that would
    matter, so passing it is evidence rather than a formality."""
    rng = np.random.default_rng(3)
    a = rng.normal(0.0, 0.3, config.N_SIM)
    b = rng.normal(0.05, 0.3, config.N_SIM)          # a sixth of a standard deviation
    assert _two_sample_ks(a, b)[1] < 0.05
    assert _two_sample_ks(a, a + 1e-9)[1] > 0.99


def test_sampling_distributions_unchanged(reference, optimized):
    """Each cell's distribution of estimates is the same as in Unit 2.

    Unpaired, so this would catch a change that the replicate-by-replicate
    comparison could not: one that moved the estimates around without moving
    any of them far.
    """
    for keys, group in reference.groupby(["scenario", "estimator"], sort=False):
        other = optimized[(optimized.scenario == keys[0])
                          & (optimized.estimator == keys[1])]
        statistic, p_value = _two_sample_ks(group["estimate"], other["estimate"])
        assert p_value > 0.01, (keys, statistic, p_value)


def test_summary_table_matches(reference, optimized):
    """Every number in the published comparison table is unchanged."""
    old = analyze.summarise(reference).set_index(["scenario", "estimator"])
    new = analyze.summarise(optimized).set_index(["scenario", "estimator"])
    assert old.index.equals(new.index)

    for measure in ("bias_mean", "bias_median", "variance", "iqr"):
        gap = (new[measure] - old[measure]).abs()
        assert gap.max() < SUMMARY_TOLERANCE, (measure, gap.idxmax(), gap.max())


def test_table_2_still_reproduces(optimized):
    """All sixteen reproduction cells still land on their published values.

    The same check the Unit 2 suite applies, restated against the optimized
    estimates: this is the study's actual claim, and it is what an optimization
    must not break.
    """
    summary = analyze.summarise(optimized)
    core = summary[summary["published_bias_mean"].notna()]
    assert len(core) == 16

    for measure in ("bias_mean", "bias_median", "variance", "iqr"):
        gap = (core[measure] - core["published_" + measure]).abs()
        tolerance = 4.0 * core[measure + "_se"] + 0.005
        failures = core.loc[gap > tolerance, ["scenario", "estimator"]]
        assert failures.empty, f"{measure} off in {failures.to_dict('records')}"

    assert core["z_bias_mean"].abs().max() < 3.0


# ------------------------------------------------------- the new safeguards

def test_logistic_fit_matches_statsmodels():
    """The hand-written logistic fit agrees with the one it replaced."""
    import statsmodels.api as sm

    for scenario in config.SCENARIOS:
        spec = config.SCENARIOS[scenario]
        for name in ("ps_true", "ps_false"):
            data = dgps.generate(scenario, 2000, np.random.default_rng(55))
            X = dgps.design_matrix(data, spec[name]["terms"])
            reference_fit = sm.GLM(data["D"].to_numpy(dtype=float), X,
                                   family=sm.families.Binomial()).fit()
            ours = methods.fit_propensity(data, spec[name])
            assert np.max(np.abs(ours - reference_fit.predict(X))) < 1e-8, \
                (scenario, name)


def test_expit_is_overflow_free():
    """The logistic function returns the right answer where the naive form
    overflows, which is the case a fitted linear predictor can reach."""
    extreme = np.array([-1e4, -800.0, -745.0, 0.0, 745.0, 800.0, 1e4])
    with np.errstate(over="raise", under="ignore"):
        values = dgps.expit(extreme)
    assert np.all(np.isfinite(values))
    assert np.all((values >= 0.0) & (values <= 1.0))
    assert values[0] == 0.0 and values[-1] == 1.0
    assert abs(values[3] - 0.5) < 1e-15
    # and it still agrees with the naive form wherever that one works
    safe = np.linspace(-30.0, 30.0, 101)
    assert np.allclose(dgps.expit(safe), 1.0 / (1.0 + np.exp(-safe)), atol=1e-15)


def test_non_convergence_is_reported():
    """A fit that cannot converge is counted, not silently returned.

    The baseline's statsmodels call set ``converged = False`` on a results
    object that was discarded, so this failure mode was invisible.  Perfect
    separation is the standard way to produce it: with the outcome a function
    of the single covariate, the likelihood has no maximum and the coefficient
    diverges.
    """
    separated = np.linspace(-1.0, 1.0, 40)
    X = np.column_stack([np.ones_like(separated), separated])
    outcome = (separated > 0).astype(float)

    report = methods.ConvergenceReport()
    fitted = methods.fit_logistic(X, outcome, report)

    assert report.fits == 1
    assert report.failures == 1, report
    assert np.all(np.isfinite(fitted))
    with pytest.warns(RuntimeWarning, match="did not converge"):
        report.warn_if_needed()


def test_fitted_probabilities_stay_inside_the_unit_interval():
    """Clipping keeps a fitted probability strictly between 0 and 1, so that
    the inverse-probability weights are finite however extreme the fit."""
    separated = np.linspace(-1.0, 1.0, 40)
    X = np.column_stack([np.ones_like(separated), separated])
    fitted = methods.fit_logistic(X, (separated > 0).astype(float))

    assert np.all(fitted > 0.0) and np.all(fitted < 1.0)
    assert np.all(np.isfinite(1.0 / fitted))
    assert np.all(np.isfinite(1.0 / (1.0 - fitted)))


def test_study_run_reports_its_convergence(tmp_path, caplog):
    """A run says how its logistic fits behaved, which Unit 2 never did."""
    import logging

    with caplog.at_level(logging.INFO, logger="src.simulation"):
        simulation.run(["B"], config.N, 20, config.MASTER_SEED,
                       str(tmp_path / "out.csv"))
    messages = " ".join(record.message for record in caplog.records)
    assert "logistic fits" in messages and "did not converge" in messages
