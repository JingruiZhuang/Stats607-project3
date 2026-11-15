"""Tests for the simulation-budget analysis of docs/EXTENSIONS.md.

The budget module makes claims that are arithmetic, not empirical: that a
replicate count derived from a dispersion really does deliver the standard
error it promises, that the equal-precision allocation is the one that
minimises the worst standard error for a given cost, and that a cell whose
variance is the artefact of one draw is kept out of the calculation.  Each of
those is checked here against data with known answers, so that a change to the
derivation fails loudly rather than quietly recommending the wrong design.
"""

import numpy as np
import pandas as pd
import pytest

from src import budget


@pytest.fixture
def frame():
    return pd.DataFrame({
        "scenario": ["A", "A", "B", "B"],
        "estimator": ["OR", "HT", "OR", "DR.ofal"],
        "n": 500,
        "sd": [0.20, 0.35, 0.28, 0.56],
        "variance": np.array([0.20, 0.35, 0.28, 0.56]) ** 2,
        "tail_share": [0.004, 0.005, 0.006, 0.02],
        "factor_bias_mean": 1.0,
        "factor_bias_median": [1.25, 1.25, 1.25, 1.25],
        "factor_iqr": [1.57, 1.57, 1.57, 1.57],
    })


# ------------------------------------------------------------------ the maths

def test_required_replicates_inverts_the_standard_error(frame):
    """n_sim = (sd / target)^2 is exactly the count that reaches the target."""
    target = 0.01
    needed = budget.required_replicates(frame, target, "bias_mean")
    assert np.allclose(frame["sd"] / np.sqrt(needed), target)


def test_required_replicates_for_the_variance_uses_its_own_formula(frame):
    """The variance's standard error is sd^2 sqrt(2/(n_sim-1)), not sd/sqrt(n)."""
    target = 0.05
    needed = budget.required_replicates(frame, target, "variance")
    achieved = frame["variance"] * np.sqrt(2.0 / (needed - 1))
    assert np.allclose(achieved, target)


def test_the_quantile_measures_carry_their_shape_factor(frame):
    """A median standard error 1.25 times the mean's needs 1.25^2 the replicates."""
    ratio = (budget.required_replicates(frame, 0.01, "bias_median")
             / budget.required_replicates(frame, 0.01, "bias_mean"))
    assert np.allclose(ratio, 1.25 ** 2)


def test_a_tighter_target_costs_quadratically(frame):
    """Halving the target standard error quadruples the replicate count."""
    loose = budget.required_replicates(frame, 0.02, "bias_mean")
    tight = budget.required_replicates(frame, 0.01, "bias_mean")
    assert np.allclose(tight / loose, 4.0)


# ------------------------------------------------------- the binding constraint

def test_the_scenario_is_sized_by_its_widest_cell(frame):
    """Replicates are shared within a scenario, so the worst cell decides."""
    for scenario, widest in (("A", "HT"), ("B", "DR.ofal")):
        found = budget.binding_cell(frame[frame["scenario"] == scenario], 0.01)
        assert found["binding_estimator"] == widest


def test_the_binding_count_is_enough_for_every_cell_in_the_scenario(frame):
    target = 0.015
    for scenario, cell in frame.groupby("scenario"):
        n_sim = budget.binding_cell(cell, target)["n_sim_required"]
        assert (cell["sd"] / np.sqrt(n_sim) <= target + 1e-12).all()


def test_uniform_target_is_the_worst_cell_not_the_average(frame):
    reported = budget.uniform_target(frame, 1000, "bias_mean")
    assert reported == pytest.approx(frame["sd"].max() / np.sqrt(1000))


# -------------------------------------------------------------- the allocation

def test_equal_precision_beats_any_reallocation_of_the_same_budget(frame):
    """The allocation is minimax-optimal: no reshuffle lowers the worst error.

    Checked directly rather than argued.  Replicates are moved from one
    scenario to the other at constant cost, and the larger of the two
    scenarios' standard errors is recomputed; the equal-precision split should
    be the minimum of that curve.
    """
    cost = pd.DataFrame({"scenario": ["A", "B"], "n": [500, 500],
                         "seconds_per_replicate": [3.0e-4, 5.0e-4]})
    plan = budget.allocate(frame, cost, 1000, "bias_mean")
    counts = dict(zip(plan["scenario"], plan["n_sim_allocated"]))
    seconds = dict(zip(cost["scenario"], cost["seconds_per_replicate"]))
    widest = frame.groupby("scenario")["sd"].max().to_dict()

    budgeted = sum(counts[s] * seconds[s] for s in counts)

    def worst_error(n_sim_a):
        n_sim_b = (budgeted - n_sim_a * seconds["A"]) / seconds["B"]
        if n_sim_b <= 1:
            return np.inf
        return max(widest["A"] / np.sqrt(n_sim_a),
                   widest["B"] / np.sqrt(n_sim_b))

    at_plan = worst_error(counts["A"])
    for shift in (-200, -100, -50, 50, 100, 200):
        assert worst_error(counts["A"] + shift) >= at_plan - 1e-9


def test_a_cell_carried_by_one_draw_is_excluded_and_counted(frame):
    """A single exploded replicate must not be allowed to set the budget."""
    exploded = frame.copy()
    exploded.loc[exploded["estimator"] == "OR", ["sd", "variance", "tail_share"]] = [
        1e6, 1e12, 0.97]
    cost = pd.DataFrame({"scenario": ["A", "B"], "n": [500, 500],
                         "seconds_per_replicate": [3.0e-4, 5.0e-4]})

    plan = budget.allocate(exploded, cost, 1000, "bias_mean")
    assert (plan["cells_excluded"] == 1).all()
    assert plan["binding_estimator"].tolist() == ["HT", "DR.ofal"]
    # And the target is the one the sound cells imply, not the exploded one.
    assert plan["target_mcse"].iloc[0] == pytest.approx(0.56 / np.sqrt(1000))


# ---------------------------------------------------------------- the pilot

def test_dispersion_reports_the_tail_share_of_an_exploded_cell():
    raw = pd.DataFrame({"estimator": ["DR"] * 100,
                        "estimate": [0.1] * 99 + [1e6]})
    measured = budget.dispersion(raw, "B", 500, np.random.default_rng(0))
    assert measured["tail_share"].iloc[0] > 0.98
    # The interquartile range is untouched by the single draw, which is the
    # point of reporting it beside the standard deviation.
    assert measured["robust_sd"].iloc[0] == pytest.approx(0.0, abs=1e-12)


def test_the_shape_factors_match_the_normal_distribution():
    """For a normal sample the median's standard error is 1.253 sd/sqrt(n)."""
    values = np.random.default_rng(11).normal(size=20000)
    raw = pd.DataFrame({"estimator": "OR", "estimate": values})
    measured = budget.dispersion(raw, "A", 500, np.random.default_rng(0))
    assert measured["factor_bias_median"].iloc[0] == pytest.approx(1.2533, rel=0.1)
