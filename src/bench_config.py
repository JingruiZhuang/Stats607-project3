"""Settings for the performance work, kept apart from the study's own design.

`config.py` holds the simulation design and must not change for the sake of
speed; this module holds only the knobs of the measurement harness, so a
profiling or timing choice can never be mistaken for a design choice.
"""

# The implementations the harness can time.  Each names an importable module
# exposing `run(scenarios, n, n_sim, master_seed, out_path)`.
IMPLEMENTATIONS = {
    "baseline": "baseline.simulation",
    "optimized": "src.simulation",
}

# Sample sizes for the complexity study.  Geometric so that a log-log fit has
# evenly spaced abscissae, and reaching 4000 because the published design's
# n = 500 is too small for the asymptotic regime to show.
COMPLEXITY_N = [125, 250, 500, 1000, 2000, 4000, 8000, 16000]

# Replicates per point in the complexity study: enough to average over the
# noise of a single draw, few enough that the sweep stays minutes not hours.
COMPLEXITY_N_SIM = 25

# Replicates for the profiling run.  cProfile's per-call overhead inflates the
# total, so this is deliberately smaller than the study; the shares of time it
# attributes are what matters, not the absolute seconds.
PROFILE_N_SIM = 200
PROFILE_N = 500

# Conditions swept by the stability check: the study's own n, plus smaller ones
# where the false propensity models are most likely to produce fitted
# probabilities near zero or one, and larger ones for contrast.
STABILITY_N = [50, 100, 250, 500, 1000]
STABILITY_N_SIM = 200

# Fitted propensities this close to 0 or 1 make the inverse-probability weights
# large enough to dominate an estimate; counted, not altered, in the baseline.
EXTREME_PROPENSITY = 1e-3

PROFILE_DIR = "results/profiles"
BENCHMARK_DIR = "results/benchmarks"

# ---------------------------------------------------------------- budget study

# Sample sizes for the simulation-budget study (docs/EXTENSIONS.md).  The same
# geometric shape as the complexity grid, stopping at 4000: the question there
# is how the estimators' sampling variance falls with n, and that is visible
# well before the sizes needed to see the asymptotic runtime exponent.
BUDGET_N = [125, 250, 500, 1000, 2000, 4000]

# Replicates per pilot point.  The relative standard error of an estimated
# variance is sqrt(2 / (m - 1)), so m = 2000 pins each cell's dispersion to
# about 3%, and the replicate counts derived from it to about 6%.
BUDGET_N_SIM = 2000

# Replicates used only for the per-replicate cost model; the timing needs far
# fewer draws than the variance does.
BUDGET_TIMING_N_SIM = 200

# Bootstrap sizes swept when asking how many resamples the median and
# interquartile-range standard errors actually need, and how many independent
# bootstrap seeds each size is judged over.
BUDGET_N_BOOTSTRAP = [125, 250, 500, 1000, 2000, 4000]
BUDGET_BOOTSTRAP_SEEDS = 12

# Prefix lengths of the pilot used to ask how large a pilot has to be before
# the replicate counts it recommends stop moving.
BUDGET_PILOT_SIZES = [50, 100, 200, 500, 1000, 2000]
