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
