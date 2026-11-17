PYTHON := python3

RAW     := results/raw/estimates.csv
SUMMARY := results/raw/summary.csv
FIGURES := results/figures/.stamp

BENCH   := results/benchmarks
PROFILES := results/profiles

# Progress bars are noise in a logged performance run, and formatting them
# costs time that would land in the measurement.
PERF := TQDM_DISABLE=1 $(PYTHON)

.PHONY: all simulate analyze figures test clean baseline-reference \
        profile complexity benchmark parallel stability-check \
        perf-figures performance regression budget clean-perf

all: figures

simulate: $(RAW)

$(RAW): src/simulation.py src/dgps.py src/methods.py src/config.py
	$(PYTHON) -m src.simulation --out $@

analyze: $(SUMMARY)

$(SUMMARY): $(RAW) src/analyze.py src/metrics.py
	$(PYTHON) -m src.analyze --raw $(RAW) --out $@

figures: $(FIGURES)

$(FIGURES): $(RAW) $(SUMMARY) src/figures.py
	$(PYTHON) -m src.figures --raw $(RAW) --summary $(SUMMARY) --outdir results/figures
	touch $@

test:
	$(PYTHON) -m pytest -q tests

# ---------------------------------------------------------------- performance

# TARGET selects the implementation to measure: `baseline` is the frozen Unit 2
# code, `optimized` the current src.  Override on the command line, e.g.
#   make profile TARGET=optimized
TARGET ?= baseline

profile:
	$(PERF) -m src.profile_run --target $(TARGET) --out-dir $(PROFILES)

complexity:
	$(PERF) -m src.complexity --target baseline --target optimized \
	        --out-dir $(BENCH)

benchmark:
	$(PERF) -m src.benchmark --out-dir $(BENCH)

parallel:
	$(PERF) -m src.simulation --out $(RAW) --jobs -1

stability-check:
	$(PERF) -m src.stability --target $(TARGET) --out-dir $(BENCH)

# The optional extension of docs/EXTENSIONS.md: how many replicates each
# scenario needs, and what the uniform design spends to no purpose.
budget:
	$(PERF) -m src.budget --target optimized --out-dir $(BENCH)

perf-figures:
	$(PERF) -m src.perf_figures --bench-dir $(BENCH) --outdir results/figures

# The whole performance study, in the order the documents report it.
performance: profile complexity benchmark stability-check budget perf-figures

regression:
	$(PYTHON) -m pytest -q tests/test_regression.py

# Rebuild the committed Unit 2 reference from the frozen code in baseline/.
# Rarely needed -- the file is in the repo -- but it is what makes that file
# reproducible rather than a number nobody can check.
baseline-reference:
	$(PERF) -m baseline.simulation --out results/baseline/estimates.csv

# The figures `make all` can rebuild in seconds, and the ones it cannot.
# `clean` removes only the first kind: the performance figures cost minutes of
# benchmarking to redraw, so deleting them with the study's own outputs would
# make `make clean` a far more expensive command than it looks.
STUDY_FIGURES := bias_grid distributions extension
PERF_FIGURES  := benchmark complexity speedup budget

clean:
	rm -f results/raw/*.csv results/raw/*.pkl
	rm -f $(addprefix results/figures/,$(addsuffix .png,$(STUDY_FIGURES)))
	rm -f results/figures/.stamp
	rm -rf __pycache__ src/__pycache__ baseline/__pycache__ tests/__pycache__ \
	       .pytest_cache

clean-perf:
	rm -f $(BENCH)/*.csv $(PROFILES)/*.prof $(PROFILES)/*.txt
	rm -f $(addprefix results/figures/,$(addsuffix .png,$(PERF_FIGURES)))
