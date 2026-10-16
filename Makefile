# Rebuild the paper's tables and figures from the cached expert answers.
#   make            -> the fast path: every table and figure, about ten minutes
#   make check      -> diff what was produced against reference/tables
#   make test       -> the assertions the paper points to
#   make full       -> also recompute the two long analyses (about 1.5 h)
PY ?= python3

.PHONY: all fast full check test clean

all: fast

fast:
	$(PY) scripts/run_paper_results.py
	$(PY) scripts/testimony_structure.py
	$(PY) scripts/support_control.py
	$(PY) scripts/refit_bootstrap.py --table
	$(PY) scripts/pool_replication.py
	$(PY) scripts/em_multistart.py
	$(PY) scripts/skillaggregation_mc.py

full: fast
	$(PY) scripts/refit_bootstrap.py 2000 --nested
	$(PY) scripts/pool_replication.py --recompute
	$(PY) scripts/em_multistart.py --recompute
	$(PY) scripts/refit_bootstrap.py --table

check:
	$(PY) check_against_paper.py

test:
	$(PY) tests/test_jury.py

clean:
	rm -rf tables figures router_lab/__pycache__ router_lab/techniques/__pycache__ \
	       router_lab/techniques/*/__pycache__ scripts/__pycache__ tests/__pycache__
