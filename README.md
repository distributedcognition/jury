# A Jury, Not an Electorate

Code and cached expert answers for *A Jury, Not an Electorate: Label-Free
Reliability Estimation for Ensembles of Language Models*, by Alessandro Rozza,
Davide Mambelli, Steven Motta, and Juri Taresch (distributed cognition).

Rebuild the paper's tables and experimental figures from the
experts' cached answers. No language model is run, no GPU or network is needed,
and the fast path takes about ten minutes on an idle laptop.

## Quick reproduction

```bash
git clone https://github.com/distributedcognition/jury.git
cd jury
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make                                # tables and experimental figures -> tables/, figures/
make check                          # diff them against the paper's copies in reference/tables
make test                           # the assertions behind the method's claims (about a minute)
```

`make` runs the seven scripts of the fast path listed under *Main experiments*.
`make full` also recomputes the nested bootstrap, sub-pool replication, and
EM initialisation analysis. The fast path rebuilds their tables from the
shipped `results/*.json`.

## Environment

**Software.** Python 3.10 or later; the results in the paper were produced
with the versions below. Only `scripts/skillaggregation_mc.py` imports `torch`
and `transformers`, and it trains nothing unless `--retrain` is passed.

| package | version used |
|---|---|
| Python | 3.12.3 |
| numpy | 2.5.1 |
| pandas | 3.0.5 |
| scipy | 1.18.0 |
| scikit-learn | 1.9.0 |
| pyyaml | 6.0.3 |
| matplotlib | 3.11.1 |
| torch (SkillAggregation only) | 2.13.0 |
| transformers (SkillAggregation only) | 5.14.1 |

**Hardware used.** Every number in the paper's tables was computed from the
cache on a laptop with a 16-core Intel Core Ultra 9 285H and 62 GB of RAM, on
CPU. The one GPU job in the paper's pipeline, the sixteen GPT-2 fine-tunings of
`scripts/skillaggregation_mc.py`, took about ten minutes per run on the same
laptop's NVIDIA RTX 5090 GPU (24 GB; about three GPU-hours in total); its
per-epoch predictions are shipped, so reproducing the table needs no GPU.

**Total computational budget.** Fast path, 10 CPU-minutes. Long analyses:
refit bootstrap about 1 hour serial for each of its two runs, sub-pool
replication about 4 CPU-hours (32 minutes on 8 processes), SkillAggregation
about 3 GPU-hours. Building the cache itself (below) took about 66 minutes on
one H100 for the original 360 questions and their GSM8K samples, plus the
local GPU runs for the 1200 added multiple-choice questions.

## Cached expert outputs

**Provenance.** The nine experts are the instruction-tuned checkpoints named
in `config.yaml` (Qwen2.5 0.5B/1.5B/3B, Qwen3-0.6B, Qwen2.5-Math-1.5B,
Qwen2.5-Coder-1.5B, OLMo-2-1B, Granite-3.3-2B, EuroLLM-1.7B), each run under
its own chat template with Qwen3's thinking mode disabled. The 360 original
questions (six MMLU subjects and GSM8K) and the eight GSM8K samples per expert
were generated on one NVIDIA H100 80GB; the 1200 ARC-Challenge and
HellaSwag questions were answered later on a local RTX 5090 laptop GPU, reusing
the original 360 unchanged. The code that ran the models is not part of this
folder, and nothing here needs it.

**Generation settings.** Multiple choice is scored in one forward pass from the
log-probabilities of the four option letters, under a prompt that states the
question and options and asks for the letter alone; the eight "samples" per
expert on multiple choice are draws from that same distribution and cost
nothing extra. GSM8K is answered by chain-of-thought generation capped at 256
new tokens, greedy for the main answer and with temperature 0.7 and top-p 0.95
for the eight samples; the final number is taken from `answer is ...` when
present and otherwise is the last number in the completion, compared after
stripping separators and rounding at 1e-4. A completion with no number is a
non-answer. Question sampling uses a fixed seed and the question set is
committed.

**File structure.**

```text
cache/questions.csv        1560 rows: task (mc | numeric), subject, question, options A-D,
                           gold answer, qid, split (the cache's own 50/50 split; the paper's
                           ten splits are drawn by the scripts from seeds 1..10)
cache/predictions.csv      14040 rows, one per expert and question: greedy answer, correctness,
                           self-confidence (normalised probability of the chosen letter on MC)
cache/fugu_rollouts.csv    112320 rows, one per expert, question and sample (8 per pair)
cache/embeddings.npy       1560 x 384 all-MiniLM-L6-v2 embeddings of the questions, used only
                           by the supervised answer scorer's regional-reliability feature
config.yaml                the pool, the benchmark, and every method's hyperparameters
```

## Dataset text in the public cache

The 600 HellaSwag rows retain their question IDs, labels, subject, and split,
but their question and answer-option text is omitted. GitHub published a
[copyright takedown notice naming the original HellaSwag repository](https://github.com/github/dmca/blob/master/2026/09/2026-09-14-wikihow.md).
The cached predictions, embeddings, and SkillAggregation outputs are included,
so the cached analyses use the same numerical inputs as the paper.

Keep `cache/embeddings.npy` and `results/skillaggregation_runs.npz` when using
this public cache. Regenerating embeddings or retraining SkillAggregation
requires complete, lawfully obtained question and choice text in the same row
order. SkillAggregation accepts a separate complete cache through `--cache`.
The scripts report missing text instead of training on blank questions.

[Third-party notices](THIRD_PARTY_NOTICES.md) describe the dataset sources,
licenses, and changes to the cached files.

## Experimental setup, seeds and hyperparameters

Every method is evaluated over ten stratified 50/50 train/test splits of the
same cached answers, drawn with seeds 1 to 10 (`router_lab/techniques/multiseed.py`);
methods that need labels see only the training half, and the jury is fitted on
the unlabelled configurations of that same half. Tables report the mean and
standard deviation over the ten splits. Paired comparisons use a bootstrap over
questions with 20,000 resamples (seed 0), where a question's score is its mean
correctness over the splits in which it was held out. The refit bootstrap draws
2000 replicas from base seed 100000, serially.

The jury (`config.yaml`, block `experiments.jury`): EM initialised at 0.6 for
every expert, at most 200 iterations, stopped when no parameter moves by more
than 1e-7, reliabilities clipped to [0.02, 0.98], evidence weight alpha = n,
the number of samples (8 with repeated sampling, 1 on greedy answers), no
shared-error channel, one reliability per expert, candidate set read off the
pool's answers. The supervised linear fusion uses each expert's greedy training
accuracy clipped to [1e-3, 1 - 1e-3] as its reliability. The supervised answer
scorer (`experiments.aggregator`) is a logistic regression with inverse L2
strength C = 0.5 and k = 15 neighbours for its regional feature. The re-implemented
baselines and every adaptation made to them are documented in the docstrings
of `scripts/aggregator_vs_literature.py`. OW-L and OW-I (Table 8) are fitted on
the unlabelled multiple-choice questions of the training half, like ISP; OW-L's
least-squares fit of the accuracies to the agreement table runs L-BFGS-B from
five starts and keeps the lowest loss. The best single expert with
self-consistency (Table 1) answers each GSM8K question with the mode of its
eight cached samples, the same samples the distributional rules read, and each
multiple-choice question with its greedy answer; it is computed for all nine
experts and the best on the training half is scored, as for the greedy best
expert.

## Which script produces which result

| Result | Reproduction script |
|---|---|
| Tables 1–5, 7, 12, Figure 2, Appendix B.2 | `scripts/run_paper_results.py` |
| Table 6 | `scripts/refit_bootstrap.py` |
| Table 8 (SkillAggregation, OW-L, OW-I) | `scripts/skillaggregation_mc.py` |
| Table 9, Figures 3–4 | `scripts/pool_replication.py` |
| Table 10, Appendix B.6 | `scripts/em_multistart.py` |
| Table 11 | `scripts/testimony_structure.py` |
| Table 13 | `scripts/support_control.py` |
| Table 3's re-implemented baselines and their adaptations | `scripts/aggregator_vs_literature.py` |

## Main experiments

| command | paper | output | time |
|---|---|---|---|
| `python scripts/run_paper_results.py` | Tables 1–5, 7, 12, Figure 2, the plug-in of Appendix B.2 | `tables/{experts,reachability,main,decomposition,bysource,specialisation,override}.tex`, `figures/jury_frontier.png`, `results/paper_results.json` | 2 min |
| `python scripts/testimony_structure.py` | Table 11, Appendix C | `tables/testimony.tex`, `results/testimony_structure.json` | 7 min |
| `python scripts/support_control.py` | Table 13, Appendix D | `tables/support.tex`, `results/support_control.json` | 1 min |
| `python scripts/refit_bootstrap.py --table` | Table 6 | `tables/refit.tex` from the shipped `results/refit_bootstrap.json` and `results/nested_bootstrap.json` | seconds |
| `python scripts/pool_replication.py` | Table 9, Figures 3–4 | `tables/pool_replication.tex`, `figures/pool_replication*.{pdf,png}` from the shipped `results/pool_replication.json` | seconds |
| `python scripts/em_multistart.py` | Table 10, Appendix B.6 | `tables/multistart.tex` from the shipped `results/em_multistart.json` | seconds |
| `python scripts/skillaggregation_mc.py` | Table 8 | `tables/skillaggregation.tex` from the shipped `results/skillaggregation_runs.npz` | 3 min |

Every script resolves its paths from its own location, so it can be run from
any working directory. Each writes its table into `tables/` and its measurements
into `results/`, overwriting the shipped copy; `reference/tables/` is never
written. `scripts/aggregator_vs_literature.py` and `scripts/headroom_ceiling.py`
hold the re-implemented baselines and the reachability analysis and are
imported by the scripts above.

## Long-running analyses

| command | replaces | time |
|---|---|---|
| `python scripts/refit_bootstrap.py 2000` then `python scripts/refit_bootstrap.py 2000 --nested` | `results/refit_bootstrap.json`, `results/nested_bootstrap.json` (Table 6, middle and right columns) | about 1 hour each, serial; `--jobs J` parallelises but changes the bootstrap stream |
| `python scripts/pool_replication.py --recompute --jobs 8` | `results/pool_replication.json`: every rule refitted on each of the 247 pools | 32 minutes on 8 processes |
| `python scripts/em_multistart.py --recompute --jobs 10` | `results/em_multistart.json`: the robustness check of the EM start, re-running the fit from 25 initialisations on each of the ten splits in both evidence regimes (sampled distributions with α = 8, greedy answers with α = 1) | about 2 minutes on 10 processes |
| `python scripts/skillaggregation_mc.py --retrain` | `results/skillaggregation_runs.npz`: sixteen GPT-2 fine-tunings of 40 epochs (batch 8, learning rate 1e-4, four values of lambda) | about 3 hours on one RTX 5090 laptop GPU; needs `torch` and `transformers`; not bit-reproducible |

`make full` runs the fast path, recomputes the nested bootstrap, sub-pool
replication, and EM initialisation checks, then rebuilds Table 6. Run the
plain refit bootstrap separately with `python scripts/refit_bootstrap.py 2000`.

## Checking against the paper

`make check` compares each regenerated table with the paper's copy line by line.
On the fast path all thirteen tables are identical on the machine described
above. On another CPU architecture or BLAS (an Apple Silicon Mac, for instance)
the rows of the supervised rules fitted by iterative solvers (MoRE, LLM-TOPLA,
the answer scorer) and the jury's override sweep can move by a tenth of a
point, as one question in 1560 changes side; everything label-free or in
closed form is still identical. `make check` reports such tables as CLOSE
rather than DIFFERS and still exits cleanly; it fails only on a larger
difference. After `make full`, the sub-pool
replication is identical as well, on every pool and split, and so is the plain
refit bootstrap (the *Split and fit* column of Table 6). The nested bootstrap
(the *Both* column) reproduces the paper's point estimates and intervals to
within a tenth of a point rather than exactly, because the shipped
`results/nested_bootstrap.json` was drawn from a different random stream than
a serial run produces; `--jobs J` likewise gives each process its own generator.

## Tests

`python tests/test_jury.py` runs fourteen checks against `cache/` in about a
minute, with plain Python and no pytest: cached embeddings work without question
text and uncached regeneration rejects missing text; the two routers satisfy the routing
contract and never read a held-out label; the jury reads no gold answer at all,
per-task reliabilities included, and neither do OW-L and OW-I; its greedy rule is exactly the log-odds
weighted vote of Proposition 1; the distributional M-step is Equation 5; a
shared-error channel under the fractional weight fits every susceptibility to
zero and changes no verdict; collapsing the samples to their mode keeps every
sample and discards only the distribution; the sampled variants cost only the
generative slice; and the four-option support control keeps every option on
multiple choice and reads no label. `--slow` adds the ten-split check that the
shared-error channel never raises mean accuracy (about eight minutes).

## What is in the folder

```text
cache/                     the expert answers the paper is computed from (above)
config.yaml                the pool, the benchmark and every method's settings
router_lab/                the code: the jury (techniques/jury), the supervised answer scorer
                           (techniques/aggregator), the routing context and the split helper
scripts/                   the paper's pipeline, one script per analysis
tests/test_jury.py         the assertions behind the method's claims
results/                   the shipped outputs of the long analyses, reused by the fast path
reference/tables/          the paper's tables, for `make check`
tables/, figures/          written by the scripts
```

## Use of generative AI

Supporting the implementation of the code for the evaluation, we used Claude
Opus 5 (Anthropic). Every method, protocol and number was specified, checked
and run by the authors.

## What is not here

The models themselves and the code that ran them to build the cache, neither
of which enters the paper's tables. Figure 1 is a hand-drawn
diagram.
