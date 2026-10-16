"""The support control: does the result depend on reading the candidate set off
the pool's answers?

    python scripts/support_control.py

The jury's candidate set V(q) is the set of answers the pool actually produced,
and its emission model is defined on that support. On multiple choice the task's
answer space is known before anyone answers, so the support need not be
reconstructed: this control fixes V(q) = {A, B, C, D} on every multiple-choice
question, so that m = 3 always, an unproposed option is a candidate with zero
mass, and a unanimous question is no longer a singleton and enters EM. Every
parameter is re-estimated under the fixed support; nothing is changed at
prediction time only. GSM8K keeps the observed support, since its answer space
is unbounded. The supervised linear fusion and Optimal Weight are run under the
same convention, so the control applies to both sides of the supervision line.

Writes tables/support.tex and results/support_control.json. Nothing here
runs a language model.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]          # this supplementary folder
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from run_paper_results import (SEEDS, TABLEDIR, bootstrap, fmt)            # noqa: E402
from aggregator_vs_literature import (answers_matrix, run_ow, run_ow_cand,  # noqa: E402
                                      run_soft_ow)
from router_lab.techniques.aggregator import rollout_distributions          # noqa: E402
from router_lab.techniques.base import build_context                        # noqa: E402
from router_lab.techniques.multiseed import _resplit                        # noqa: E402
from router_lab.techniques.runner import fit_router                         # noqa: E402
from router_lab.utils import load_config                                    # noqa: E402

# name -> (table label, needs labels)
RULES = {
    "OW": ("Optimal Weight, greedy answers", "yes"),
    "jury": ("Jury (ours), greedy answers", "no"),
    "soft-OW": ("Supervised linear fusion", "yes"),
    "jury+rollouts": ("Jury (ours) + repeated sampling", "no"),
}
SUPPORTS = ("observed", "task")


def measure(cfg, df, preds, cache):
    perq = defaultdict(lambda: defaultdict(list))   # (rule, support) -> qid -> [0/1]
    acc = defaultdict(list)                          # (rule, support) -> per-split accuracy
    acc_mc = defaultdict(list)
    pi = defaultdict(list)
    n_unanimous = []
    dist, n_samples = rollout_distributions(cache, {e["id"] for e in cfg["experts"]})

    for seed in SEEDS:
        d = df.copy()
        d["split"] = _resplit(d, cfg["dataset"]["train_frac"], seed)
        cfg.raw["seed"] = seed
        ctx = build_context(cfg, d, preds, cache)
        A = answers_matrix(ctx)
        tr, te = np.where(ctx.train_mask)[0], np.where(ctx.test_mask)[0]
        qids = ctx.df["qid"].to_numpy()
        is_mc = np.array([ctx.tasks[i] == "mc" for i in te])
        # Multiple-choice training questions on which the greedy answers are
        # unanimous: singletons under the observed support, four-option
        # configurations under the task support.
        n_unanimous.append(int(sum(len({a for a in A[i] if a is not None}) == 1
                                   for i in tr if ctx.tasks[i] == "mc")))

        def record(rule, support, correct):
            acc[(rule, support)].append(float(np.mean(correct)))
            acc_mc[(rule, support)].append(float(np.mean(correct[is_mc])))
            for q, c in zip(qids[te], correct):
                perq[(rule, support)][int(q)].append(int(c))

        record("OW", "observed", run_ow_cand(ctx, A, tr, te)[0])
        record("OW", "task", run_ow(ctx, A, tr, te)[0])
        for support in SUPPORTS:
            fixed = support == "task"
            record("soft-OW", support,
                   run_soft_ow(ctx, A, tr, te, dist, n_samples, fixed_mc=fixed)[0])
            for rule, use_roll in (("jury", False), ("jury+rollouts", True)):
                cfg.raw["experiments"]["jury"].update(
                    use_rollouts=use_roll, lure="none", collapse_modes=False,
                    per_task=False, support=support)
                router = fit_router("jury", cfg, ctx, seed)
                record(rule, support, router.predict(ctx).correct)
                pi[(rule, support)].append([float(x) for x in router.pi])
        cfg.raw["experiments"]["jury"].update(support="observed")
    return perq, acc, acc_mc, pi, n_unanimous


def table(acc, acc_mc, boot):
    lines = ["\\begin{tabular}{llrrrrr}", "\\toprule",
             "& & \\multicolumn{2}{c}{All $1560$ questions} & "
             "& \\multicolumn{2}{c}{Multiple choice only} \\\\",
             "\\cmidrule(lr){3-4} \\cmidrule(lr){6-7}",
             "Rule & Labels & Observed & Four options & $\\Delta$~[95\\% CI] "
             "& Observed & Four options \\\\", "\\midrule"]
    for rule, (label, lab) in RULES.items():
        b = boot[rule]
        star = "$^{*}$" if b["p"] < 0.05 else ""
        lines.append(
            f"{label} & {lab} & "
            f"{fmt(np.mean(acc[(rule, 'observed')]), np.std(acc[(rule, 'observed')]))} & "
            f"{fmt(np.mean(acc[(rule, 'task')]), np.std(acc[(rule, 'task')]))} & "
            f"{100*b['delta']:+.1f}{star}~[{100*b['lo']:+.1f}, {100*b['hi']:+.1f}] & "
            f"{fmt(np.mean(acc_mc[(rule, 'observed')]))} & "
            f"{fmt(np.mean(acc_mc[(rule, 'task')]))} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    TABLEDIR.mkdir(parents=True, exist_ok=True)
    out = TABLEDIR / "support.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"[tables] -> {out}")


def main() -> None:
    cfg = load_config(ROOT / "config.yaml")
    if "--cache" in sys.argv:
        cfg.raw["paths"]["cache_dir"] = sys.argv[sys.argv.index("--cache") + 1]
    cache = Path(cfg["paths"]["cache_dir"])
    df = pd.read_csv(cache / "questions.csv").fillna("")
    preds = pd.read_csv(cache / "predictions.csv")

    perq, acc, acc_mc, pi, n_unanimous = measure(cfg, df, preds, cache)
    # Paired over questions, four-option support minus observed support.
    boot = {rule: bootstrap({"task": perq[(rule, "task")],
                             "observed": perq[(rule, "observed")]}, "task")["observed"]
            for rule in RULES}
    for b in boot.values():          # a rule whose verdicts never change has p = 1
        b["p"] = min(b["p"], 1.0)
    table(acc, acc_mc, boot)

    print(f"\n{'rule':<32}{'observed':>14}{'four options':>14}{'delta':>8}{'95% CI':>18}")
    for rule, (label, _) in RULES.items():
        a, b, d = 100 * np.asarray(acc[(rule, 'observed')]), 100 * np.asarray(acc[(rule, 'task')]), boot[rule]
        print(f"{label:<32}{a.mean():>9.1f} +- {a.std():<3.1f}{b.mean():>9.1f} +- {b.std():<3.1f}"
              f"{100*d['delta']:>+8.1f}  [{100*d['lo']:+.1f}, {100*d['hi']:+.1f}]  p={d['p']:.3f}")
    print(f"\nunanimous multiple-choice training questions per split: "
          f"{np.mean(n_unanimous):.0f} of {int(np.mean([len(np.where(_resplit(df.copy(), cfg['dataset']['train_frac'], s) == 'train')[0]) for s in SEEDS]))}"
          " (singletons under the observed support, in EM under the four-option one)")

    out = Path(cfg["paths"]["results_dir"]) / "support_control.json"
    out.write_text(json.dumps({
        "seeds": SEEDS, "supports": list(SUPPORTS),
        "accuracy": {f"{r}/{s}": v for (r, s), v in acc.items()},
        "accuracy_mc": {f"{r}/{s}": v for (r, s), v in acc_mc.items()},
        "delta_task_minus_observed": boot,
        "jury_pi": {f"{r}/{s}": v for (r, s), v in pi.items()},
        "unanimous_mc_train": n_unanimous,
    }, indent=2, default=float))
    print(f"[results] -> {out}")


if __name__ == "__main__":
    main()
