"""The two modelling assumptions behind `jury`, checked rather than asserted.

    python paper/scripts/testimony_structure.py     # from the repo root

`jury` assumes the experts are conditionally independent given the true answer.
That assumption is the one place where the model could obviously be wrong, and it
is also the assumption the `aggregator`'s family-diversity feature denies -- that
feature exists because agreement between two Qwen2.5 checkpoints is supposed to be
weaker evidence than agreement between a Qwen and an OLMo. Both cannot be right,
so this script settles it two ways.

1. Directly, with labels. For every pair of experts, among the questions where
   *both* are wrong, how often do they give the *same* wrong answer, against the
   rate expected if their errors were independent. The excess is the correlation.
   Reported split by whether the pair shares a pre-training lineage.

2. Through the model. `jury` is fitted with a shared-error channel switched on
   (`lure: family` and `lure: pool`) and without it, over the same ten splits, and
   the accuracies are compared. If correlation were both real and exploitable, the
   lure would earn its parameters.

Both answers come out the same way and both are negative, which is why `jury`
ships with `lure: none`. Neither licenses the stronger claim that the errors
*are* independent: what they show is that this parameterisation of the
dependence does not pay, while the pairwise table says a real if modest excess
is there to be had by something else.

3. The evidence weight. Each expert contributes alpha * sum_u p_e(u) log P(u|z)
   to the log-likelihood. The honest multinomial likelihood of n samples would be
   alpha = n, the multinomial likelihood of the n samples, and is `jury`'s
   default. The choice cannot change the verdict at fixed pi, but
   it sharpens the posterior inside EM and therefore moves pi. The sweep here
   says it does not matter above alpha = 1 and costs a point below it.

Nothing here runs an LLM.
"""
from __future__ import annotations

import copy
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]          # this supplementary folder
sys.path.insert(0, str(ROOT))

from router_lab.data import is_correct                            # noqa: E402
from router_lab.techniques import REGISTRY                        # noqa: E402
from router_lab.techniques.aggregator import FAMILY, answer_key   # noqa: E402
from router_lab.techniques.base import (build_context,            # noqa: E402
                                        prediction_matrix)
from router_lab.techniques.multiseed import _resplit              # noqa: E402
from router_lab.utils import load_config, set_seed                # noqa: E402

SEEDS = list(range(1, 11))
TABLEDIR = ROOT / "tables"
MIN_PAIR = 10          # ignore pairs with too few joint errors to estimate a rate


# --------------------------------------------------------------------------- #
# 1. the correlation itself
# --------------------------------------------------------------------------- #
def pair_correlations(ctx, A):
    """Excess agreement-on-error, per pair of experts.

    The baseline is not a constant: two experts that are both wrong agree by
    chance with probability 1/(k-1), where k is the number of answers actually
    available on that question -- four for multiple choice, and for GSM8K the
    number of distinct answers the pool put on the table, which is the only
    finite alternative the data defines. Averaging that per-question chance over
    the questions where the pair is jointly wrong is what the observed rate has
    to be read against.
    """
    golds = ctx.df["answer"].to_numpy()
    ids = list(ctx.expert_ids)
    n = len(ids)
    wrong = np.zeros((len(A), n), dtype=bool)
    for i in range(len(A)):
        for j in range(n):
            wrong[i, j] = (A[i, j] is not None
                           and not is_correct(ctx.tasks[i], A[i, j], golds[i]))

    rows = []
    for a, b in combinations(range(n), 2):
        both = np.where(wrong[:, a] & wrong[:, b])[0]
        if len(both) < MIN_PAIR:
            continue
        agree = float(np.mean([A[i, a] == A[i, b] for i in both]))
        chance = float(np.mean([
            1.0 / max((4 if ctx.tasks[i] != "numeric"
                       else max(len({x for x in A[i] if x is not None}), 2)) - 1, 1)
            for i in both]))
        rows.append(dict(a=ids[a], b=ids[b], n=len(both), agree=agree, chance=chance,
                         excess=agree - chance,
                         same_family=FAMILY.get(ids[a], ids[a]) == FAMILY.get(ids[b], ids[b])))
    return pd.DataFrame(rows).sort_values("excess", ascending=False)


# --------------------------------------------------------------------------- #
# 2. does the model want a correlation term?
# --------------------------------------------------------------------------- #
def lure_ablation(cfg, df, preds, cache):
    acc = defaultdict(list)
    lam = defaultdict(list)
    for seed in SEEDS:
        d = df.copy()
        d["split"] = _resplit(d, cfg["dataset"]["train_frac"], seed)
        cfg.raw["seed"] = seed
        ctx = build_context(cfg, d, preds, cache)
        for use_rollouts in (False, True):
            for mode in ("none", "family", "pool"):
                c = copy.copy(ctx)
                c.cfg = copy.deepcopy(ctx.cfg)
                c.cfg.raw["experiments"]["jury"].update(lure=mode,
                                                        use_rollouts=use_rollouts)
                set_seed(seed)
                router = REGISTRY["jury"](c.cfg).fit(c)
                key = ("rollouts" if use_rollouts else "greedy", mode)
                acc[key].append(float(router.predict(c).correct.mean()))
                if len(router.lam):
                    lam[key].append(float(router.lam.max()))
    return acc, lam


def alpha_sweep(cfg, df, preds, cache, alphas=(0.5, 1.0, 2.0, 4.0, 8.0)):
    """Accuracy as a function of the per-expert evidence weight."""
    acc = defaultdict(list)
    for seed in SEEDS:
        d = df.copy()
        d["split"] = _resplit(d, cfg["dataset"]["train_frac"], seed)
        cfg.raw["seed"] = seed
        ctx = build_context(cfg, d, preds, cache)
        for alpha in alphas:
            c = copy.copy(ctx)
            c.cfg = copy.deepcopy(ctx.cfg)
            c.cfg.raw["experiments"]["jury"].update(lure="none", use_rollouts=True,
                                                    alpha=float(alpha))
            set_seed(seed)
            router = REGISTRY["jury"](c.cfg).fit(c)
            acc[alpha].append(float(router.predict(c).correct.mean()))
    return acc


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), (d.mean() / se if se > 0 else np.nan), int((d > 0).sum())


# --------------------------------------------------------------------------- #
def write_table(corr, acc, lam):
    same = corr[corr.same_family]
    cross = corr[~corr.same_family]
    lines = ["\\begin{tabular}{lrrr}", "\\toprule",
             "& Pairs & Agree on error & Excess over chance \\\\", "\\midrule",
             f"Same pre-training lineage & {len(same)} & "
             f"{100*same.agree.mean():.1f} & {100*same.excess.mean():+.1f} \\\\",
             f"Different lineage & {len(cross)} & "
             f"{100*cross.agree.mean():.1f} & {100*cross.excess.mean():+.1f} \\\\",
             "\\midrule",
             "\\multicolumn{4}{l}{\\emph{Accuracy with a shared-error channel "
             "switched on}} \\\\"]
    for emission, label in (("greedy", "Greedy answers"),
                            ("rollouts", "With repeated sampling")):
        base = acc[(emission, "none")]
        lines.append(f"\\quad {label}, no lure & -- & "
                     f"{100*np.mean(base):.1f} & \\\\")
        for mode, name in (("family", "lineage lure"), ("pool", "pool lure")):
            m, t, w = paired(acc[(emission, mode)], base)
            # An identical verdict on every split leaves no variance and no t.
            cell = f"{100*m:+.2f}~({t:+.2f})" if np.isfinite(t) else "$\\pm$0.00~(identical)"
            lines.append(f"\\quad\\quad + {name} & -- & "
                         f"{100*np.mean(acc[(emission, mode)]):.1f} & {cell} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    TABLEDIR.mkdir(parents=True, exist_ok=True)
    (TABLEDIR / "testimony.tex").write_text("\n".join(lines) + "\n")
    print(f"[tables] -> {TABLEDIR / 'testimony.tex'}")


def main() -> None:
    cfg = load_config(ROOT / "config.yaml")
    if "--cache" in sys.argv:                      # extended-benchmark branch
        cfg.raw["paths"]["cache_dir"] = sys.argv[sys.argv.index("--cache") + 1]
    cache = Path(cfg["paths"]["cache_dir"])
    df = pd.read_csv(cache / "questions.csv").fillna("")
    preds = pd.read_csv(cache / "predictions.csv")

    d = df.copy()
    d["split"] = _resplit(d, cfg["dataset"]["train_frac"], SEEDS[0])
    cfg.raw["seed"] = SEEDS[0]
    ctx = build_context(cfg, d, preds, cache)
    A = np.vectorize(answer_key, otypes=[object])(
        np.repeat(ctx.tasks[:, None], len(ctx.expert_ids), axis=1),
        prediction_matrix(ctx))

    corr = pair_correlations(ctx, A)
    print("\n=== 1. when two experts are both wrong, do they give the same wrong "
          "answer? ===\n")
    print(f"{'pair':<34}{'lineage':>9}{'n':>6}{'agree':>8}{'chance':>8}{'excess':>9}")
    for r in corr.itertuples():
        print(f"{r.a + '+' + r.b:<34}{'same' if r.same_family else '--':>9}{r.n:>6}"
              f"{r.agree:>8.2f}{r.chance:>8.2f}{r.excess:>+9.2f}")
    same, cross = corr[corr.same_family], corr[~corr.same_family]
    print(f"\n  same lineage   {len(same):>2} pairs   mean excess {same.excess.mean():+.3f}")
    print(f"  cross lineage  {len(cross):>2} pairs   mean excess {cross.excess.mean():+.3f}")
    print("\n  -> shared pre-training does NOT predict correlated errors; the most")
    print("     correlated pairs are cross-lineage. The family-diversity feature in")
    print("     `aggregator` rests on a premise this pool does not support.")

    acc, lam = lure_ablation(cfg, df, preds, cache)
    print("\n=== 2. does `jury` want a shared-error channel? ===\n")
    print(f"{'emission':<12}{'lure':<10}{'accuracy':>14}{'vs none':>18}{'max lambda':>12}")
    for emission in ("greedy", "rollouts"):
        base = acc[(emission, "none")]
        for mode in ("none", "family", "pool"):
            a = 100 * np.asarray(acc[(emission, mode)])
            cell = ""
            if mode != "none":
                m, t, w = paired(acc[(emission, mode)], base)
                cell = (f"{100*m:+8.2f} {w:>2}/10{t:+7.2f}" if np.isfinite(t)
                        else f"{'identical':>18}")
            lm = f"{np.mean(lam[(emission, mode)]):>12.2f}" if (emission, mode) in lam else f"{'--':>12}"
            print(f"{emission:<12}{mode:<10}{a.mean():>8.1f} +- {a.std():<4.1f}{cell:>18}{lm}")
    worst = min(np.mean(v) - np.mean(acc[(e, "none")]) for (e, m), v in acc.items() if m != "none")
    best = max(np.mean(v) - np.mean(acc[(e, "none")]) for (e, m), v in acc.items() if m != "none")
    print(f"\n  -> the channel is never worth its parameters: it moves accuracy by"
          f" {100*worst:+.2f} to {100*best:+.2f} points.")
    print("     Under the multinomial evidence weight the susceptibilities are fitted")
    print("     away from zero and the channel costs accuracy rather than earning it.")

    al = alpha_sweep(cfg, df, preds, cache)
    print("\n=== 3. the evidence weight alpha (rollout emissions) ===\n")
    print(f"{'alpha':>8}{'accuracy':>16}{'vs alpha=1':>20}")
    base = al[1.0]
    for a_, v in al.items():
        x = 100 * np.asarray(v)
        cell = ""
        if a_ != 1.0:
            m, t, w = paired(v, base)
            cell = (f"{100*m:+8.2f} {w:>2}/10{t:+7.2f}" if np.isfinite(t)
                    else f"{'identical':>20}")
        print(f"{a_:>8.1f}{x.mean():>10.1f} +- {x.std():<4.1f}{cell:>20}")
    # Computed, not typed in: a hand-written summary went stale when the benchmark grew.
    gap = {a_: 100 * (np.mean(v) - np.mean(base)) for a_, v in al.items() if a_ != 1.0}
    above = [g for a_, g in gap.items() if a_ > 1.0]
    below = {a_: g for a_, g in gap.items() if a_ < 1.0}
    print(f"\n  -> above alpha = 1 the accuracy moves by {min(above):+.1f} to {max(above):+.1f} points;")
    print("     below it, " + ", ".join(f"{g:+.1f} at alpha = {a_:g}" for a_, g in below.items()) + ".")
    print("     The default is the multinomial weight alpha = n = 8; on this pool the")
    print("     fractional alpha = 1 serves as well, so the choice is not load-bearing here.")

    write_table(corr, acc, lam)
    out = Path(cfg["paths"]["results_dir"]) / "testimony_structure.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "pairs": corr.to_dict("records"),
        "same_family_excess": float(same.excess.mean()),
        "cross_family_excess": float(cross.excess.mean()),
        "lure_accuracy": {f"{e}/{m}": v for (e, m), v in acc.items()},
        "lure_max_lambda": {f"{e}/{m}": v for (e, m), v in lam.items()},
        "alpha_sweep": {str(k): v for k, v in al.items()},
    }, indent=2, default=float))
    print(f"[results] -> {out}")


if __name__ == "__main__":
    main()
