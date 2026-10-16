"""Replication across ensembles: the jury on every sub-pool of the nine experts.

    python paper/scripts/pool_replication.py --cache results/cache_ext

The main comparison is one pool of nine models. What generalises is a question
about ensembles rather than about more questions inside the same ensemble, so
this script re-runs the comparison on every sub-pool of 3, 5 and 7 experts drawn
from the nine (84 + 126 + 36 = 246 pools), on the same ten splits, and reports
the distribution over pools of three paired differences:

    jury - equal-weight pool      what label-free weighting buys
    jury - supervised fusion      what labels would have bought
    jury - best expert            whether combining beats calling the best one

Every rule is the one of the main comparison, fitted as there, on the pool's
experts only: the jury (with repeated sampling) by EM on the training half, the
supervised linear fusion with weights fitted on the training labels, the
equal-weight pool with none, majority voting, and the best expert picked on the
training half. The jury is fitted twice: with the paper's evidence weight
alpha = n = 8, the multinomial likelihood of the eight samples, and with the
fractional alpha = 1 (one witness, one testimony), which the paper keeps as an
ablation. The reason is what the sub-pools reveal: on pools dominated by weak
experts, the alpha = 1 objective has its maximum at a degenerate fixed point
where every reliability falls below chance and the verdict votes against the
pool, while the multinomial objective does not. Nothing here runs an LLM; the
answers and the samples come from the cache.

The pools overlap, so their number overstates the independent evidence and the
spread is descriptive rather than an interval. A pool's number is its mean over
the ten splits, and the per-pool results are written to
results/pool_replication.json as they come, so an interrupted run resumes.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]          # this supplementary folder
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from run_paper_results import FIGDIR, SEEDS, TABLEDIR, write          # noqa: E402
from aggregator_vs_literature import (run_majority, run_soft_equal,   # noqa: E402
                                      run_soft_ow)
from router_lab.techniques.aggregator import (answer_key,             # noqa: E402
                                              rollout_distributions)
from router_lab.techniques.base import (build_context,                # noqa: E402
                                        prediction_matrix)
from router_lab.techniques.multiseed import _resplit                  # noqa: E402
from router_lab.techniques.runner import fit_router                   # noqa: E402
from router_lab.utils import load_config                              # noqa: E402

SIZES = (3, 5, 7, 9)
RULES = ("jury", "jury@a8", "majority", "soft-equal", "soft-OW", "best_single")
DIFFS = (("soft-equal", "equal-weight pool"),
         ("soft-OW", "supervised fusion"),
         ("best_single", "best expert"))
JURIES = (("jury@a8", "$\\alpha = 8$"), ("jury", "$\\alpha = 1$"))


def evaluate_pool(cfg, experts, df, preds, cache, roll_dist, n_samples):
    """-> {rule: [test accuracy on each of the ten splits]} for one pool."""
    cfg.raw["experts"] = [e for e in experts]
    acc = {r: [] for r in RULES}
    for seed in SEEDS:
        d = df.copy()
        d["split"] = _resplit(d, cfg["dataset"]["train_frac"], seed)
        cfg.raw["seed"] = seed
        ctx = build_context(cfg, d, preds, cache)
        A = np.vectorize(answer_key, otypes=[object])(
            np.repeat(ctx.tasks[:, None], len(ctx.expert_ids), axis=1),
            prediction_matrix(ctx))
        tr, te = np.where(ctx.train_mask)[0], np.where(ctx.test_mask)[0]
        acc["majority"].append(float(run_majority(ctx, A, tr, te)[0].mean()))
        acc["soft-equal"].append(float(run_soft_equal(ctx, A, tr, te, roll_dist, n_samples)[0].mean()))
        acc["soft-OW"].append(float(run_soft_ow(ctx, A, tr, te, roll_dist, n_samples)[0].mean()))
        cfg.raw["experiments"]["jury"].update(use_rollouts=True, lure="none")
        for name, alpha in (("jury", 1.0), ("jury@a8", 8.0)):
            cfg.raw["experiments"]["jury"]["alpha"] = alpha
            acc[name].append(float(fit_router("jury", cfg, ctx, seed).predict(ctx).correct.mean()))
        cfg.raw["experiments"]["jury"]["alpha"] = 1.0
        best_j = int(np.argmax(ctx.Ytr.mean(0)))          # picked on the training half
        acc["best_single"].append(float(ctx.Yte[:, best_j].mean()))
    return acc


def summarise(pools):
    """Per pool size: the distribution over pools of each rule and difference."""
    out = {}
    for k in SIZES:
        rows = [p for p in pools if p["k"] == k]
        if not rows:
            continue
        s = {"n_pools": len(rows)}
        for r in RULES:
            v = np.array([np.mean(p["acc"][r]) for p in rows])
            s[r] = dict(mean=float(v.mean()), sd=float(v.std()), min=float(v.min()), max=float(v.max()))
        for jury, _ in JURIES:
            for other, _ in DIFFS:
                v = np.array([np.mean(p["acc"][jury]) - np.mean(p["acc"][other]) for p in rows])
                s[jury + "-" + other] = dict(mean=float(v.mean()), sd=float(v.std()),
                                             min=float(v.min()), max=float(v.max()),
                                             median=float(np.median(v)),
                                             share_positive=float((v > 0).mean()),
                                             share_within_1=float((np.abs(v) <= 0.01).mean()))
            # the degenerate fit: the jury below majority voting by ten points or more
            v = np.array([np.mean(p["acc"][jury]) - np.mean(p["acc"]["majority"]) for p in rows])
            s[jury + "-degenerate"] = float((v < -0.10).mean())
        out[str(k)] = s
    return out


def table(summary):
    def cell(d):
        return f"{100*d['mean']:+.1f} $\\pm$ {100*d['sd']:.1f} & {100*d['share_positive']:.0f}"
    lines = ["\\begin{tabular}{lrrrr" + "rr" * len(DIFFS) + "}", "\\toprule",
             "Jury & Pool & Pools & Jury & Degen. & " + " & ".join(
                 f"\\multicolumn{{2}}{{c}}{{$-$ {label}}}" for _, label in DIFFS) + " \\\\",
             " & size & & & (\\%) & " + " & ".join("$\\Delta$ & $>0$ (\\%)" for _ in DIFFS) + " \\\\"]
    for jury, label in JURIES:
        lines.append("\\midrule")
        for n, (k, s) in enumerate(summary.items()):
            j = s[jury]
            cells = " & ".join(cell(s[jury + "-" + other]) for other, _ in DIFFS)
            lines.append(f"{label if n == 0 else ''} & {k} & {s['n_pools']} & "
                         f"{100*j['mean']:.1f} $\\pm$ {100*j['sd']:.1f} & "
                         f"{100*s[jury + '-degenerate']:.0f} & {cells} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(TABLEDIR / "pool_replication.tex", "\n".join(lines))


def figure(pools, juries=JURIES, name="pool_replication"):
    """One dot per pool: the three differences against pool size.

    The main figure shows the default jury alone, so that its spread is not
    compressed by the collapse of the alpha = 1 ablation; the second figure
    overlays both and is the one that shows the collapse.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colour = {"jury@a8": "#2a78d6", "jury": "#eb6834"}
    offset = {"jury@a8": -0.18, "jury": 0.18} if len(juries) > 1 else {"jury@a8": 0.0, "jury": 0.0}
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(DIFFS), figsize=(7.6, 3.0), sharey=True)
    for ax, (other, label) in zip(axes, DIFFS):
        ax.axhline(0, color="#737373", lw=0.8, zorder=1)
        for jury, jlabel in juries:
            for k in SIZES:
                rows = [q for q in pools if q["k"] == k]
                if not rows:
                    continue
                d = 100 * np.array([np.mean(q["acc"][jury]) - np.mean(q["acc"][other]) for q in rows])
                x = k + offset[jury] + rng.uniform(-0.09, 0.09, len(d))
                ax.scatter(x, d, s=14, color=colour[jury], edgecolor="white", linewidth=0.5,
                           alpha=0.85, zorder=3, label=f"jury, {jlabel}" if (k == SIZES[0] and ax is axes[0]) else None)
                ax.hlines(d.mean(), k + offset[jury] - 0.16, k + offset[jury] + 0.16,
                          color=colour[jury], lw=2.2, zorder=4)
        ax.set_title(f"Jury $-$ {label}", fontsize=9)
        ax.set_xticks(list(SIZES))
        ax.set_xlabel("experts in the pool", fontsize=8)
        ax.tick_params(labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6, zorder=0)
    axes[0].set_ylabel("difference (points)", fontsize=8)
    if len(juries) > 1:
        axes[0].legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / f"{name}.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"[figure] -> {out}")


def main() -> None:
    cfg = load_config(ROOT / "config.yaml")
    if "--cache" in sys.argv:
        cfg.raw["paths"]["cache_dir"] = sys.argv[sys.argv.index("--cache") + 1]
    cache = Path(cfg["paths"]["cache_dir"])
    df = pd.read_csv(cache / "questions.csv").fillna("")
    preds = pd.read_csv(cache / "predictions.csv")
    all_experts = list(cfg.raw["experts"])
    ids = [e["id"] for e in all_experts]
    roll_dist, n_samples = rollout_distributions(cache, set(ids))

    out = Path(cfg["paths"]["results_dir"]) / "pool_replication.json"
    pools = []
    if out.exists() and "--recompute" not in sys.argv:
        pools = json.loads(out.read_text())["pools"]
    done = {tuple(p["experts"]) for p in pools}

    todo = [c for k in SIZES for c in itertools.combinations(ids, k) if c not in done]
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv \
        else max(1, (os.cpu_count() or 2) - 2)
    print(f"[pools] {len(done)} cached, {len(todo)} to evaluate on {jobs} processes", flush=True)

    def _job(combo):
        experts = [e for e in all_experts if e["id"] in combo]
        return combo, evaluate_pool(cfg, experts, df, preds, cache, roll_dist, n_samples)

    global _WORK
    _WORK = _job
    with Pool(jobs) as pool:
        for n, (combo, acc) in enumerate(pool.imap_unordered(_run, todo), 1):
            pools.append({"experts": list(combo), "k": len(combo), "acc": acc})
            m = {r: 100 * np.mean(acc[r]) for r in RULES}
            print(f"[{n}/{len(todo)}] k={len(combo)} {'+'.join(combo)}: jury(a=8) {m['jury@a8']:.1f}  "
                  f"jury(a=1) {m['jury']:.1f}  majority {m['majority']:.1f}  "
                  f"equal {m['soft-equal']:.1f}  supervised {m['soft-OW']:.1f}  "
                  f"best {m['best_single']:.1f}", flush=True)
            out.write_text(json.dumps({"seeds": SEEDS, "sizes": SIZES, "experts": ids,
                                       "pools": pools, "summary": summarise(pools)}, indent=1))

    summary = summarise(pools)
    table(summary)
    figure(pools, juries=JURIES[:1])                       # the default jury alone
    figure(pools, juries=JURIES, name="pool_replication_alpha")   # with the ablation
    for jury, _ in JURIES:
        for k, s in summary.items():
            print(f"{jury} k={k} ({s['n_pools']} pools): {100*s[jury]['mean']:.1f} +- {100*s[jury]['sd']:.1f}  "
                  f"degenerate {100*s[jury+'-degenerate']:.0f}%   "
                  + "  ".join(f"vs {o}: {100*s[jury+'-'+o]['mean']:+.1f} +- {100*s[jury+'-'+o]['sd']:.1f} "
                              f"[{100*s[jury+'-'+o]['min']:+.1f}, {100*s[jury+'-'+o]['max']:+.1f}] "
                              f">0 in {100*s[jury+'-'+o]['share_positive']:.0f}%" for o, _ in DIFFS))
    print(f"[results] -> {out}")


_WORK = None


def _run(combo):
    """Module-level trampoline: forked workers inherit `_WORK` from `main`."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):      # silence the routers' logging
        return _WORK(combo)


if __name__ == "__main__":
    main()
