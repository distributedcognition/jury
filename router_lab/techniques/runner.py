"""Run one or several routing experiments and compare them against the baselines.

Baselines (router-independent) are computed once: each small LLM alone, random
routing, and the oracle upper bound. Then every selected experiment is trained on
the router-train split and evaluated on the held-out test split, reporting overall
accuracy, per-task accuracy and average compute (parameters used per question).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .base import build_context
from ..utils import set_seed
from . import REGISTRY


def fit_router(name, cfg, ctx, seed):
    """Construct, seed and fit one router.

    Re-seeding here, immediately before each fit, rather than once per split, is
    what makes a stochastic router's result a function of (split, router) alone.
    Seeding once per split instead leaves every router after the first starting
    from whatever RNG state its predecessors happened to leave behind, so its
    number depends on which other routers ran, and in what order: `gnn` moved by
    up to 2.9 points between orderings, which is larger than most of the
    differences this benchmark reports.
    """
    set_seed(seed)
    router = REGISTRY[name](cfg)
    router.fit(ctx)
    return router


def _acc(y):
    return float(np.mean(y)) if len(y) else 0.0


def _baselines(ctx):
    per_expert = {eid: _acc(ctx.Yte[:, j]) for j, eid in enumerate(ctx.expert_ids)}
    best_id = max(per_expert, key=per_expert.get)
    return {
        "experts": [{"id": eid, "params_b": float(ctx.params_b[j]), "accuracy": per_expert[eid]}
                    for j, eid in enumerate(ctx.expert_ids)],
        "best_single_expert": {"id": best_id, "accuracy": per_expert[best_id],
                               "params_b": float(ctx.params_b[ctx.expert_ids.index(best_id)])},
        "random_routing": _acc(ctx.Yte.mean(axis=1)),
        "oracle": _acc(ctx.Yte.max(axis=1)),
    }


def _summarize(res, ctx):
    tasks = ctx.tasks_te
    per_task = {}
    for t in sorted(set(tasks)):
        m = tasks == t
        per_task[t] = {"n": int(m.sum()), "accuracy": _acc(res.correct[m])}
    out = {
        "name": res.name,
        "kind": next(c.kind for c in REGISTRY.values() if c.name == res.name),
        "accuracy": _acc(res.correct),
        "avg_params_b": float(res.cost.mean()),
        "per_task": per_task,
        "note": res.note,
    }
    if res.calls is not None:
        out["avg_calls"] = float(np.mean(res.calls))
    return out


def run_experiments(cfg, df, preds, results_dir: Path, cache_dir: Path, names):
    ctx = build_context(cfg, df, preds, cache_dir)
    base = _baselines(ctx)

    if "all" in names:
        names = list(REGISTRY.keys())
    results = []
    for name in names:
        if name not in REGISTRY:
            print(f"[runner] unknown experiment '{name}', skipping")
            continue
        print(f"\n===== experiment: {name} =====")
        router = fit_router(name, cfg, ctx, cfg["seed"])
        res = router.predict(ctx)
        results.append(_summarize(res, ctx))

    task_names = sorted(set(ctx.tasks_te))
    _print_comparison(base, results, task_names, len(ctx.Yte))

    out = {"n_test": int(ctx.test_mask.sum()), "n_train": int(ctx.train_mask.sum()),
           "baselines": base, "experiments": results}
    with open(results_dir / "experiments.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[runner] results -> {results_dir / 'experiments.json'}")
    _plot(results_dir, base, results, task_names)
    return out


def _print_comparison(base, results, task_names, n_test):
    line = "=" * (46 + 12 * len(task_names))
    print(f"\n{line}\n  ROUTING EXPERIMENTS — COMPARISON (held-out test: {n_test} questions)\n{line}")
    hdr = f"  {'method':<20}{'overall':>9}{'params':>8}" + "".join(f"{t:>12}" for t in task_names)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    print("  BASELINES (small LLMs alone)")
    for e in base["experts"]:
        star = "  <-best" if e["id"] == base["best_single_expert"]["id"] else ""
        print(f"    {e['id']:<18}{e['accuracy']*100:>8.1f}%{e['params_b']:>7.1f}{'':>{12*len(task_names)}}{star}")
    print(f"    {'random routing':<18}{base['random_routing']*100:>8.1f}%{'-':>7}")
    print(f"    {'oracle (upper bnd)':<18}{base['oracle']*100:>8.1f}%{'-':>7}")
    print("  " + "-" * (len(hdr) - 2))
    print("  ROUTING EXPERIMENTS")
    best_single = base["best_single_expert"]["accuracy"]
    winner = max(results, key=lambda r: r["accuracy"]) if results else None
    for r in results:
        marks = "  ★" if winner and r["name"] == winner["name"] else ""
        tcols = "".join(f"{r['per_task'].get(t, {}).get('accuracy', 0)*100:>11.1f}%" for t in task_names)
        gain = (r["accuracy"] - best_single) * 100
        print(f"    {r['name']:<18}{r['accuracy']*100:>8.1f}%{r['avg_params_b']:>7.2f}{tcols}"
              f"   ({gain:+.1f} vs best){marks}")
    print(line)
    if winner:
        print(f"  best experiment: {winner['name']} "
              f"({winner['accuracy']*100:.1f}%, {(winner['accuracy']-best_single)*100:+.1f} pts vs best single expert)")


def _plot(results_dir, base, results, task_names):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[runner] skipping plot ({e})")
        return
    labels = [e["id"] for e in base["experts"]] + ["random"] + [r["name"] for r in results] + ["oracle"]
    values = ([e["accuracy"] * 100 for e in base["experts"]] + [base["random_routing"] * 100]
              + [r["accuracy"] * 100 for r in results] + [base["oracle"] * 100])
    # compute (parameters) used by each: single experts -> their size; routers -> avg
    # params/query; random & oracle have no meaningful compute figure.
    params = ([e["params_b"] for e in base["experts"]] + [None]
              + [r["avg_params_b"] for r in results] + [None])
    calls = ([None] * (len(base["experts"]) + 1)
             + [r.get("avg_calls") for r in results] + [None])
    fugu_names = {"fugu", "fugu_ultra"}
    router_colors = ["#756bb1" if r["name"] in fugu_names else "#e6550d"
                     for r in results]
    colors = (["#9ecae1"] * len(base["experts"]) + ["#bdbdbd"]
              + router_colors + ["#31a354"])
    fig, ax = plt.subplots(figsize=(11, max(8, 0.43 * len(labels))))
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, edgecolor="white")
    ax.axvline(base["best_single_expert"]["accuracy"] * 100,
               ls="--", lw=1.2, color="#3182bd")
    for b, v, p, c in zip(bars, values, params, calls):
        suffix = f"  {v:.1f}%" + (f"  |  {p:.2f}B" if p is not None else "")
        suffix += f"  |  {c:.2f} calls" if c is not None else ""
        ax.text(v + 0.6, b.get_y() + b.get_height() / 2, suffix,
                va="center", fontsize=8, fontweight="bold")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("All routing methods and model baselines\n"
                 "Labels show accuracy and average parameters per question")
    ax.set_xlim(0, min(110, base["oracle"] * 100 + 16))
    ax.grid(axis="x", alpha=0.18)
    plt.tight_layout()
    out = results_dir / "experiments.png"
    fig.savefig(out, dpi=160)
    print(f"[runner] comparison chart -> {out}")
