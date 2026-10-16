"""Multi-seed evaluation: average every experiment over several train/test splits.

The expensive expert inference is fixed (same cached correctness/confidence/embeddings,
indexed by qid), so we only resample the stratified train/test split and re-seed the
router training for each seed. This isolates the variance that actually matters for the
comparison — the split and the RL training stochasticity — and reports mean ± std,
turning the noisy single-run numbers into stable estimates with error bars.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import REGISTRY
from .base import build_context
from .runner import _acc, _baselines, _summarize, fit_router


def _resplit(df, train_frac, seed):
    """Stratified (per-subject) train/test split for the given seed."""
    rng = np.random.default_rng(seed)
    split = np.empty(len(df), dtype=object)
    for _, grp in df.groupby("subject"):
        ids = grp.index.to_numpy().copy()
        rng.shuffle(ids)
        cut = int(round(len(ids) * train_frac))
        split[ids[:cut]] = "train"
        split[ids[cut:]] = "test"
    return split


def run_multiseed(cfg, df, preds, results_dir: Path, cache_dir: Path, names=None, seeds=None):
    seeds = seeds or [1, 2, 3, 4, 5]
    if names is None or "all" in names:
        names = list(REGISTRY.keys())
    train_frac = cfg["dataset"]["train_frac"]

    per_exp = {n: {"acc": [], "params": [], "calls": []} for n in names}
    per_exp_task = {n: {} for n in names}
    base_acc = {"random": [], "oracle": [], "best_single": []}
    expert_acc = {e["id"]: [] for e in cfg["experts"]}

    for seed in seeds:
        print(f"\n########## SEED {seed} ##########")
        d = df.copy()
        d["split"] = _resplit(d, train_frac, seed)
        cfg.raw["seed"] = seed                      # re-seed router training
        ctx = build_context(cfg, d, preds, cache_dir)
        base = _baselines(ctx)
        for e in base["experts"]:
            expert_acc[e["id"]].append(e["accuracy"])
        base_acc["random"].append(base["random_routing"])
        base_acc["oracle"].append(base["oracle"])
        # Best expert *on this split* — the strongest (post-hoc) single-model baseline.
        base_acc["best_single"].append(base["best_single_expert"]["accuracy"])

        for name in names:
            # Seeded per (split, router): see fit_router. Without it the numbers
            # below depend on the order the routers happen to run in.
            router = fit_router(name, cfg, ctx, seed)
            res = router.predict(ctx)
            per_exp[name]["acc"].append(_acc(res.correct))
            per_exp[name]["params"].append(float(res.cost.mean()))
            if res.calls is not None:
                per_exp[name]["calls"].append(float(np.mean(res.calls)))
            s = _summarize(res, ctx)
            for t, v in s["per_task"].items():
                per_exp_task[name].setdefault(t, []).append(v["accuracy"])

    def ms(x):
        return {"mean": float(np.mean(x)), "std": float(np.std(x))}

    # Paired comparison: on every split the router and the best single expert are scored
    # on the SAME test questions, so the per-seed difference removes split difficulty as a
    # nuisance factor. That is what makes a few points on 5 splits interpretable at all.
    best = np.asarray(base_acc["best_single"])

    def vs_best(accs):
        d = np.asarray(accs) - best
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        return {"mean": float(d.mean()), "std": float(d.std()),
                "t": float(d.mean() / se) if se > 0 else float("inf") * np.sign(d.mean()),
                "wins": int((d > 0).sum()), "n": len(d)}

    summary = {
        "seeds": seeds,
        "experts": {k: ms(v) for k, v in expert_acc.items()},
        "random": ms(base_acc["random"]),
        "oracle": ms(base_acc["oracle"]),
        "best_single": ms(base_acc["best_single"]),
        "per_seed": {"best_single": base_acc["best_single"],
                     "oracle": base_acc["oracle"],
                     "experts": expert_acc,
                     "experiments": {n: per_exp[n]["acc"] for n in names}},
        "experiments": {},
    }
    for n in names:
        summary["experiments"][n] = {
            "accuracy": ms(per_exp[n]["acc"]),
            "avg_params_b": ms(per_exp[n]["params"]),
            "vs_best_single": vs_best(per_exp[n]["acc"]),
            "per_task": {t: ms(v) for t, v in per_exp_task[n].items()},
        }
        if per_exp[n]["calls"]:
            summary["experiments"][n]["avg_calls"] = ms(per_exp[n]["calls"])
    _print(summary, names)
    with open(results_dir / "multiseed.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[multiseed] data -> {results_dir / 'multiseed.json'}")
    _plot(results_dir, summary, names)
    return summary


def _print(summary, names):
    best_single = max(summary["experts"].items(), key=lambda kv: kv[1]["mean"])
    line = "=" * 66
    print(f"\n{line}\n  MULTI-SEED RESULTS  (mean ± std over {len(summary['seeds'])} splits)\n{line}")
    print(f"  {'method':<20}{'accuracy':>16}{'params(B)':>14}")
    print("  " + "-" * 62)
    print("  BASELINES")
    for eid, m in summary["experts"].items():
        star = "  <-best" if eid == best_single[0] else ""
        print(f"    {eid:<18}{m['mean']*100:>7.1f} ± {m['std']*100:>3.1f}%{'':>14}{star}")
    print(f"    {'random':<18}{summary['random']['mean']*100:>7.1f} ± {summary['random']['std']*100:>3.1f}%")
    print(f"    {'oracle':<18}{summary['oracle']['mean']*100:>7.1f} ± {summary['oracle']['std']*100:>3.1f}%")
    print("  " + "-" * 62)
    print("  ROUTING EXPERIMENTS       (Δ = paired, per split, vs the best expert there)")
    ranked = sorted(names, key=lambda n: summary["experiments"][n]["accuracy"]["mean"], reverse=True)
    for n in ranked:
        e = summary["experiments"][n]
        a, p, v = e["accuracy"], e["avg_params_b"], e.get("vs_best_single")
        mark = "  ★" if n == ranked[0] else ""
        delta = (f"Δ{v['mean']*100:+.1f} ± {v['std']*100:.1f}  t={v['t']:+.1f}  "
                 f"{v['wins']}/{v['n']} wins" if v else "")
        calls = f"  calls={e['avg_calls']['mean']:.2f}" if "avg_calls" in e else ""
        print(f"    {n:<18}{a['mean']*100:>7.1f} ± {a['std']*100:>3.1f}%"
              f"{p['mean']:>9.2f} ± {p['std']:.2f}   {delta}{calls}{mark}")
    print(line)
    n = len(summary["seeds"])
    print(f"  |t| > {_t_crit(n - 1):.2f} is significant at p<0.05 "
          f"({n} splits, paired, two-sided).")


# Two-sided t critical values at p<0.05, indexed by degrees of freedom (scipy-free).
_T05 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31,
        9: 2.26, 10: 2.23, 11: 2.20, 12: 2.18, 14: 2.14, 19: 2.09, 29: 2.05}


def _t_crit(dof: int) -> float:
    if dof < 1:
        return float("inf")
    return _T05.get(dof, min((v for d, v in _T05.items() if d >= dof), default=1.96))


def _plot(results_dir, summary, names):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[multiseed] skipping plot ({e})")
        return
    labels = list(summary["experts"]) + ["random"] + names + ["oracle"]
    means = ([summary["experts"][e]["mean"] * 100 for e in summary["experts"]]
             + [summary["random"]["mean"] * 100]
             + [summary["experiments"][n]["accuracy"]["mean"] * 100 for n in names]
             + [summary["oracle"]["mean"] * 100])
    errs = ([summary["experts"][e]["std"] * 100 for e in summary["experts"]]
            + [summary["random"]["std"] * 100]
            + [summary["experiments"][n]["accuracy"]["std"] * 100 for n in names]
            + [summary["oracle"]["std"] * 100])
    params = ([None] * len(summary["experts"]) + [None]
              + [summary["experiments"][n]["avg_params_b"]["mean"] for n in names] + [None])
    calls = ([None] * (len(summary["experts"]) + 1)
             + [summary["experiments"][n].get("avg_calls", {}).get("mean") for n in names]
             + [None])
    ne = len(summary["experts"])
    fugu_names = {"fugu", "fugu_ultra"}
    router_colors = ["#756bb1" if n in fugu_names else "#e6550d" for n in names]
    colors = ["#9ecae1"] * ne + ["#bdbdbd"] + router_colors + ["#31a354"]

    fig, ax = plt.subplots(figsize=(11, max(8, 0.43 * len(labels))))
    y = np.arange(len(labels))
    bars = ax.barh(y, means, xerr=errs, color=colors, edgecolor="white",
                   capsize=3, error_kw={"elinewidth": 1, "ecolor": "#444"})
    best = max(summary["experts"].values(), key=lambda m: m["mean"])["mean"] * 100
    ax.axvline(best, ls="--", lw=1.2, color="#3182bd")
    for b, v, e, p, c in zip(bars, means, errs, params, calls):
        suffix = f"  {v:.1f} ± {e:.1f}%" + (f"  |  {p:.2f}B" if p is not None else "")
        suffix += f"  |  {c:.2f} calls" if c is not None else ""
        ax.text(v + e + 0.6, b.get_y() + b.get_height() / 2, suffix,
                va="center", fontsize=8, fontweight="bold")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy (%)")
    ax.set_title(f"All routing methods and model baselines\n"
                 f"Mean ± std over {len(summary['seeds'])} train/test splits")
    ax.set_xlim(0, min(112, summary["oracle"]["mean"] * 100 + 18))
    ax.grid(axis="x", alpha=0.18)
    plt.tight_layout()
    out = results_dir / "multiseed.png"
    fig.savefig(out, dpi=160)
    print(f"[multiseed] chart -> {out}")
