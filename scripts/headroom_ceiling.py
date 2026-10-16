"""How much of the pool's headroom is actually reachable?

The oracle answers 92.1% of the questions, every router in the benchmark answers
between 67% and 72%. This script asks where those 20 points go, and evaluates the
`aggregator` router against the two techniques that share its family -- the ones
that also run every expert and combine their answers, `ensemble` (fixed weights)
and `soft_actor` (learned weights).

    python paper/scripts/headroom_ceiling.py          # from the repo root

  1. Reachability. A weighted vote over discrete answers can only ever return an
     answer some expert already gave: however the weights move, they are
     quotiented onto the set of distinct answers. Fusing the experts' answer
     *distributions* can have a strictly larger reachable set -- an answer can win a
     convex combination without being any single expert's top choice. Section 1
     decides, per question and exactly, by linear programming, how much larger.
  2. The fuel. How often the gold answer is held by a minority of the pool, i.e.
     how often a plurality rule is wrong no matter how it is weighted.
  3. The post-hoc ceiling. `aggregator` against `ensemble` and `soft_actor`,
     broken down by stratum and reported in-sample as well as held-out. The
     in/out gap separates "the signal is not in the agreement pattern" from
     "180 training questions are not enough to generalise it".
  4. Selective override. Keep the consensus answer unless the aggregator has
     strong evidence the majority is wrong, and sweep that threshold.

Everything reuses the cached expert predictions and rollouts; no LLM is run.
Writes paper/figures/headroom_ceiling.png.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]          # this supplementary folder
sys.path.insert(0, str(ROOT))

from scipy.optimize import linprog                              # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier     # noqa: E402
from sklearn.linear_model import LogisticRegression             # noqa: E402
from sklearn.model_selection import KFold                       # noqa: E402

from router_lab.data import is_correct                          # noqa: E402
from router_lab.techniques.aggregator import (answer_key,       # noqa: E402
                                              rollout_distributions)
from router_lab.techniques.base import (build_context,          # noqa: E402
                                        knn_regional_accuracy,
                                        prediction_matrix)
from router_lab.techniques.multiseed import _resplit            # noqa: E402
from router_lab.techniques.runner import fit_router             # noqa: E402
from router_lab.utils import load_config                        # noqa: E402

SEEDS = list(range(1, 11))
SIBLINGS = ["ensemble", "soft_actor"]   # same family: run everyone, combine answers
TAUS = np.r_[np.arange(0.0, 0.90, 0.02), np.inf]
LP_TOL = 1e-9
FIGDIR = ROOT / "figures"

C_VOTE = "#4292c6"
C_AGG = "#e6550d"
C_SOFT = "#6a51a3"
C_CEIL = "#737373"


# --------------------------------------------------------------------------- #
# 1. Reachability: weighted vote vs. distribution fusion
# --------------------------------------------------------------------------- #
def _max_margin(P: np.ndarray, g: int, signed: bool = False) -> float:
    """max over w of  min_u  sum_e w_e (P[e,g] - P[e,u]).

    Positive iff some weight vector makes column `g` the strict argmax of the
    fused distribution, i.e. iff the gold answer is reachable at all. By default
    w ranges over the simplex (a linear opinion pool); with `signed` it ranges
    over the box [-1, 1]^E, the signed linear fusion the jury actually is. Either
    normalisation only fixes the scale, which the argmax does not see.
    """
    others = [u for u in range(P.shape[1]) if u != g]
    if not others:
        return 1.0
    n_exp = P.shape[0]
    D = P[:, [g]] - P[:, others]                       # [E, K-1]
    res = linprog(c=np.r_[np.zeros(n_exp), -1.0],
                  A_ub=np.hstack([-D.T, np.ones((len(others), 1))]),
                  b_ub=np.zeros(len(others)),
                  A_eq=None if signed else np.r_[np.ones(n_exp), 0.0][None, :],
                  b_eq=None if signed else [1.0],
                  bounds=[(-1 if signed else 0, 1)] * n_exp + [(-1, 1)],
                  method="highs")
    return float(-res.fun) if res.success else -1.0


def section1_reachability(df, G, dist, expert_ids) -> pd.DataFrame:
    rows = []
    for i, (qid, task, subject, gold) in enumerate(
            zip(df["qid"], df["task"], df["subject"], df["answer"])):
        keys = sorted({k for e in expert_ids for k in dist.get(qid, {}).get(e, {})})
        greedy = {G[i, j] for j in range(len(expert_ids)) if G[i, j] is not None}
        row = dict(qid=qid, task=task, subject=subject, n_keys=len(keys),
                   vote_greedy=int(any(is_correct(task, k, gold) for k in greedy)))
        if keys:
            P = np.array([[dist[qid].get(e, {}).get(k, 0.0) for k in keys]
                          for e in expert_ids])
            gold_cols = [j for j, k in enumerate(keys) if is_correct(task, k, gold)]
            modes = {keys[int(np.argmax(P[j]))] for j in range(len(expert_ids))
                     if P[j].sum() > 0}
            row["vote_modes"] = int(any(is_correct(task, k, gold) for k in modes))
            row["in_support"] = int(bool(gold_cols))
            margin = margin_signed = -1.0
            if gold_cols:                              # collapse gold-equivalent keys
                gj = gold_cols[0]
                if len(gold_cols) > 1:
                    P[:, gj] = P[:, gold_cols].sum(1)
                    keep = [j for j in range(P.shape[1]) if j not in gold_cols[1:]]
                    P, gj = P[:, keep], keep.index(gj)
                margin = _max_margin(P, gj)
                margin_signed = _max_margin(P, gj, signed=True)
            row["fusion"] = int(margin > LP_TOL)
            row["fusion_signed"] = int(margin_signed > LP_TOL)
            row["tie"] = int(abs(margin) <= LP_TOL and bool(gold_cols))
        rows.append(row)
    out = pd.DataFrame(rows)

    print("\n=== 1. what a weight vector can reach (all questions, exact by LP) ===")
    print(f"{'':38}{'all':>8}{'mc':>8}{'numeric':>9}")
    for col, label in (("vote_greedy", "vote over greedy answers"),
                       ("vote_modes", "vote over sampled modes"),
                       ("fusion", "distribution fusion, exists w >= 0"),
                       ("fusion_signed", "distribution fusion, signed w"),
                       ("in_support", "upper bound: gold in any sample")):
        if col not in out:
            continue
        print(f"{label:38}{100*out[col].mean():7.1f}%"
              f"{100*out.loc[out.task=='mc', col].mean():7.1f}%"
              f"{100*out.loc[out.task=='numeric', col].mean():8.1f}%")
    if "fusion" in out:
        d = 100 * (out.fusion.mean() - out.vote_modes.mean())
        print(f"  distribution fusion buys {d:+.1f} points of reachability over voting on\n"
              f"  the same samples; {int(((out.fusion==1)&(out.vote_modes==0)).sum())} "
              f"questions rescued, {int(out.tie.sum())} exact ties, "
              f"mean |candidates| {out.n_keys.mean():.1f}")
    return out


# --------------------------------------------------------------------------- #
# 2. The fuel: how often is the gold answer in the minority
# --------------------------------------------------------------------------- #
def stratum(task, gold, answers) -> str:
    """`plurality` (the majority already has it), `fuel` (reachable but in the
    minority, so no weighting of a plurality rule can be relied on) or
    `unreachable` (nobody has it)."""
    votes = defaultdict(int)
    for a in answers:
        if a is not None:
            votes[a] += 1
    if not votes or not any(is_correct(task, k, gold) for k in votes):
        return "unreachable"
    top = max(votes.values())
    winners = [k for k, v in votes.items() if v == top]
    return "plurality" if (len(winners) == 1 and is_correct(task, winners[0], gold)) \
        else "fuel"


def section2_fuel(df, G) -> pd.DataFrame:
    rows = []
    for i, (qid, task, gold) in enumerate(zip(df["qid"], df["task"], df["answer"])):
        answers = [G[i, j] for j in range(G.shape[1])]
        votes = defaultdict(int)
        for a in answers:
            if a is not None:
                votes[a] += 1
        rows.append(dict(qid=qid, task=task,
                         n_gold=sum(1 for a in answers
                                    if a is not None and is_correct(task, a, gold)),
                         plurality=max(votes.values()) if votes else 0,
                         stratum=stratum(task, gold, answers)))
    out = pd.DataFrame(rows)

    print("\n=== 2. the fuel: how often is the gold answer held by a minority ===")
    print(f"{'':22}{'plurality':>11}{'FUEL':>8}{'unreachable':>13}")
    for label, sub in (("all", out), ("mc", out[out.task == "mc"]),
                       ("numeric", out[out.task == "numeric"])):
        f = sub.stratum.value_counts(normalize=True) * 100
        print(f"{label:22}{f.get('plurality', 0):10.1f}%{f.get('fuel', 0):7.1f}%"
              f"{f.get('unreachable', 0):12.1f}%")
    print("\n  experts holding the gold answer:")
    h = out.n_gold.value_counts().sort_index()
    print("   " + "  ".join(f"{k}:{100*v/len(out):.1f}%" for k, v in h.items()))
    fuel = out[out.stratum == "fuel"]
    print(f"  on the {len(fuel)} FUEL questions the gold answer is held by "
          f"{fuel.n_gold.mean():.1f} experts on average (median "
          f"{int(fuel.n_gold.median())}), against a plurality of "
          f"{fuel.plurality.mean():.1f}")
    return out


# --------------------------------------------------------------------------- #
# 3-4. The aggregator against its family
# --------------------------------------------------------------------------- #
def consensus_answer(i, G, reg, n_exp):
    """The `ensemble` router's answer: a vote weighted by regional reliability."""
    votes = defaultdict(float)
    for j in range(n_exp):
        if G[i, j] is not None:
            votes[G[i, j]] += reg[i, j]
    return max(votes, key=votes.get) if votes else None


def sweep_tau(scores, y, groups, cands, row_of, G, reg, n_exp, taus, strata):
    """Accuracy / override rate / per-stratum accuracy for every threshold.

    tau = inf never overrides (pure consensus); tau = 0 always takes the
    aggregator's argmax.
    """
    uq = np.unique(groups)
    acc, ovr = np.zeros(len(taus)), np.zeros(len(taus))
    per = {s: [[] for _ in taus] for s in ("plurality", "fuel", "unreachable")}
    for qid in uq:
        m = groups == qid
        sc, yy, cc = scores[m], y[m], cands[m]
        base = consensus_answer(row_of[qid], G, reg, n_exp)
        hit = np.where(cc == base)[0]
        bi = int(hit[0]) if len(hit) else int(np.argmax(sc))
        ai = int(np.argmax(sc))
        margin = sc[ai] - sc[bi]
        for ti, tau in enumerate(taus):
            pick = ai if margin > tau else bi
            acc[ti] += yy[pick]
            ovr[ti] += int(pick != bi)
            per[strata[qid]][ti].append(yy[pick])
    n = len(uq)
    return acc / n, ovr / n, {k: np.array([np.mean(v) if v else np.nan for v in vv])
                              for k, vv in per.items()}


def sections34(cfg, df, preds, cache, have_roll, strata):
    variants = [("greedy answers only", False)] + ([("+ rollouts", True)] if have_roll else [])
    res = defaultdict(lambda: defaultdict(list))
    curves = defaultdict(list)

    for seed in SEEDS:
        d = df.copy()
        d["split"] = _resplit(d, cfg["dataset"]["train_frac"], seed)
        cfg.raw["seed"] = seed
        ctx = build_context(cfg, d, preds, cache)
        n_exp = len(ctx.expert_ids)
        G = np.vectorize(answer_key, otypes=[object])(
            np.repeat(ctx.tasks[:, None], n_exp, axis=1), prediction_matrix(ctx))
        reg = knn_regional_accuracy(ctx.Xtr, ctx.Ytr, ctx.emb,
                                    int(cfg["experiments"]["aggregator"].get("k", 15)))
        row_of = {q: i for i, q in enumerate(ctx.df["qid"].to_numpy())}
        tr_q = ctx.df.loc[ctx.train_mask, "qid"].to_numpy()
        te_q = ctx.df.loc[ctx.test_mask, "qid"].to_numpy()

        for name in SIBLINGS:
            r = fit_router(name, cfg, ctx, seed).predict(ctx)
            res[name]["acc"].append(float(r.correct.mean()))
            res[name]["cost"].append(float(r.cost.mean()))
            for s in ("plurality", "fuel"):
                sel = np.array([strata[q] == s for q in te_q])
                res[name][s].append(float(r.correct[sel].mean()))
        res["best_single"]["acc"].append(
            float(ctx.Yte[:, int(np.argmax(ctx.Ytr.mean(0)))].mean()))
        res["oracle"]["acc"].append(float(ctx.Yte.max(axis=1).mean()))

        for tag, use_roll in variants:
            cfg.raw["experiments"]["aggregator"]["use_rollouts"] = use_roll
            router = fit_router("aggregator", cfg, ctx, seed)
            out = router.predict(ctx)
            key = f"{tag}|aggregator"
            res[key]["acc"].append(float(out.correct.mean()))
            res[key]["cost"].append(float(out.cost.mean()))
            for s in ("plurality", "fuel"):
                sel = np.array([strata[q] == s for q in te_q])
                res[key][s].append(float(out.correct[sel].mean()))

            # the router's own feature matrices, reused for the diagnostics
            Xtr, ytr, gtr, ctr = router.candidate_rows(ctx, tr_q)
            Xte, yte, gte, cte = router.candidate_rows(ctx, te_q)
            sc_te = router.model.predict_proba(Xte)[:, 1]
            a_in, _, p_in = sweep_tau(router.model.predict_proba(Xtr)[:, 1], ytr, gtr,
                                      ctr, row_of, G, reg, n_exp, [0.0], strata)
            res[key]["acc_insample"].append(a_in[0])
            res[key]["fuel_insample"].append(p_in["fuel"][0])

            # an unregularised learner on the same features: how much of the
            # agreement pattern is memorisable but not generalisable
            gbm = HistGradientBoostingClassifier(max_iter=200, max_depth=4,
                                                 learning_rate=0.06,
                                                 random_state=seed).fit(Xtr, ytr)
            a_g, _, p_g = sweep_tau(gbm.predict_proba(Xte)[:, 1], yte, gte, cte,
                                    row_of, G, reg, n_exp, [0.0], strata)
            a_gi, _, p_gi = sweep_tau(gbm.predict_proba(Xtr)[:, 1], ytr, gtr, ctr,
                                      row_of, G, reg, n_exp, [0.0], strata)
            gk = f"{tag}|gbm"
            res[gk]["acc"].append(a_g[0])
            res[gk]["cost"].append(float(out.cost.mean()))
            res[gk]["plurality"].append(p_g["plurality"][0])
            res[gk]["fuel"].append(p_g["fuel"][0])
            res[gk]["acc_insample"].append(a_gi[0])
            res[gk]["fuel_insample"].append(p_gi["fuel"][0])

            # --- section 4: override threshold, chosen inside the train half ---
            oof = np.zeros(len(ytr))
            uq = np.unique(gtr)
            for a_idx, b_idx in KFold(5, shuffle=True, random_state=seed).split(uq):
                m_tr, m_va = np.isin(gtr, uq[a_idx]), np.isin(gtr, uq[b_idx])
                inner = LogisticRegression(max_iter=2000, C=router.C).fit(Xtr[m_tr],
                                                                         ytr[m_tr])
                oof[m_va] = inner.predict_proba(Xtr[m_va])[:, 1]
            a_tr, _, _ = sweep_tau(oof, ytr, gtr, ctr, row_of, G, reg, n_exp,
                                   TAUS, strata)
            tau_star = TAUS[int(np.argmax(a_tr))]

            a_te, o_te, p_te = sweep_tau(sc_te, yte, gte, cte, row_of, G, reg,
                                         n_exp, TAUS, strata)
            curves[tag].append((a_te, o_te, p_te["plurality"], p_te["fuel"]))
            a_h, o_h, _ = sweep_tau(sc_te, yte, gte, cte, row_of, G, reg, n_exp,
                                    [tau_star, np.inf], strata)
            ok = f"{tag}|override"
            res[ok]["acc"].append(a_h[0])
            res[ok]["cost"].append(float(out.cost.mean()))
            res[ok]["tau"].append(tau_star)
            res[ok]["rate"].append(o_h[0])
            res[ok]["base"].append(a_h[1])
    return res, curves, variants


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.std(), (d.mean() / se if se > 0 else np.nan), int((d > 0).sum())


def report34(res, curves, variants):
    print("\n=== 3. `aggregator` against the techniques that share its family ===")
    print("    (all of these run every expert and combine the answers; a selection")
    print("     router is not a comparable point and is left out on purpose)")
    print(f"\n{'system':<40}{'params':>10}{'accuracy':>16}{'plurality':>11}{'FUEL':>8}")
    print(f"{'best single expert':<40}{'':>10}{100*np.mean(res['best_single']['acc']):>11.1f}")
    rows = [(n, n) for n in SIBLINGS]
    for tag, _ in variants:
        rows.append((f"{tag}|aggregator", f"aggregator, {tag}"))
        rows.append((f"{tag}|gbm", f"  same features, GBM instead"))
    for key, label in rows:
        print(f"{label:<40}{np.mean(res[key]['cost']):>9.2f}B"
              f"{100*np.mean(res[key]['acc']):>11.1f} +- {100*np.std(res[key]['acc']):<4.1f}"
              f"{100*np.nanmean(res[key]['plurality']):>10.1f}%"
              f"{100*np.nanmean(res[key]['fuel']):>7.1f}%")
    print(f"{'oracle':<40}{'':>10}{100*np.mean(res['oracle']['acc']):>11.1f}")

    print("\n  paired against `ensemble`, ten splits:")
    for tag, _ in variants:
        for suffix, label in (("aggregator", "aggregator"), ("gbm", "GBM")):
            key = f"{tag}|{suffix}"
            m, s, t, w = paired(res[key]["acc"], res["ensemble"]["acc"])
            print(f"    {label + ', ' + tag:<34}{100*m:>+6.1f} +- {100*s:<4.1f}"
                  f"t={t:+5.2f}  {w}/10 splits")
    for name in SIBLINGS[1:]:
        m, s, t, w = paired(res[name]["acc"], res["ensemble"]["acc"])
        print(f"    {name:<34}{100*m:>+6.1f} +- {100*s:<4.1f}t={t:+5.2f}  {w}/10 splits")

    print("\n  in-sample vs held-out on the FUEL stratum -- a large gap means the"
          "\n  testimony carries signal this much training data cannot generalise:")
    for tag, _ in variants:
        for suffix, label in (("aggregator", "logreg"), ("gbm", "GBM")):
            key = f"{tag}|{suffix}"
            ins = 100 * np.nanmean(res[key]["fuel_insample"])
            out = 100 * np.nanmean(res[key]["fuel"])
            print(f"    {tag:<22}{label:<8}{ins:5.1f}% in-sample -> {out:5.1f}% "
                  f"held-out   ({ins - out:+.1f})")

    print("\n=== 4. selective override: keep the consensus unless the evidence is strong ===")
    for tag, _ in variants:
        A = np.array([c[0] for c in curves[tag]])
        O = np.array([c[1] for c in curves[tag]])
        PL = np.array([c[2] for c in curves[tag]])
        FU = np.array([c[3] for c in curves[tag]])
        print(f"\n  -- {tag} --")
        print(f"  {'tau':>6}{'override rate':>15}{'accuracy':>11}{'plurality':>11}{'FUEL':>8}")
        for ti, tau in enumerate(TAUS):
            if np.isfinite(tau) and (tau > 0.5 or ti % 3):
                continue
            label = "never" if not np.isfinite(tau) else f"{tau:.2f}"
            print(f"  {label:>6}{100*O[:, ti].mean():14.1f}%{100*A[:, ti].mean():10.1f}%"
                  f"{100*np.nanmean(PL[:, ti]):10.1f}%{100*np.nanmean(FU[:, ti]):7.1f}%")
        key = f"{tag}|override"
        acc = 100 * np.array(res[key]["acc"])
        base = 100 * np.array(res[key]["base"])
        m, s, t, w = paired(acc, base)
        print(f"  tau chosen by 5-fold CV inside the train half: "
              f"{np.mean(res[key]['tau']):.2f}")
        print(f"  held-out {acc.mean():.1f} +- {acc.std():.1f}%  vs consensus "
              f"{base.mean():.1f}%   delta {m:+.1f} +- {s:.1f}  t={t:+.2f}  "
              f"{w}/10 splits")
        if int(np.argmax(A.mean(axis=0))) == 0:
            print("  the curve is monotone in tau: no threshold beats always trusting"
                  "\n  the aggregator, so gating the override buys nothing here")


def figure(res, variants):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                                   # pragma: no cover
        print(f"[figures] skipped ({exc})")
        return
    pts = [("ensemble", np.mean(res["ensemble"]["cost"]),
            np.mean(res["ensemble"]["acc"]), C_VOTE),
           ("soft_actor", np.mean(res["soft_actor"]["cost"]),
            np.mean(res["soft_actor"]["acc"]), C_SOFT)]
    for tag, _ in variants:
        key = f"{tag}|aggregator"
        pts.append((f"aggregator\n{tag}", np.mean(res[key]["cost"]),
                    np.mean(res[key]["acc"]), C_AGG))

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for label, cost, acc, colour in pts:
        ax.scatter(cost, 100 * acc, s=70, color=colour, edgecolor="white", zorder=4)
        ax.annotate(label, (cost, 100 * acc), fontsize=8, xytext=(8, -3),
                    textcoords="offset points", color=colour)
    for y, label in ((100 * np.mean(res["oracle"]["acc"]), "oracle (gold reachable)"),
                     (100 * np.mean(res["best_single"]["acc"]), "best single expert")):
        ax.axhline(y, ls="--", lw=1.1, color=C_CEIL)
        ax.text(0.02, y + 0.8, label, transform=ax.get_yaxis_transform(),
                fontsize=8, color=C_CEIL)
    ax.set_xscale("log")
    ax.set_xlabel("Average parameters activated per query (B, log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Combination routers: what more compute buys\n"
                 "(mean over 10 splits)", fontsize=10)
    ax.grid(alpha=0.25, zorder=0)
    plt.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "headroom_ceiling.png", dpi=150)
    print(f"\n[figures] -> {FIGDIR / 'headroom_ceiling.png'}")


def main() -> None:
    cfg = load_config(ROOT / "config.yaml")
    cache = Path(cfg["paths"]["cache_dir"])
    df = pd.read_csv(cache / "questions.csv").fillna("")
    preds = pd.read_csv(cache / "predictions.csv")

    d = df.copy()
    d["split"] = _resplit(d, cfg["dataset"]["train_frac"], SEEDS[0])
    ctx0 = build_context(cfg, d, preds, cache)
    expert_ids = ctx0.expert_ids
    G = np.vectorize(answer_key, otypes=[object])(
        np.repeat(ctx0.tasks[:, None], len(expert_ids), axis=1),
        prediction_matrix(ctx0))
    dist, n_samples = rollout_distributions(cache, set(expert_ids))
    if not dist:
        print("[headroom] no fugu_rollouts.csv -- distribution variants skipped")

    print(f"[headroom] {len(df)} questions, {len(expert_ids)} experts, "
          f"{len(SEEDS)} splits, {n_samples} rollouts/expert")
    section1_reachability(df, G, dist, expert_ids)
    fuel = section2_fuel(df, G)
    strata = dict(zip(fuel.qid, fuel.stratum))
    res, curves, variants = sections34(cfg, df, preds, cache, bool(dist), strata)
    report34(res, curves, variants)
    figure(res, variants)


if __name__ == "__main__":
    main()
