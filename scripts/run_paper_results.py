"""Regenerate every number, table and figure in `paper/jury.tex`.

    python paper/scripts/run_paper_results.py        # from the repo root

One entry point so the paper is never hand-edited: it writes
`results/paper_results.json` with every per-split measurement, LaTeX fragments
into `paper/tables/*.tex` that `jury.tex` \\input{}s directly, and the figure
`paper/figures/jury_frontier.png`. Nothing here runs an LLM -- the expert answers
and the repeated samples come from the committed cache.

The measurement code is not duplicated: the reachability LP and the stratum
definition come from `headroom_ceiling.py`, the literature baselines from
`aggregator_vs_literature.py`, and the routers from `router_lab` itself.
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

from aggregator_vs_literature import (run_isp, run_majority,      # noqa: E402
                                      run_more, run_ow, run_ow_cand,
                                      run_smoothie_global,
                                      run_smoothie_local,
                                      run_soft_equal, run_soft_ow,
                                      run_soft_ow_dist, run_soft_ow_task, run_topla)
from headroom_ceiling import (section1_reachability, section2_fuel,  # noqa: E402
                              sweep_tau, TAUS)

from router_lab.data import is_correct                            # noqa: E402
from router_lab.techniques.aggregator import (answer_key,         # noqa: E402
                                              collapse_to_modes,
                                              rollout_distributions)
from router_lab.techniques.base import (build_context,            # noqa: E402
                                        knn_regional_accuracy,
                                        prediction_matrix)
from router_lab.techniques.multiseed import _resplit              # noqa: E402
from router_lab.techniques.runner import fit_router               # noqa: E402
from router_lab.utils import load_config                          # noqa: E402

SEEDS = list(range(1, 11))
# The six MMLU subjects collapse to one source; ARC, HellaSwag and GSM8K are their
# own. Ordered as the breakdown table prints them.
SOURCE = {s: "MMLU" for s in ("world_religions", "professional_law",
                              "high_school_biology", "prehistory", "marketing",
                              "high_school_geography")}
SOURCE.update({"gsm8k": "GSM8K", "arc": "ARC-C", "hellaswag": "HellaSwag"})
SOURCE_ORDER = ["MMLU", "GSM8K", "ARC-C", "HellaSwag"]
TABLEDIR = ROOT / "tables"
FIGDIR = ROOT / "figures"

# name -> (pretty label, whether the method needs gold labels, citation key)
LITERATURE = {
    "majority": ("Majority vote", "no", "li2024moreagents"),
    # Both of these are from one paper, which is why they share a key. Each also
    # carries the mechanism it inverts or generalises, on the same policy as the
    # opinion pools below: the family first, then the form we actually ran.
    "ISP": ("Inverse Surprising Popularity", "no",
            "prelec2017surprising,beyondmajority2025"),
    "smoothie_global": ("Smoothie-Global", "no", "guha2024smoothie"),
    "smoothie_local": ("Smoothie-Local", "no", "guha2024smoothie"),
    "OW": ("Optimal Weight", "yes", "nitzan1982optimal,beyondmajority2025"),
    "TOPLA": ("LLM-TOPLA", "yes", "tekin2024topla"),
    "MoRE": ("MoRE", "yes", "si2023more"),
    "MoRE+rollouts": ("MoRE + repeated sampling", "yes", "si2023more"),
    "soft-equal": ("Equal-weight pool", "no", "stone1961opinion"),
    # Ours only as a pairing: the pool is Stone's and the weights are OW's, so
    # both are cited and neither is credited with the combination.
    # Equal weights really is Stone's pool; log-odds weights go negative on this
    # pool, so that row is a signed linear fusion and is named as one.
    "soft-OW": ("Supervised linear fusion", "yes",
                "stone1961opinion,beyondmajority2025"),
}
BOOT = 20000                 # paired bootstrap resamples, over questions
# name -> (pretty label, whether it needs gold labels)
OURS = {
    "aggregator": ("Answer scoring, discriminative", "yes"),
    "aggregator+rollouts": ("\\quad + repeated sampling", "yes"),
    "jury": ("\\textbf{Jury (ours), greedy answers}", "\\textbf{no}"),
    "jury+rollouts": ("\\textbf{\\quad + repeated sampling}", "\\textbf{no}"),
}
REF = "jury+rollouts"        # the row every Delta is measured against

# Six methods share x = 14.4B and three share x = 48.2B, so the frontier labels
# have to be placed by hand or they overprint each other.
LABEL_POS = {
    "majority": (6, -3, "left"), "ISP": (6, -3, "left"),
    "OW": (-9, -3, "right"), "jury": (6, -3, "left"),
    "aggregator": (-9, -3, "right"), "MoRE": (6, 2, "left"),
    "soft-equal": (-9, -3, "right"), "MoRE+rollouts": (-9, -3, "right"),
    "aggregator+rollouts": (-9, 2, "right"), "soft-OW": (6, -3, "left"),
    "jury+rollouts": (6, -3, "left"),
    "smoothie_global": (6, -3, "left"), "smoothie_local": (6, -3, "left"),
    "TOPLA": (6, -3, "left"),
}
JURY_TAUS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]


def fmt(mean, std=None):
    return f"{100*mean:.1f}" + (f" $\\pm$ {100*std:.1f}" if std is not None else "")


def bootstrap(perq, ref, n=BOOT, seed=0):
    """Paired bootstrap over questions: ref minus each method.

    A question's score for a method is its mean correctness over the splits in
    which it was held out, so every question contributes once and the resampling
    unit is the question rather than the (overlapping) split.
    """
    qs = sorted(perq[ref])
    M = {k: np.array([np.mean(perq[k][q]) for q in qs]) for k in perq}
    idx = np.random.default_rng(seed).integers(0, len(qs), size=(n, len(qs)))
    out = {}
    for k in perq:
        if k == ref:
            continue
        d = M[ref] - M[k]
        bs = d[idx].mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        p = 2 * min(float((bs <= 0).mean()), float((bs >= 0).mean()))
        out[k] = dict(delta=float(d.mean()), lo=float(lo), hi=float(hi),
                      p=max(p, 1.0 / n))
    return out


def equivalence(perq, a, b, n=BOOT, seed=0):
    """Two one-sided tests for a - b, by the same paired bootstrap over questions.

    A two-sided interval that contains zero says only that no difference was
    detected; it is not evidence that two methods are the same. The statements
    that *are* available from these resamples are the two one-sided ones, and
    they are reported as the smallest margin each supports at the 5% level:

      `noninferiority`  a is not worse than b by more than this many points
                        (the 90% interval's lower end, one-sided 95%);
      `equivalence`     the smallest delta for which the 90% interval fits inside
                        (-delta, +delta), which is the TOST margin the data
                        establish. A small non-inferiority margin with a large
                        equivalence margin is the normal outcome when the point
                        estimate favours `a`: it is licence to say b does not
                        win, not licence to say the two are interchangeable.
    """
    qs = sorted(perq[a])
    A = np.array([np.mean(perq[a][q]) for q in qs])
    B = np.array([np.mean(perq[b][q]) for q in qs])
    d = A - B
    idx = np.random.default_rng(seed).integers(0, len(qs), size=(n, len(qs)))
    bs = d[idx].mean(1)
    lo90, hi90 = np.percentile(bs, [5, 95])
    lo95, hi95 = np.percentile(bs, [2.5, 97.5])
    return dict(delta=float(d.mean()),
                lo95=float(lo95), hi95=float(hi95),
                lo90=float(lo90), hi90=float(hi90),
                noninferiority=float(max(0.0, -lo90)),
                equivalence=float(max(abs(lo90), abs(hi90))))


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.std(), (d.mean() / se if se > 0 else np.nan), int((d > 0).sum())


def write(path: Path, body: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body.rstrip() + "\n")
    print(f"[tables] -> {path}")


# --------------------------------------------------------------------------- #
# Table 1: the experts alone, and what the pool allows
# --------------------------------------------------------------------------- #
def table_experts(cfg, ctx, df):
    """Each expert alone, split by source dataset.

    The columns are per source and not per `task`: ARC and HellaSwag are multiple
    choice like MMLU, so a column keyed on the task type would pool all three under
    an "MMLU" heading and misreport every expert's MMLU accuracy by several points.
    """
    ids = list(ctx.expert_ids)
    src = np.array([SOURCE.get(s, s) for s in ctx.subjects])
    masks = [(name, src == name) for name in SOURCE_ORDER]

    rows = []
    for j, e in enumerate(ids):
        rows.append((e, float(cfg["experts"][j]["params_b"]),
                     float(ctx.Y[:, j].mean()),
                     [float(ctx.Y[m, j].mean()) for _, m in masks]))
    best_all = max(rows, key=lambda r: r[2])
    # The task-only routing baseline now has four sources to choose between.
    per_source = sum(ctx.Y[m, int(np.argmax(ctx.Y[m].mean(0)))].sum()
                     for _, m in masks) / len(ctx.Y)

    head = " & ".join(SOURCE_ORDER)
    lines = ["\\begin{tabular}{lrr" + "r" * len(masks) + "}", "\\toprule",
             f"Expert & Params (B) & All & {head} \\\\", "\\midrule"]
    for e, p, a, per in sorted(rows, key=lambda r: -r[2]):
        name = e.replace("_", "\\_")
        star, end = ("\\textbf{", "}") if e == best_all[0] else ("", "")
        cells = " & ".join(f"{100*x:.1f}" for x in per)
        lines.append(f"\\texttt{{{name}}} & {p:.2f} & {star}{100*a:.1f}{end} "
                     f"& {cells} \\\\")
    lines += ["\\midrule",
              f"Random expert & {np.mean([r[1] for r in rows]):.2f} & "
              f"{100*ctx.Y.mean():.1f} & "
              + " & ".join(f"{100*ctx.Y[m].mean():.1f}" for _, m in masks) + " \\\\",
              f"Best expert per source & -- & {100*per_source:.1f}"
              + " & --" * len(masks) + " \\\\",
              f"Oracle (best per question) & -- & {100*ctx.Y.max(axis=1).mean():.1f} & "
              + " & ".join(f"{100*ctx.Y[m].max(axis=1).mean():.1f}" for _, m in masks)
              + " \\\\",
              "\\bottomrule", "\\end{tabular}"]
    write(TABLEDIR / "experts.tex", "\n".join(lines))
    return dict(per_task_ceiling=per_source, oracle=float(ctx.Y.max(axis=1).mean()),
                best_single=best_all[2], best_single_id=best_all[0],
                random=float(ctx.Y.mean()),
                experts={r[0]: dict(params_b=r[1], all=r[2],
                                    **{k: v for (k, _), v in zip(masks, r[3])})
                         for r in rows})


# --------------------------------------------------------------------------- #
# Table 2: what a weight vector can reach
# --------------------------------------------------------------------------- #
def table_reachability(reach):
    def pct(col, mask=None):
        s = reach[col] if mask is None else reach.loc[mask, col]
        return f"{100*s.mean():.1f}"
    src = reach["subject"].map(lambda s: SOURCE.get(s, s))
    masks = [(name, src == name) for name in SOURCE_ORDER]
    head = " & ".join(SOURCE_ORDER)
    lines = ["\\begin{tabular}{lr" + "r" * len(masks) + "}", "\\toprule",
             f"Reachable set & All & {head} \\\\", "\\midrule"]
    for col, label in (("vote_greedy", "Weighted vote, greedy answers"),
                       ("vote_modes", "Weighted vote, sampled modes"),
                       ("fusion", "Distribution fusion, $\\exists\\,w \\ge 0$"),
                       ("fusion_signed", "Distribution fusion, $\\exists\\,w$ signed"),
                       ("in_support", "Upper bound: gold sampled at all")):
        if col in reach:
            cells = " & ".join(pct(col, m) for _, m in masks)
            lines.append(f"{label} & {pct(col)} & {cells} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(TABLEDIR / "reachability.tex", "\n".join(lines))


# --------------------------------------------------------------------------- #
# Table 3 + 4: the comparison
# --------------------------------------------------------------------------- #
def measure(cfg, df, preds, cache, strata):
    acc, cost = defaultdict(list), defaultdict(list)
    by_source = defaultdict(lambda: defaultdict(list))
    stratum_acc = defaultdict(lambda: defaultdict(list))
    extra = defaultdict(list)
    curves = defaultdict(list)
    perq = defaultdict(lambda: defaultdict(list))
    roll_dist, n_samples = rollout_distributions(cache, {e["id"] for e in cfg["experts"]})
    mode_dist = collapse_to_modes(roll_dist)

    for seed in SEEDS:
        d = df.copy()
        d["split"] = _resplit(d, cfg["dataset"]["train_frac"], seed)
        cfg.raw["seed"] = seed
        ctx = build_context(cfg, d, preds, cache)
        n_exp = len(ctx.expert_ids)
        A = np.vectorize(answer_key, otypes=[object])(
            np.repeat(ctx.tasks[:, None], n_exp, axis=1), prediction_matrix(ctx))
        tr, te = np.where(ctx.train_mask)[0], np.where(ctx.test_mask)[0]
        te_q = ctx.df["qid"].to_numpy()[te]
        reg = knn_regional_accuracy(ctx.Xtr, ctx.Ytr, ctx.emb,
                                    int(cfg["experiments"]["aggregator"].get("k", 15)))
        row_of = {q: i for i, q in enumerate(ctx.df["qid"].to_numpy())}

        src_te = np.array([SOURCE.get(s, s) for s in ctx.subjects[te]])

        def record(name, correct, costs):
            acc[name].append(float(np.mean(correct)))
            cost[name].append(float(np.mean(costs)))
            # Per source, because three quarters of the benchmark is now ARC and
            # HellaSwag and a single overall number would hide which of them the
            # result rests on.
            for s in np.unique(src_te):
                m = src_te == s
                by_source[name][s].append(float(np.mean(np.asarray(correct)[m])))
            for s in ("plurality", "fuel"):
                sel = np.array([strata[q] == s for q in te_q])
                stratum_acc[name][s].append(float(np.mean(np.asarray(correct)[sel])))
            # Per question, for the paired bootstrap. The ten splits share the
            # same 360 questions and overlap by construction, so a paired t-test
            # over splits treats ten dependent numbers as ten observations and
            # overstates the evidence; resampling questions does not.
            for q, y in zip(te_q, np.asarray(correct)):
                perq[name][q].append(int(y))

        for name, fn in (("majority", run_majority), ("OW", run_ow),
                         ("OW@cand", run_ow_cand), ("ISP", run_isp),
                         ("smoothie_global", run_smoothie_global),
                         ("smoothie_local", run_smoothie_local), ("MoRE", run_more)):
            record(name, *fn(ctx, A, tr, te))
        record("MoRE+rollouts", *run_more(ctx, A, tr, te, roll_dist, n_samples))
        for name, fn in (("soft-equal", run_soft_equal), ("soft-OW", run_soft_ow)):
            record(name, *fn(ctx, A, tr, te, roll_dist, n_samples))
        # Sample-matched controls: the same eight samples per expert, collapsed to
        # the mode before the rule sees them. Everything these pay for the
        # distributional rules also pay for, so the difference between a row and
        # its collapsed twin is the distribution and nothing else.
        record("majority@modes", *run_soft_equal(ctx, A, tr, te, mode_dist, n_samples))
        record("OW@modes", *run_soft_ow(ctx, A, tr, te, mode_dist, n_samples))
        # Specialisation: the same supervised pool with one weight vector per task.
        record("soft-OW@task", *run_soft_ow_task(ctx, A, tr, te, roll_dist, n_samples))
        # The distributional supervised plug-in: the jury's M-step with the gold
        # answer in place of the posterior.
        record("soft-OW@dist", *run_soft_ow_dist(ctx, A, tr, te, roll_dist, n_samples))
        c, k, sub, _ = run_topla(ctx, A, tr, te, seed)
        record("TOPLA", c, k)
        extra["topla_subset"].append(tuple(ctx.expert_ids[j] for j in sub))

        for tag, use_roll in (("", False), ("+rollouts", True)):
            cfg.raw["experiments"]["aggregator"]["use_rollouts"] = use_roll
            router = fit_router("aggregator", cfg, ctx, seed)
            out = router.predict(ctx)
            record("aggregator" + tag, out.correct, out.cost)

            Xtr, ytr, gtr, ctr = router.candidate_rows(ctx, ctx.df["qid"].to_numpy()[tr])
            Xte, yte, gte, cte = router.candidate_rows(ctx, te_q)
            _, _, p_in = sweep_tau(router.model.predict_proba(Xtr)[:, 1], ytr, gtr, ctr,
                                   row_of, A, np.ones_like(reg), n_exp, [0.0], strata)
            extra["fuel_insample" + tag].append(p_in["fuel"][0])
            a_te, o_te, p_te = sweep_tau(router.model.predict_proba(Xte)[:, 1], yte, gte,
                                         cte, row_of, A, np.ones_like(reg),
                                         n_exp, TAUS, strata)
            curves[tag or "greedy"].append((a_te, o_te, p_te["plurality"], p_te["fuel"]))

        for tag, use_roll in (("", False), ("+rollouts", True)):
            cfg.raw["experiments"]["jury"]["use_rollouts"] = use_roll
            cfg.raw["experiments"]["jury"]["lure"] = "none"
            router = fit_router("jury", cfg, ctx, seed)
            out = router.predict(ctx)
            record("jury" + tag, out.correct, out.cost)
            extra["jury_pi" + tag].append([float(x) for x in router.pi])

            # The selective override, on a calibrated posterior rather than on an
            # uncalibrated scorer margin: a different question from the one
            # `sweep_tau` asks of `aggregator`, and worth asking separately.
            golds = ctx.df.loc[ctx.test_mask, "answer"].to_numpy()
            tasks_te = ctx.tasks[ctx.test_mask]
            # Calibration of the posterior. We call it a probability, so it has
            # to be checked rather than asserted: ECE on the top-1 confidence,
            # plus the Brier score and NLL of the full posterior against the gold
            # answer (skipping the questions where gold is not on the table, for
            # which no selection rule has a defined loss).
            conf, hit, brier, nll = [], [], [], []
            for n, qid in enumerate(te_q):
                out2 = router.deliberate(qid)
                if out2 is None:
                    continue
                cands, gamma, _ = out2
                k = int(np.argmax(gamma))
                conf.append(float(gamma[k]))
                hit.append(is_correct(tasks_te[n], cands[k], golds[n]))
                gold_ix = [j for j, c in enumerate(cands)
                           if is_correct(tasks_te[n], c, golds[n])]
                if gold_ix:
                    tgt = np.zeros(len(cands)); tgt[gold_ix[0]] = 1.0
                    brier.append(float(((gamma - tgt) ** 2).sum()))
                    nll.append(float(-np.log(max(gamma[gold_ix[0]], 1e-12))))
            conf, hit = np.asarray(conf), np.asarray(hit, dtype=float)
            edges = np.linspace(0, 1, 11)
            ece = sum(abs(hit[m].mean() - conf[m].mean()) * m.mean()
                      for lo_, hi_ in zip(edges[:-1], edges[1:])
                      if (m := (conf > lo_) & (conf <= hi_)).any())
            extra["jury_calibration" + tag].append(
                dict(ece=float(ece), brier=float(np.mean(brier)),
                     nll=float(np.mean(nll)), mean_conf=float(conf.mean()),
                     accuracy=float(hit.mean())))

            plural = [router.verdict(qid, np.inf) for qid in te_q]
            in_stratum = {s: np.array([strata[q] == s for q in te_q])
                          for s in ("plurality", "fuel")}
            per_tau = []
            for tau in JURY_TAUS:
                hit, over = [], []
                for n, qid in enumerate(te_q):
                    pick = router.verdict(qid, tau)
                    hit.append(is_correct(tasks_te[n], pick, golds[n]) if pick else 0)
                    over.append(pick != plural[n])
                hit = np.asarray(hit, dtype=float)
                per_tau.append((float(hit.mean()), float(np.mean(over)),
                                float(hit[in_stratum["plurality"]].mean()),
                                float(hit[in_stratum["fuel"]].mean())))
            curves["jury" + (tag or "_greedy")].append(per_tau)

        for tag, flags in (("@modes", dict(collapse_modes=True, per_task=False)),
                           ("@task", dict(collapse_modes=False, per_task=True))):
            cfg.raw["experiments"]["jury"].update(use_rollouts=True, lure="none", **flags)
            router = fit_router("jury", cfg, ctx, seed)
            out = router.predict(ctx)
            record("jury" + tag, out.correct, out.cost)
            if flags["per_task"]:
                extra["jury_pi_task"].append(
                    {t: [float(x) for x in p] for t, p in router.pi_task.items()})
        cfg.raw["experiments"]["jury"].update(collapse_modes=False, per_task=False)

        # The best expert is a comparison like any other and gets the same paired
        # bootstrap: picked on the training half, scored per question on the test
        # half, so "the jury beats the pool's best member" has an interval rather
        # than a t over ten overlapping splits.
        best_j = int(np.argmax(ctx.Ytr.mean(0)))
        record("best_single", ctx.Yte[:, best_j], np.full(len(te), ctx.params_b[best_j]))
        extra["best_single_expert"].append(ctx.expert_ids[best_j])
        # The same expert given the same repeated sampling the jury gets: on
        # GSM8K its answer is the mode of its eight cached samples (the ones
        # the distributional rules read), on multiple choice its greedy answer,
        # which already comes from one forward pass over the four options.
        # Computed for all nine experts and picked on the training half, so the
        # comparison is against whichever expert self-consistency makes best.
        qids_all = ctx.df["qid"].to_numpy()
        golds_all = ctx.df["answer"].to_numpy()
        Ysc = ctx.Y.copy()
        for i in np.where(ctx.tasks == "numeric")[0]:
            for j, e in enumerate(ctx.expert_ids):
                m = mode_dist.get(qids_all[i], {}).get(e)
                Ysc[i, j] = is_correct("numeric", next(iter(m)), golds_all[i]) if m else 0
        best_sc = int(np.argmax(Ysc[tr].mean(0)))
        record("best_single+sc", Ysc[te, best_sc], np.full(len(te), ctx.params_b[best_sc]))
        extra["best_single_sc_expert"].append(ctx.expert_ids[best_sc])
        extra["self_consistency_train"].append(
            {e: float(v) for e, v in zip(ctx.expert_ids, Ysc[tr].mean(0))})
        extra["self_consistency_test"].append(
            {e: float(v) for e, v in zip(ctx.expert_ids, Ysc[te].mean(0))})
        acc["oracle"].append(float(ctx.Yte.max(axis=1).mean()))
    return acc, cost, stratum_acc, extra, curves, perq, by_source


def table_main(acc, cost, stratum_acc, boot):
    ref = REF

    def delta(name):
        b = boot[name]
        star = "$^{*}$" if b["p"] < 0.05 else ""
        return f"{100*b['delta']:+.1f}{star}~[{100*b['lo']:+.1f}, {100*b['hi']:+.1f}]"

    lines = ["\\begin{tabular}{>{\\raggedright\\arraybackslash}p{5.2cm}cccrrr}", "\\toprule",
             "Method & Labels & Params (B) & Accuracy & Plurality & Minority "
             "& $\\Delta$ vs.\\ jury~[95\\% CI] \\\\", "\\midrule",
             f"Best single expert & -- & -- & {fmt(np.mean(acc['best_single']), np.std(acc['best_single']))}"
             f" & -- & -- & {delta('best_single')} \\\\",
             "\\quad + self-consistency on GSM8K & -- & -- & "
             f"{fmt(np.mean(acc['best_single+sc']), np.std(acc['best_single+sc']))}"
             f" & -- & -- & {delta('best_single+sc')} \\\\", "\\midrule"]
    order = ["majority", "ISP", "smoothie_global", "smoothie_local", "OW", "TOPLA",
             "MoRE", "MoRE+rollouts", "soft-equal", "soft-OW"]
    for name in order:
        label, lab, key = LITERATURE[name]
        # The citation travels with the row: a reader comparing two numbers should
        # not have to go looking in the prose for whose method the second one is.
        # The first column is a paragraph column so that an author-year citation
        # can wrap onto a second line instead of widening the table.
        lines.append(
            f"{label}~\\cite{{{key}}} & {lab} & {np.mean(cost[name]):.1f} & "
            f"{fmt(np.mean(acc[name]), np.std(acc[name]))} & "
            f"{fmt(np.mean(stratum_acc[name]['plurality']))} & "
            f"{fmt(np.mean(stratum_acc[name]['fuel']))} & {delta(name)} \\\\")
    lines.append("\\midrule")
    for name, (label, needs) in OURS.items():
        cell = "" if name == ref else delta(name)
        lines.append(
            f"{label} & {needs} & {np.mean(cost[name]):.1f} & "
            f"{fmt(np.mean(acc[name]), np.std(acc[name]))} & "
            f"{fmt(np.mean(stratum_acc[name]['plurality']))} & "
            f"{fmt(np.mean(stratum_acc[name]['fuel']))} & {cell} \\\\")
    lines += ["\\midrule",
              f"Oracle & -- & -- & {fmt(np.mean(acc['oracle']), np.std(acc['oracle']))}"
              " & -- & -- & \\\\",
              "\\bottomrule", "\\end{tabular}"]
    write(TABLEDIR / "main.tex", "\n".join(lines))


def table_decomposition(acc, cost, boot):
    """3 x 3: what the rule is allowed to see, against how it weights the experts.

    Reading down a column isolates the evidence at a fixed weighting scheme, and
    the middle row is the sample-matched control: it has paid for the eight
    samples and then discarded everything but their mode.
    """
    def cell(name):
        return fmt(np.mean(acc[name]), np.std(acc[name]))

    rows = [("Greedy answer, 1 per expert", "majority", "OW", "jury"),
            ("Mode of 8 samples", "majority@modes", "OW@modes", "jury@modes"),
            ("Full 8-sample distribution", "soft-equal", "soft-OW", "jury+rollouts")]
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule",
             "& & \\multicolumn{3}{c}{Weights over the experts} \\\\",
             "\\cmidrule(lr){3-5}",
             "Evidence the rule is given & Params (B) & Equal "
             "& Log-odds, supervised & Log-odds, label-free \\\\", "\\midrule"]
    for label, a, b, c in rows:
        lines.append(f"{label} & {np.mean(cost[a]):.1f} & {cell(a)} & {cell(b)} "
                     f"& {cell(c)} \\\\")
    lines.append("\\midrule")
    lines.append("\\emph{Distribution minus mode} & & "
                 + " & ".join(
                     f"{100*(np.mean(acc[d]) - np.mean(acc[m])):+.1f}"
                     for d, m in (("soft-equal", "majority@modes"),
                                  ("soft-OW", "OW@modes"),
                                  ("jury+rollouts", "jury@modes")))
                 + " \\\\")
    lines.append("\\emph{Mode minus greedy} & & "
                 + " & ".join(
                     f"{100*(np.mean(acc[m]) - np.mean(acc[g])):+.1f}"
                     for m, g in (("majority@modes", "majority"),
                                  ("OW@modes", "OW"), ("jury@modes", "jury")))
                 + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(TABLEDIR / "decomposition.tex", "\n".join(lines))


def table_specialisation(acc, boot, eq):
    """One reliability per expert against one per expert and task."""
    def cell(name):
        return fmt(np.mean(acc[name]), np.std(acc[name]))

    def gap(new, old):
        d = 100 * (np.mean(acc[new]) - np.mean(acc[old]))
        return f"{d:+.1f}"

    lines = ["\\begin{tabular}{llrrr}", "\\toprule",
             "Weights & Labels & Global & Per task & $\\Delta$ \\\\", "\\midrule",
             f"Supervised linear fusion & yes & {cell('soft-OW')} & "
             f"{cell('soft-OW@task')} & {gap('soft-OW@task', 'soft-OW')} \\\\",
             f"Jury (ours) & no & {cell('jury+rollouts')} & "
             f"{cell('jury@task')} & {gap('jury@task', 'jury+rollouts')} \\\\",
             "\\midrule",
             "\\multicolumn{5}{l}{\\emph{Label-free minus supervised, same "
             "parameterisation}} \\\\",
             f"\\quad global & & {100*eq['global']['delta']:+.1f} "
             f"[{100*eq['global']['lo95']:+.1f}, {100*eq['global']['hi95']:+.1f}] & & \\\\",
             f"\\quad per task & & {100*eq['task']['delta']:+.1f} "
             f"[{100*eq['task']['lo95']:+.1f}, {100*eq['task']['hi95']:+.1f}] & & \\\\",
             "\\bottomrule", "\\end{tabular}"]
    write(TABLEDIR / "specialisation.tex", "\n".join(lines))


def table_by_source(acc, by_source):
    """Accuracy per source dataset, for the rules a reader compares.

    Three quarters of the benchmark is ARC and HellaSwag, and those two changed
    the pool's ordering: an overall number alone cannot say whether the result is
    carried by one of them or holds across all four.
    """
    rows = [("Best single expert", "best_single"),
            ("\\quad + self-consistency on GSM8K", "best_single+sc"),
            ("Majority vote", "majority"),
            ("Equal-weight pool", "soft-equal"),
            ("MoRE + repeated sampling", "MoRE+rollouts"),
            ("Supervised linear fusion", "soft-OW"),
            ("\\textbf{Jury (ours) + repeated sampling}", "jury+rollouts")]
    lines = ["\\begin{tabular}{lrrrrr}", "\\toprule",
             "Method & " + " & ".join(SOURCE_ORDER) + " & Overall \\\\",
             "\\midrule"]
    for label, k in rows:
        cells = [f"{100*np.mean(by_source[k][s]):.1f}" if by_source[k].get(s) else "--"
                 for s in SOURCE_ORDER]
        lines.append(f"{label} & " + " & ".join(cells)
                     + f" & {100*np.mean(acc[k]):.1f} \\\\")
    lines += ["\\midrule",
              "Questions & " + " & ".join(str(n) for n in (240, 120, 600, 600))
              + " & 1560 \\\\",
              "\\bottomrule", "\\end{tabular}"]
    write(TABLEDIR / "bysource.tex", "\n".join(lines))


def table_override(curves):
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule",
             "$\\tau$ & Override rate & Accuracy & Plurality & Minority \\\\",
             "\\midrule",
             "\\multicolumn{5}{l}{\\emph{Jury score, with repeated sampling}} \\\\"]
    J = np.array(curves["jury+rollouts"])   # [seed, tau, (acc, override, plur, fuel)]
    for ti, tau in enumerate(JURY_TAUS):
        name = "$0$ (trust the jury)" if tau == 0 else f"{tau:.1f}"
        lines.append(f"\\quad {name} & {100*J[:, ti, 1].mean():.1f} & "
                     f"{100*J[:, ti, 0].mean():.1f} & {100*J[:, ti, 2].mean():.1f} & "
                     f"{100*J[:, ti, 3].mean():.1f} \\\\")
    lines.append("\\midrule")
    for tag, label in (("greedy", "discriminative scorer, greedy answers"),
                       ("+rollouts", "discriminative scorer, with repeated sampling")):
        A = np.array([c[0] for c in curves[tag]])
        O = np.array([c[1] for c in curves[tag]])
        PL = np.array([c[2] for c in curves[tag]])
        FU = np.array([c[3] for c in curves[tag]])
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{label}}}}} \\\\")
        for ti, tau in enumerate(TAUS):
            if np.isfinite(tau) and (tau > 0.45 or ti % 6):
                continue
            name = "$\\infty$ (consensus)" if not np.isfinite(tau) else f"{tau:.2f}"
            lines.append(f"\\quad {name} & {100*O[:, ti].mean():.1f} & "
                         f"{100*A[:, ti].mean():.1f} & {100*np.nanmean(PL[:, ti]):.1f} & "
                         f"{100*np.nanmean(FU[:, ti]):.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(TABLEDIR / "override.tex", "\n".join(lines))


def figure(acc, cost, pool):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                                   # pragma: no cover
        print(f"[figures] skipped ({exc})")
        return
    # The split that matters is what each rule is allowed to see, not who wrote
    # it: every method above the pack consumes answer distributions.
    groups = {"over argmax answers": (["majority", "OW", "smoothie_global",
                                       "smoothie_local", "ISP", "TOPLA", "MoRE",
                                       "aggregator"], "#4292c6", "o"),
              "over answer distributions": (["soft-equal", "soft-OW",
                                             "MoRE+rollouts",
                                             "aggregator+rollouts"], "#e6550d", "D"),
              "jury, label-free (ours)": (["jury", "jury+rollouts"], "#238b45", "*")}
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for label, (names, colour, marker) in groups.items():
        ax.scatter([np.mean(cost[n]) for n in names],
                   [100 * np.mean(acc[n]) for n in names],
                   s=190 if marker == "*" else 64, color=colour, marker=marker,
                   edgecolor="white", zorder=5 if marker == "*" else 4, label=label)
        for n in names:
            dx, dy, ha = LABEL_POS.get(n, (6, -3, "left"))
            weight = "bold" if n.startswith("jury") else "normal"
            ax.annotate(n.replace("_", " "), (np.mean(cost[n]), 100 * np.mean(acc[n])),
                        fontsize=7.5, xytext=(dx, dy), textcoords="offset points",
                        color=colour, ha=ha, fontweight=weight)
    for y, label, style in ((100 * pool["oracle"], "oracle", "--"),
                            (100 * pool["per_task_ceiling"],
                             "source-only routing baseline", ":"),
                            (100 * pool["best_single"], "best single expert", "-.")):
        ax.axhline(y, ls=style, lw=1.1, color="#737373")
        ax.text(0.015, y + 0.7, label, transform=ax.get_yaxis_transform(),
                fontsize=8, color="#737373")
    ax.set_xscale("log")
    ax.set_xlim(1.6, 95)
    ax.set_ylim(60.5, 94.5)
    ax.set_xlabel("Average parameters activated per question (B, log scale)")
    ax.set_ylabel("Accuracy (\\%)".replace("\\", ""))
    ax.set_title("Rules that see only each expert's argmax answer sit at the bottom;\n"
                 "the top of the figure is where the answer distributions are kept",
                 fontsize=10)
    ax.grid(alpha=0.25, zorder=0)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
    plt.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "jury_frontier.png", dpi=150)
    print(f"[figures] -> {FIGDIR / 'jury_frontier.png'}")


def main() -> None:
    cfg = load_config(ROOT / "config.yaml")
    # --cache points the whole pipeline at another cache directory without
    # editing the config.
    if "--cache" in sys.argv:
        cfg.raw["paths"]["cache_dir"] = sys.argv[sys.argv.index("--cache") + 1]
    cache = Path(cfg["paths"]["cache_dir"])
    print(f"[paper] cache: {cache}")
    df = pd.read_csv(cache / "questions.csv").fillna("")
    preds = pd.read_csv(cache / "predictions.csv")

    d = df.copy()
    d["split"] = _resplit(d, cfg["dataset"]["train_frac"], SEEDS[0])
    cfg.raw["seed"] = SEEDS[0]
    ctx0 = build_context(cfg, d, preds, cache)
    G = np.vectorize(answer_key, otypes=[object])(
        np.repeat(ctx0.tasks[:, None], len(ctx0.expert_ids), axis=1),
        prediction_matrix(ctx0))
    dist, n_samples = rollout_distributions(cache, set(ctx0.expert_ids))

    pool = table_experts(cfg, ctx0, df)
    reach = section1_reachability(df, G, dist, ctx0.expert_ids)
    table_reachability(reach)
    fuel = section2_fuel(df, G)
    strata = dict(zip(fuel.qid, fuel.stratum))

    acc, cost, stratum_acc, extra, curves, perq, by_source = measure(
        cfg, df, preds, cache, strata)
    boot = bootstrap(perq, REF)
    # The greedy comparison of the Results: Optimal Weight against the greedy
    # jury, with the task-defined K and with the jury's own m = |V(q)| - 1.
    boot_greedy = bootstrap({k: perq[k] for k in ("jury", "OW", "OW@cand")}, "jury")
    eq = {"global": equivalence(perq, REF, "soft-OW"),
          "task": equivalence(perq, "jury@task", "soft-OW@task")}
    table_main(acc, cost, stratum_acc, boot)
    table_by_source(acc, by_source)
    table_decomposition(acc, cost, boot)
    table_specialisation(acc, boot, eq)
    table_override(curves)
    figure(acc, cost, pool)

    share = fuel.stratum.value_counts(normalize=True).to_dict()
    picks = defaultdict(int)
    for sub in extra["topla_subset"]:
        picks[sub] += 1
    summary = {
        "seeds": SEEDS, "n_questions": int(len(df)), "n_samples": int(n_samples),
        "pool": pool, "strata_share": share,
        "fuel_n_gold_mean": float(fuel.loc[fuel.stratum == "fuel", "n_gold"].mean()),
        "fuel_plurality_mean": float(fuel.loc[fuel.stratum == "fuel", "plurality"].mean()),
        "reachability": {c: float(reach[c].mean()) for c in
                         ("vote_greedy", "vote_modes", "fusion", "fusion_signed",
                          "in_support")
                         if c in reach},
        "n_candidates_mean": float(reach.n_keys.mean()),
        "accuracy": {k: v for k, v in acc.items()},
        "params_b": {k: v for k, v in cost.items()},
        "stratum_accuracy": {k: dict(v) for k, v in stratum_acc.items()},
        "fuel_insample": {k: extra[k] for k in ("fuel_insample", "fuel_insample+rollouts")},
        "topla_subset": max(picks.items(), key=lambda kv: kv[1])[0],
        "jury_pi": {k: extra[k] for k in ("jury_pi", "jury_pi+rollouts")},
        "jury_calibration": {k: extra[k] for k in
                             ("jury_calibration", "jury_calibration+rollouts")},
        "jury_override": {k: curves[k] for k in ("jury_greedy", "jury+rollouts")},
        "jury_taus": JURY_TAUS,
        "bootstrap_vs_jury": boot,
        "bootstrap_vs_greedy_jury": boot_greedy,
        "equivalence": eq,
        "jury_pi_task": extra["jury_pi_task"],
        "best_single_expert": extra["best_single_expert"],
        "best_single_sc_expert": extra["best_single_sc_expert"],
        "self_consistency_train": extra["self_consistency_train"],
        "self_consistency_test": extra["self_consistency_test"],
        "paired_vs_jury": {
            k: dict(zip(("delta", "std", "t", "wins"),
                        [float(x) for x in paired(acc[REF], acc[k])]))
            for k in acc if k not in (REF, "oracle", "best_single", "best_single+sc")},
        "paired_vs_aggregator": {
            k: dict(zip(("delta", "std", "t", "wins"),
                        [float(x) for x in paired(acc["aggregator"], acc[k])]))
            for k in acc if k not in ("aggregator", "oracle", "best_single", "best_single+sc")},
        "paired_vs_majority": {
            k: dict(zip(("delta", "std", "t", "wins"),
                        [float(x) for x in paired(acc[k], acc["majority"])]))
            for k in acc if k not in ("majority", "oracle", "best_single", "best_single+sc")},
    }
    out = Path(cfg["paths"]["results_dir"]) / "paper_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n[results] -> {out}")
    print(f"[results] fuel share {100*share.get('fuel', 0):.1f}%")
    for n in ("majority", "MoRE", "MoRE+rollouts", "aggregator",
              "aggregator+rollouts", "jury", "jury+rollouts"):
        d = paired(acc[REF], acc[n])
        tail = "" if n == REF else f"   vs {REF}: {100*d[0]:+.1f} ({d[2]:+.2f}, {d[3]}/10)"
        print(f"           {n:<22}{100*np.mean(acc[n]):5.1f}% "
              f"@ {np.mean(cost[n]):5.2f}B{tail}")


if __name__ == "__main__":
    main()
