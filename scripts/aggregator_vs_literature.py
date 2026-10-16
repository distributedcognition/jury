"""`aggregator` against the published post-hoc combiners it is closest to.

The techniques already in `router_lab/` are compared by `main.py multiseed`. This
script adds the methods from the LLM-ensemble literature that the benchmark does
*not* implement but that `aggregator` has to answer to, re-implemented here and
run on the same ten splits, the same experts and the same cached answers:

    python paper/scripts/aggregator_vs_literature.py       # from the repo root

  majority    "More Agents Is All You Need" (Li et al., 2024) -- unweighted
              plurality over the pool. No training signal.
  OW          Optimal Weight, from "Beyond Majority Voting: LLM Aggregation by
              Leveraging Higher-Order Information" (2025, arXiv:2510.01499).
              Weight each expert by the inverse sigmoid of its accuracy,
              omega_i = sigma_K^-1(x_i) with sigma_K(x) = e^x / (K - 1 + e^x),
              then take the weighted plurality. Needs labels (we fit x_i on the
              training half); this is the K-class Nitzan-Paroush log-odds rule.
  ISP         Inverse Surprising Popularity, same paper. Label-free: estimate the
              second-order table P(A_i = s | A_j = a) from the training half and
              pick argmax_s [ votes(s) - sum_i S_isp(s, i) ], i.e. the answer
              that is more popular than the pairwise statistics predict.
  OW-L        Optimal Weight with accuracies *learned* without labels, same
              paper (Section 5.2, eq. 6, Appendix F.2). Under their symmetric
              model the second-order table P(A_i = s_k | A_j = s_l) is a known
              function of the accuracies x_1..x_N; x is the least-squares fit
              of that function to the empirical table, and OW is then applied
              with sigma_K^-1(x_i).
  OW-I        Optimal Weight with accuracies read off ISP's verdicts, same
              paper (Section 5.2): x_i is the fraction of unlabelled questions
              on which expert i agrees with ISP, then OW with sigma_K^-1(x_i).
  smoothie    "Smoothie: Label Free Language Model Routing" (Guha et al., 2024).
              The triplet identity E||l_i - z*||^2 = 1/2 (d_ij + d_ik - d_jk)
              recovers each expert's error from pairwise distances alone, with no
              labels; route to the expert with the highest quality score. Global
              estimates one score per expert, Local re-estimates the distances on
              the k nearest training questions of each test question.
  soft-equal  The linear opinion pool (Stone, 1961) with equal weights:
              argmax_v sum_e p_e(v) over the sampled answer distributions. No
              weights and no labels -- the control that says how much of any
              distributional method's gain is simply *not* taking each expert's
              argmax before combining.
  soft-OW     The same pool with the supervised log-odds weights of OW, i.e.
              argmax_v sum_e omega_e p_e(v). This is the strongest baseline in
              the comparison and the one `jury` has to answer to: identical
              functional form, weights fitted on labels instead of by EM.
  MoRE        "Getting MoRE out of Mixture of Language Model Reasoning Experts"
              (Si et al., 2023). A supervised selector over candidate answers
              whose features are inter-expert agreement and the experts' own
              confidence.
  TOPLA       "LLM-TOPLA: Efficient LLM Ensemble by Maximising Diversity" (Tekin
              et al., 2024). Pick the ensemble subset that maximises focal
              diversity, then combine with the learned weighted combiner.

Two of the neighbours cannot be run on this benchmark and are left out rather
than approximated into something else:

  LLM-Blender  PairRanker is a cross-encoder over the *text* of the candidates.
               Here a candidate is a single letter or a number, so there is no
               text to rank and the reranker has no input.
  LENS         Learns confidence from each model's layer-wise hidden states; the
               cache stores answers and a scalar self-confidence, not activations.

Adaptations are flagged in each function's docstring. Everything reuses the
cached predictions; no LLM is run.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]          # this supplementary folder
sys.path.insert(0, str(ROOT))

from scipy.optimize import minimize                             # noqa: E402
from sklearn.linear_model import LogisticRegression             # noqa: E402
from sklearn.neural_network import MLPClassifier                # noqa: E402

from router_lab.data import is_correct                          # noqa: E402
from router_lab.techniques.aggregator import (answer_key,        # noqa: E402
                                              rollout_distributions)
from router_lab.techniques.base import (build_context,          # noqa: E402
                                        prediction_matrix,
                                        selfconf_matrix)
from router_lab.techniques.multiseed import _resplit            # noqa: E402
from router_lab.techniques.runner import fit_router             # noqa: E402
from router_lab.utils import load_config                        # noqa: E402

SEEDS = list(range(1, 11))
OURS = ["ensemble", "soft_actor", "aggregator"]
LETTERS = ["A", "B", "C", "D"]
KNN_K = 15
SMOOTHING = 1.0


# --------------------------------------------------------------------------- #
# Helpers shared by the baselines
# --------------------------------------------------------------------------- #
def answers_matrix(ctx):
    """[n_questions, n_experts] canonical answers, df-aligned."""
    tasks = np.repeat(ctx.tasks[:, None], len(ctx.expert_ids), axis=1)
    return np.vectorize(answer_key, otypes=[object])(tasks, prediction_matrix(ctx))


def candidates(row) -> dict:
    out = defaultdict(list)
    for j, a in enumerate(row):
        if a is not None:
            out[a].append(j)
    return out


def score_choice(ctx, rows, pick_fn, cost_fn):
    """Run a decision rule over the test rows -> (correct, cost) arrays."""
    golds = ctx.df["answer"].to_numpy()
    correct = np.zeros(len(rows), dtype=int)
    cost = np.zeros(len(rows), dtype=np.float32)
    for n, i in enumerate(rows):
        choice = pick_fn(i)
        correct[n] = is_correct(ctx.tasks[i], choice, golds[i]) if choice is not None else 0
        cost[n] = cost_fn(i)
    return correct, cost


# --------------------------------------------------------------------------- #
# More Agents Is All You Need -- unweighted plurality
# --------------------------------------------------------------------------- #
def run_majority(ctx, A, tr, te):
    def pick(i):
        c = candidates(A[i])
        return max(c, key=lambda k: len(c[k])) if c else None
    return score_choice(ctx, te, pick, lambda i: float(ctx.params_b.sum()))


# --------------------------------------------------------------------------- #
# OW / ISP  (arXiv:2510.01499)
# --------------------------------------------------------------------------- #
def run_ow(ctx, A, tr, te):
    """omega_i = sigma_K^-1(x_i) = log(x_i (K-1) / (1 - x_i)), weighted plurality.

    x_i is expert i's accuracy on the training half. K is the size of the answer
    space: 4 for multiple choice; for GSM8K, where the space is unbounded, we use
    the number of distinct answers actually on the table for that question, which
    is the only finite K the data defines.
    """
    x = np.clip(ctx.Y[tr].mean(0), 1e-3, 1 - 1e-3)

    def pick(i):
        c = candidates(A[i])
        if not c:
            return None
        K = 4 if ctx.tasks[i] == "mc" else max(2, len(c))
        w = np.log(x * (K - 1) / (1 - x))
        return max(c, key=lambda k: sum(w[j] for j in c[k]))
    return score_choice(ctx, te, pick, lambda i: float(ctx.params_b.sum()))


def run_ow_cand(ctx, A, tr, te):
    """Optimal Weight under the jury's convention: m = |V(q)| - 1 on every task.

    The same supervised accuracies as `run_ow`, but the log-odds weight uses the
    number of distinct answers actually proposed for the question, as the jury's
    emission does, instead of the task-defined K = 4 on multiple choice. Against
    the greedy jury this isolates how the reliabilities are estimated and nothing
    else; the two rules coincide whenever all four options are proposed.
    """
    x = np.clip(ctx.Y[tr].mean(0), 1e-3, 1 - 1e-3)

    def pick(i):
        c = candidates(A[i])
        if not c:
            return None
        m = max(1, len(c) - 1)
        w = np.log(x * m / (1 - x))
        return max(c, key=lambda k: sum(w[j] for j in c[k]))
    return score_choice(ctx, te, pick, lambda i: float(ctx.params_b.sum()))


def _letter(v):
    """Letter index, or -1 when the expert did not answer.

    `A` holds None for an expert whose output did not parse, and `None in
    LETTERS` raises rather than returning False. Every expert in this pool
    answers every multiple-choice question, so the direct indexing this
    replaces never faulted here; it faults on any pool with an abstainer,
    which is not a property a baseline implementation should depend on.
    """
    return LETTERS.index(v) if isinstance(v, str) and v in LETTERS else -1


def _letters(A, rows):
    """[len(rows), n_experts] letter indices of the greedy answers, -1 = none."""
    return np.array([[_letter(v) for v in A[i]] for i in rows], dtype=int).reshape(len(rows), -1)


def _isp_rule(ctx, A, fit):
    """ISP's decision rule, with its second-order table P(A_i = s | A_j = a)
    estimated, Laplace-smoothed and without labels, on the multiple-choice
    questions among `fit`. Returns pick(i) -> answer or None."""
    n_exp = len(ctx.expert_ids)
    li = _letter
    L = _letters(A, [i for i in fit if ctx.tasks[i] == "mc"])

    # P[i, j, s, a] = P(A_i = s | A_j = a), Laplace-smoothed
    joint = np.full((n_exp, n_exp, 4, 4), SMOOTHING)
    for row in L:
        for p in range(n_exp):
            if row[p] < 0:
                continue
            for qx in range(n_exp):
                if row[qx] >= 0:
                    joint[p, qx, row[p], row[qx]] += 1
    cond = joint / joint.sum(axis=2, keepdims=True)

    def pick(i):
        c = candidates(A[i])
        if not c:
            return None
        if ctx.tasks[i] != "mc":
            return max(c, key=lambda k: len(c[k]))
        best, best_score = None, -np.inf
        for s in c:
            si = li(s)
            if si < 0:
                continue
            votes = len(c[s])
            penalty = 0.0
            for p in range(n_exp):
                acc, seen = 0.0, 0
                for qx in range(n_exp):
                    aj = li(A[i, qx])
                    if qx == p or aj < 0:
                        continue
                    others = [a for a in range(4) if a != aj]
                    acc += float(np.mean(cond[p, qx, si, others]))
                    seen += 1
                penalty += acc / seen if seen else 0.0
            if votes - penalty > best_score:
                best, best_score = s, votes - penalty
        return best if best is not None else max(c, key=lambda k: len(c[k]))
    return pick


def run_isp(ctx, A, tr, te):
    """argmax_s [ votes(s) - sum_i S_isp(s, i) ] with the second-order table
    P(A_i = s | A_j = a) estimated, without labels, on the training half.

    The rule needs a fixed finite answer space S, so it applies to the multiple
    choice two thirds; on GSM8K, where S is unbounded and every conditional would be
    estimated from a handful of co-occurrences, it falls back to plurality.
    """
    return score_choice(ctx, te, _isp_rule(ctx, A, tr), lambda i: float(ctx.params_b.sum()))


def _ow_pick(ctx, A, x):
    """Optimal Weight's weighted plurality with label-free accuracies `x` and
    the task-defined K = 4, on multiple choice. The accuracies are estimated on
    multiple-choice questions only, so on GSM8K the rule falls back to plurality,
    as ISP does there."""
    x = np.clip(np.asarray(x, dtype=float), 1e-3, 1 - 1e-3)
    w = np.log(x * 3 / (1 - x))

    def pick(i):
        c = candidates(A[i])
        if not c:
            return None
        if ctx.tasks[i] != "mc":
            return max(c, key=lambda k: len(c[k]))
        return max(c, key=lambda k: sum(w[j] for j in c[k]))
    return pick


def ow_l_accuracies(L, K=4, starts=(0.3, 0.45, 0.6, 0.75, 0.9)):
    """Equation (6) of Ai et al.: the accuracies x_1..x_N that best reproduce the
    empirical second-order table, by least squares, from the greedy answers alone.

    `L` is [n_questions, n_experts] of letter indices (-1 = no answer). Under
    the paper's model (Proposition 2.1, every wrong answer equally likely), the
    conditional P(A_i = s_k | A_j = s_l) is, for i != j (their Appendix F.2),

        x_i x_j + (1 - x_i)(1 - x_j) / (K - 1)                        if k == l,
        [x_i (1 - x_j) + (1 - x_i) x_j] / (K - 1)
            + (K - 2)(1 - x_i)(1 - x_j) / (K - 1)^2                   if k != l,

    and the empirical table is #{A_i = k, A_j = l} / #{A_j = l}. The paper solves
    the least-squares problem by gradient descent and does not give a start; we
    run L-BFGS-B in [1e-3, 1 - 1e-3] from five uniform starts and keep the
    lowest loss. On this pool every start at or below 0.75 reaches the same
    point; a start at 0.9 runs to the corner x ~ 0 (every expert wrong, which
    also makes experts agree), at a loss several times higher.
    """
    n_exp = L.shape[1]
    emp = np.zeros((n_exp, n_exp, K, K))
    mask = np.zeros((n_exp, n_exp, K, K), dtype=bool)
    for i in range(n_exp):
        for j in range(n_exp):
            if i == j:
                continue
            ok = (L[:, i] >= 0) & (L[:, j] >= 0)
            joint = np.zeros((K, K))
            np.add.at(joint, (L[ok, i], L[ok, j]), 1.0)
            col = joint.sum(0)
            emp[i, j] = joint / np.maximum(col, 1.0)[None, :]
            mask[i, j] = (col > 0)[None, :].repeat(K, 0)
    same = np.eye(K, dtype=bool)[None, None]

    def model(x):
        xi, xj = x[:, None, None, None], x[None, :, None, None]
        diag = xi * xj + (1 - xi) * (1 - xj) / (K - 1)
        off = ((xi * (1 - xj) + (1 - xi) * xj) / (K - 1)
               + (K - 2) * (1 - xi) * (1 - xj) / (K - 1) ** 2)
        return np.where(same, diag, off)

    def loss(x):
        r = np.where(mask, model(x) - emp, 0.0)
        return float((r ** 2).sum())

    best = None
    for s0 in starts:
        res = minimize(loss, np.full(n_exp, s0), method="L-BFGS-B",
                       bounds=[(1e-3, 1 - 1e-3)] * n_exp)
        if best is None or res.fun < best.fun:
            best = res
    return np.clip(best.x, 1e-3, 1 - 1e-3)


def run_ow_l(ctx, A, tr, te, fit=None):
    """OW-L: Optimal Weight with accuracies learned from the second-order table
    (Ai et al., Section 5.2). `fit` is the set of unlabelled questions the table
    is estimated on: the training half by default, as for ISP and the jury; the
    paper's own protocol fits on every question of the dataset, test included,
    which `fit=range(n)` reproduces. Multiple choice only, like ISP; GSM8K falls
    back to plurality.
    """
    fit = tr if fit is None else fit
    L = _letters(A, [i for i in fit if ctx.tasks[i] == "mc"])
    x = ow_l_accuracies(L)
    correct, cost = score_choice(ctx, te, _ow_pick(ctx, A, x),
                                 lambda i: float(ctx.params_b.sum()))
    return correct, cost, x


def run_ow_i(ctx, A, tr, te, fit=None):
    """OW-I: Optimal Weight with accuracies read off ISP's verdicts (Ai et al.,
    Section 5.2): x_i = #{f_ISP(a_1..a_N) = a_i} / M over the unlabelled
    multiple-choice questions in `fit`, with ISP's own table estimated on the
    same questions. `fit` defaults to the training half; see `run_ow_l`.
    """
    fit = tr if fit is None else fit
    mc_fit = [i for i in fit if ctx.tasks[i] == "mc"]
    isp = _isp_rule(ctx, A, fit)
    pseudo = np.array([_letter(isp(i)) for i in mc_fit])
    L = _letters(A, mc_fit)
    x = (L == pseudo[:, None]).mean(0)
    correct, cost = score_choice(ctx, te, _ow_pick(ctx, A, x),
                                 lambda i: float(ctx.params_b.sum()))
    return correct, cost, x


# --------------------------------------------------------------------------- #
# Smoothie (arXiv:2412.04692)
# --------------------------------------------------------------------------- #
def _smoothie_scores(A, rows):
    """Triplet estimate of each expert's error from pairwise distances only.

    Smoothie embeds [input, output] with SentenceBERT; on a discrete answer space
    the embedding of an answer is its indicator, so ||l_i - l_j||^2 = 2 * 1{a_i !=
    a_j} and the paper's delta_ij reduces exactly to the pairwise disagreement
    rate. The triplet identity is unchanged, and the dimension d in theta = d/e
    cancels in the argmax, so it is dropped.
    """
    n_exp = A.shape[1]
    delta = np.zeros((n_exp, n_exp))
    for p in range(n_exp):
        for qx in range(p + 1, n_exp):
            dis = [A[i, p] != A[i, qx] for i in rows
                   if A[i, p] is not None and A[i, qx] is not None]
            delta[p, qx] = delta[qx, p] = 2.0 * (np.mean(dis) if dis else 1.0)
    err = np.zeros(n_exp)
    for p in range(n_exp):
        vals = [0.5 * (delta[p, j] + delta[p, k] - delta[j, k])
                for j, k in combinations([m for m in range(n_exp) if m != p], 2)]
        err[p] = max(float(np.mean(vals)), 1e-6)
    return 1.0 / err                                   # theta, up to the constant d


def run_smoothie_global(ctx, A, tr, te):
    """One quality score per expert over the whole training half; route to the
    argmax. Only that expert runs, so the cost is its own parameter count."""
    theta = _smoothie_scores(A, tr)
    best = int(np.argmax(theta))
    return (np.array([ctx.Y[i, best] for i in te]),
            np.full(len(te), float(ctx.params_b[best]), np.float32))


def run_smoothie_local(ctx, A, tr, te):
    """Smoothie-Local: re-estimate the pairwise distances on the k nearest
    training questions of each test question, then route. The neighbours supply
    the statistics, so the test question still costs one expert."""
    sims = ctx.emb[te] @ ctx.emb[tr].T                 # embeddings are L2-normalised
    nn = np.asarray(tr)[np.argsort(-sims, axis=1)[:, :KNN_K]]
    correct = np.zeros(len(te), dtype=int)
    cost = np.zeros(len(te), dtype=np.float32)
    for n, i in enumerate(te):
        best = int(np.argmax(_smoothie_scores(A, nn[n])))
        correct[n] = ctx.Y[i, best]
        cost[n] = float(ctx.params_b[best])
    return correct, cost


# --------------------------------------------------------------------------- #
# Linear opinion pools over the sampled distributions
# --------------------------------------------------------------------------- #
def _soft_table(ctx, A, dist, fixed_mc=False):
    """qid -> (candidates, [n_experts, n_cands] row-stochastic distributions).

    An expert with no cached samples falls back to a point mass on its greedy
    answer; one that did not answer at all contributes a zero row, so it is
    absent from the pool rather than voting uniformly. `fixed_mc` keeps the four
    options of a multiple-choice question as its candidate set whether or not
    each was proposed, the jury's `support: task` convention.
    """
    ids = list(ctx.expert_ids)
    out = {}
    for i, qid in enumerate(ctx.df["qid"].to_numpy()):
        keys = [a for a in A[i] if a is not None]
        for e in ids:
            keys += list(dist.get(qid, {}).get(e, {}))
        if fixed_mc and ctx.tasks[i] == "mc":
            keys = [k for k in keys if k in LETTERS] + list(LETTERS)
        cands = sorted(set(keys))
        if not cands:
            out[qid] = None
            continue
        idx = {c: j for j, c in enumerate(cands)}
        P = np.zeros((len(ids), len(cands)))
        for j, e in enumerate(ids):
            for k, v in dist.get(qid, {}).get(e, {}).items():
                if k in idx:
                    P[j, idx[k]] += v
            if P[j].sum() <= 0 and A[i, j] in idx:
                P[j, idx[A[i, j]]] = 1.0
            if P[j].sum() > 0:
                P[j] /= P[j].sum()
        out[qid] = (cands, P)
    return out


def _run_pool(ctx, A, te, dist, n_samples, weight_fn, fixed_mc=False):
    """Shared driver: score every candidate by <w, p(.|v)> and take the argmax."""
    S = _soft_table(ctx, A, dist, fixed_mc)
    qids = ctx.df["qid"].to_numpy()

    def pick(i):
        row = S[qids[i]]
        if row is None:
            return None
        cands, P = row
        return cands[int(np.argmax(weight_fn(len(cands), ctx.tasks[i]) @ P))]

    correct, _ = score_choice(ctx, te, pick, lambda i: 0.0)
    cost = np.full(len(te), float(ctx.params_b.sum()), np.float32)
    if n_samples > 1:
        cost[[ctx.tasks[i] == "numeric" for i in te]] *= n_samples
    return correct, cost


def run_soft_equal(ctx, A, tr, te, dist=None, n_samples=1):
    """argmax_v sum_e p_e(v): the opinion pool with no weights and no labels."""
    n = len(ctx.expert_ids)
    return _run_pool(ctx, A, te, dist or {}, n_samples, lambda K, t: np.ones(n))


def run_soft_ow(ctx, A, tr, te, dist=None, n_samples=1, fixed_mc=False):
    """argmax_v sum_e omega_e p_e(v) with OW's supervised log-odds weights.

    The reliability of expert e is its greedy accuracy on the training half,
    x_e = mean_q 1[a_{e,q} = y_q] over every training question, unanimous ones
    included and a non-answer counted as wrong, clipped to [1e-3, 1 - 1e-3];
    the weight is log(x_e (K - 1) / (1 - x_e)) with K the size of the
    question's candidate set. Same weights as `run_ow`, applied to the
    distributions rather than to the argmax answers, so the pair isolates what
    taking the argmax costs. `fixed_mc` applies the four-option support.
    """
    x = np.clip(ctx.Y[tr].mean(0), 1e-3, 1 - 1e-3)
    return _run_pool(ctx, A, te, dist or {}, n_samples,
                     lambda K, t: np.log(x * max(K - 1, 1) / (1 - x)), fixed_mc)


def run_soft_ow_dist(ctx, A, tr, te, dist=None, n_samples=1):
    """The distributional supervised plug-in: the jury's M-step with the gold.

    Replacing the posterior gamma_q(v) of the jury's update with the point mass
    on the gold answer turns sum_v gamma_q(v) p_{e,q}(v) into p_{e,q}(y_q), so
    the reliability of expert e is the mass it put on the gold answer, averaged
    over every training question (zero when it put none, or did not answer),
    clipped to [1e-3, 1 - 1e-3] and turned into the same log-odds weight.
    Exactly the jury's estimator with the label in place of the latent answer.
    """
    S = _soft_table(ctx, A, dist or {})
    qids = ctx.df["qid"].to_numpy()
    golds = ctx.df["answer"].to_numpy()
    n = len(ctx.expert_ids)
    mass = np.zeros(n)
    for i in tr:
        row = S[qids[i]]
        y = answer_key(ctx.tasks[i], golds[i])
        if row is None or y not in row[0]:
            continue
        mass += row[1][:, row[0].index(y)]
    x = np.clip(mass / len(tr), 1e-3, 1 - 1e-3)
    return _run_pool(ctx, A, te, dist or {}, n_samples,
                     lambda K, t: np.log(x * max(K - 1, 1) / (1 - x)))


def run_soft_ow_task(ctx, A, tr, te, dist=None, n_samples=1):
    """soft-OW with one supervised weight vector per task instead of one global.

    Same functional form and the same labels, but the accuracy each weight is
    built from is measured on the training questions of that task alone, so the
    rule can express that an expert is strong on multiple choice and weak on
    arithmetic. This is the supervised half of the specialisation control: a
    global weight is a restriction, and the question is whether lifting it helps
    supervision more than it helps the label-free estimator.
    """
    tasks = sorted(set(ctx.tasks))
    x = {}
    for t in tasks:
        rows = [i for i in tr if ctx.tasks[i] == t]
        x[t] = np.clip(ctx.Y[rows].mean(0), 1e-3, 1 - 1e-3) if rows \
            else np.clip(ctx.Y[tr].mean(0), 1e-3, 1 - 1e-3)
    return _run_pool(ctx, A, te, dist or {}, n_samples,
                     lambda K, t: np.log(x[t] * max(K - 1, 1) / (1 - x[t])))


# --------------------------------------------------------------------------- #
# MoRE (arXiv:2305.14628)
# --------------------------------------------------------------------------- #
def _more_rows(ctx, A, conf, rows, dist=None):
    """MoRE's feature set: inter-expert agreement plus the experts' confidence.

    Deliberately excludes what `aggregator` adds on top -- regional reliability
    and the family diversity of the coalition -- so the gap between the two rows
    in the table is exactly what those extra features buy.

    With `dist` the same sampled mass that `aggregator+rollouts` uses is added,
    which is the control that separates "the new features help" from "repeated
    sampling helps": without it the two variants differ on both at once.
    """
    n_exp = len(ctx.expert_ids)
    golds = ctx.df["answer"].to_numpy()
    X, y, grp, cand = [], [], [], []
    for i in rows:
        c = candidates(A[i])
        if dist is not None:
            qid = ctx.df["qid"].to_numpy()[i]
            for e in ctx.expert_ids:
                for key in dist.get(qid, {}).get(e, {}):
                    c.setdefault(key, c.get(key, []))
        if not c:
            continue
        top = max((len(v) for v in c.values()), default=0)
        for k, js in c.items():
            s = np.zeros(n_exp)
            s[js] = 1.0
            feats = [len(js), len(js) / n_exp, float(s @ conf[i]),
                     float(conf[i][js].max()) if js else 0.0, int(len(js) == top and top),
                     len(c), int(ctx.tasks[i] == "numeric")] + list(s)
            if dist is not None:
                qid = ctx.df["qid"].to_numpy()[i]
                mass = np.array([dist.get(qid, {}).get(e, {}).get(k, 0.0)
                                 for e in ctx.expert_ids])
                feats += [float(mass.sum()), float(mass.max()),
                          float(mass.mean())] + list(mass)
            X.append(feats)
            y.append(is_correct(ctx.tasks[i], k, golds[i]))
            grp.append(i)
            cand.append(k)
    return (np.asarray(X, np.float32), np.asarray(y), np.asarray(grp),
            np.asarray(cand, dtype=object))


def run_more(ctx, A, tr, te, dist=None, n_samples=1):
    conf = np.nan_to_num(selfconf_matrix(ctx), nan=0.0)
    Xtr, ytr, _, _ = _more_rows(ctx, A, conf, tr, dist)
    Xte, yte, gte, _ = _more_rows(ctx, A, conf, te, dist)
    model = LogisticRegression(max_iter=2000, C=0.5).fit(Xtr, ytr)
    sc = model.predict_proba(Xte)[:, 1]
    correct = np.zeros(len(te), dtype=int)
    for n, i in enumerate(te):
        m = gte == i
        if m.any():
            correct[n] = yte[m][int(np.argmax(sc[m]))]
    cost = np.full(len(te), float(ctx.params_b.sum()), np.float32)
    if n_samples > 1:
        # See AggregatorRouter.predict: only the generative third re-runs the model.
        cost[[ctx.tasks[i] == "numeric" for i in te]] *= n_samples
    return correct, cost


# --------------------------------------------------------------------------- #
# LLM-TOPLA (arXiv:2410.03953)
# --------------------------------------------------------------------------- #
def _focal_diversity(fail, subset):
    """lambda^focal = mean over focal models of 1 - P(2)/P(1), where P(1) and
    P(2) are, among the questions the focal model got wrong, the chance that one
    and that two randomly drawn other members also fail.

    `fail` is the [n_train, n_experts] 0/1 failure matrix of the training half.
    """
    if len(subset) < 3:
        return 0.0
    vals = []
    for f in subset:
        others = [m for m in subset if m != f]
        fails = fail[fail[:, f] == 1][:, others]
        if not len(fails):
            continue
        n = len(others)
        p1 = float(fails.mean())
        k = fails.sum(axis=1)
        p2 = float(np.mean(k * (k - 1))) / (n * (n - 1)) if n > 1 else 0.0
        vals.append(1.0 - p2 / p1 if p1 > 0 else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def run_topla(ctx, A, tr, te, seed):
    """Focal-diversity subset selection, then the weighted combiner.

    TOPLA-Weighted is an MLP over the concatenated per-model answer distributions.
    The cache stores each expert's chosen letter and its probability, so the
    distribution is reconstructed as p(chosen) = confidence with the rest spread
    uniformly -- one forward pass per expert, no extra sampling. Only the
    multiple-choice two thirds has a fixed class space; GSM8K falls back to a plurality
    inside the selected subset. TOPLA-Summary is not runnable here: it is a
    seq2seq model over the candidates' text.
    """
    n_exp = len(ctx.expert_ids)
    acc_tr = ctx.Y[tr].mean(0)
    fail = (1 - ctx.Y[tr]).astype(np.int8)
    floor = float(np.median(acc_tr))
    best, best_div = tuple(range(n_exp)), -1.0
    for size in range(3, n_exp + 1):
        for sub in combinations(range(n_exp), size):
            if acc_tr[list(sub)].mean() < floor:
                continue
            div = _focal_diversity(fail, sub)
            if div > best_div:
                best, best_div = sub, div
    sub = list(best)
    conf = np.nan_to_num(selfconf_matrix(ctx), nan=0.25)

    def dists(rows):
        out = np.zeros((len(rows), len(sub) * 4), np.float32)
        for n, i in enumerate(rows):
            for m, j in enumerate(sub):
                p = np.full(4, (1.0 - conf[i, j]) / 3.0)
                if A[i, j] in LETTERS:
                    p[LETTERS.index(A[i, j])] = conf[i, j]
                out[n, 4 * m:4 * m + 4] = p
        return out

    mc_tr = [i for i in tr if ctx.tasks[i] == "mc"]
    mc_te = [i for i in te if ctx.tasks[i] == "mc"]
    golds = ctx.df["answer"].to_numpy()
    mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=800, random_state=seed)
    mlp.fit(dists(mc_tr), [golds[i] for i in mc_tr])
    pred_mc = dict(zip(mc_te, mlp.predict(dists(mc_te)))) if mc_te else {}

    def pick(i):
        if ctx.tasks[i] == "mc":
            return pred_mc.get(i)
        c = candidates([A[i, j] for j in sub])
        return max(c, key=lambda k: len(c[k])) if c else None
    correct, _ = score_choice(ctx, te, pick, lambda i: 0.0)
    cost = np.full(len(te), float(ctx.params_b[sub].sum()), np.float32)
    return correct, cost, sub, best_div


# --------------------------------------------------------------------------- #
def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.std(), (d.mean() / se if se > 0 else np.nan), int((d > 0).sum())


def main() -> None:
    cfg = load_config(ROOT / "config.yaml")
    cache = Path(cfg["paths"]["cache_dir"])
    df = pd.read_csv(cache / "questions.csv").fillna("")
    preds = pd.read_csv(cache / "predictions.csv")

    acc = defaultdict(list)
    cost = defaultdict(list)
    topla_pick = []
    ctx_ids = [e["id"] for e in cfg["experts"]]
    roll_dist, n_samples = rollout_distributions(cache, set(ctx_ids))
    for seed in SEEDS:
        d = df.copy()
        d["split"] = _resplit(d, cfg["dataset"]["train_frac"], seed)
        cfg.raw["seed"] = seed
        ctx = build_context(cfg, d, preds, cache)
        A = answers_matrix(ctx)
        tr = np.where(ctx.train_mask)[0]
        te = np.where(ctx.test_mask)[0]

        for name, fn in (("majority", run_majority), ("OW", run_ow), ("ISP", run_isp),
                         ("smoothie_global", run_smoothie_global),
                         ("smoothie_local", run_smoothie_local), ("MoRE", run_more)):
            c, k = fn(ctx, A, tr, te)
            acc[name].append(float(c.mean()))
            cost[name].append(float(k.mean()))
        c, k = run_more(ctx, A, tr, te, roll_dist, n_samples)
        acc["MoRE+rollouts"].append(float(c.mean()))
        cost["MoRE+rollouts"].append(float(k.mean()))
        c, k, sub, div = run_topla(ctx, A, tr, te, seed)
        acc["TOPLA"].append(float(c.mean()))
        cost["TOPLA"].append(float(k.mean()))
        topla_pick.append((tuple(ctx.expert_ids[j] for j in sub), div))

        for name in OURS:
            for tag, use_roll in ((("", False), ("+rollouts", True))
                                  if name == "aggregator" else ((("", False),))):
                if name == "aggregator":
                    cfg.raw["experiments"]["aggregator"]["use_rollouts"] = use_roll
                r = fit_router(name, cfg, ctx, seed).predict(ctx)
                acc[name + tag].append(float(r.correct.mean()))
                cost[name + tag].append(float(r.cost.mean()))
        acc["best_single"].append(float(ctx.Yte[:, int(np.argmax(ctx.Ytr.mean(0)))].mean()))
        acc["oracle"].append(float(ctx.Yte.max(axis=1).mean()))

    out_path = Path(cfg["paths"]["results_dir"]) / "aggregator_vs_literature.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"seeds": SEEDS, "accuracy": dict(acc),
                                    "params_b": dict(cost)}, indent=2))
    print(f"\n[results] per-split accuracies -> {out_path}")

    refs = ["aggregator", "aggregator+rollouts"]
    print("\n=== `aggregator` vs. the published post-hoc combiners, 10 splits ===")
    print("    (all re-implemented on this benchmark's cache; adaptations are")
    print("     documented in each function's docstring)")
    print(f"\n{'method':<24}{'labels':>7}{'params':>9}{'accuracy':>15}"
          f"{'vs aggregator':>15}{'t':>6}"
          f"{'vs +rollouts':>15}{'t':>6}")
    order = [("best_single", "-"), ("majority", "no"), ("ISP", "no"),
             ("smoothie_global", "no"), ("smoothie_local", "no"), ("OW", "yes"),
             ("MoRE", "yes"), ("MoRE+rollouts", "yes"), ("TOPLA", "yes"),
             ("ensemble", "yes"),
             ("soft_actor", "yes"), ("aggregator", "yes"),
             ("aggregator+rollouts", "yes"), ("oracle", "-")]
    for name, lab in order:
        a = 100 * np.asarray(acc[name])
        c = f"{np.mean(cost[name]):.2f}B" if name in cost else ""
        line = f"{name:<24}{lab:>7}{c:>9}{a.mean():>9.1f} +- {a.std():<4.1f}"
        for ref in refs:
            if name in (ref, "best_single", "oracle"):
                line += f"{'':>21}"
                continue
            m, _, t, w = paired(acc[ref], acc[name])
            line += f"{100*m:>+10.1f} {w:>2}/10{t:>+6.2f}"
        print(line)

    picks = defaultdict(int)
    for sub, _ in topla_pick:
        picks[sub] += 1
    top = max(picks.items(), key=lambda kv: kv[1])
    print(f"\n  TOPLA picked {len(top[0])} of {len(cfg['experts'])} experts on "
          f"{top[1]}/10 splits (focal diversity "
          f"{np.mean([d for _, d in topla_pick]):.3f}): {', '.join(top[0])}")
    print("\n  not runnable on this benchmark, and left out rather than approximated:")
    print("    LLM-Blender  PairRanker is a cross-encoder over candidate *text*; a")
    print("                 candidate here is one letter or one number.")
    print("    LENS         needs each model's layer-wise hidden states; the cache")
    print("                 stores answers and a scalar self-confidence.")


if __name__ == "__main__":
    main()
