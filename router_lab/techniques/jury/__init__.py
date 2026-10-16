"""A generative model of the testimony, fitted without labels.

`aggregator` scores each candidate answer discriminatively: it estimates
P(candidate is correct | features describing that candidate's supporters) and
returns the argmax. This router turns the arrow round and models how the answers
were *produced*:

    z_q                 the latent true answer to question q
    expert e            emits z_q with probability pi_e, and otherwise a wrong
                        answer drawn uniformly from the rest of the candidate set

so the verdict is the posterior argmax_v P(z = v | the whole configuration), and
`pi` is estimated by EM from the agreement pattern alone. Nothing here reads a
gold answer, at fit time or at predict time: this is Dawid-Skene (1979) over
answers rather than annotators, with one parameter per expert and no training
set at all -- the EM does see the training half's answer configurations, so what
is absent is the labels and not the fitting.

Why that matters here. `paper/scripts/headroom_ceiling.py` measures that the
scorer in `aggregator` is bounded by supervision rather than by model class --
52.6% in-sample against 41.4% held-out on the minority stratum, on 180 training
questions. A model that needs no labels is not subject to that bound, which is
the actual reason to prefer it, and it ports to a new benchmark without anyone
having to label one.

What it reduces to. With one greedy answer per expert the emission is one-hot and

    log P(a | z = v) = const + sum_{e : a_e = v} log( pi_e * m / (1 - pi_e) )

with m = |candidates| - 1: the verdict is *exactly* the Nitzan-Paroush (1982)
log-odds weighted vote, which is the `OW` baseline of
`paper/scripts/aggregator_vs_literature.py` with the accuracies estimated
unsupervised instead of read off the training labels. The two duly score the
same (70.6 +- 2.1 here against 69.9 +- 2.2 for OW). So on greedy answers this
router is a weighted vote, and the reachability observation of the paper's
Section 3.1 applies to it: its reachable set is the distinct argmax answers.

Where the gain comes from. `use_rollouts` replaces the one-hot emission with the
expert's empirical answer *distribution* over the repeated Fugu samples. The
likelihood consumes it as fractional evidence -- every candidate an expert put
mass on contributes, weighted by that mass -- and the rule stops being a weighted
vote over argmax answers, so it escapes the quotient. That is where the result
is: 76.4 +- 1.3 with no labels, against 73.8 +- 1.8 for supervised MoRE with the
same samples and 74.1 +- 2.3 for `aggregator` with the same samples.

A negative result kept in the code because it is reported in the paper. The
`lure` option adds a second latent per question -- a wrong answer that a group of
experts is jointly drawn to -- so that agreement inside the group is discounted
as shared error rather than counted as independent evidence. It does nothing. On
the rollout emissions EM drives every susceptibility to exactly zero and the
verdicts are identical to the no-lure model; on greedy emissions a `pool` lure
does settle on a non-zero susceptibility off 180 questions, but it is worth
+0.11 points (t = +0.80, 2 splits of 10), so it is fitted and then not used.
Clamping z with the training labels, which removes any question of identifying
the channel, does not revive it either.

The reason is measured directly in `paper/scripts/testimony_structure.py`: pairs
from the same pre-training lineage agree on their errors *less* than
cross-lineage pairs (+0.012 against +0.035 excess over chance). Shared
pre-training is not what makes two small models fail together, which is also why
the family-diversity feature in `aggregator` buys nothing. Default is `none`.

Two controls, both reported in the paper because a reviewer is right to ask for
them. `collapse_modes` keeps the eight samples and reduces each expert to the one
answer it sampled most often, which costs exactly what the distributional variant
costs and scores 70.2 +- 1.3 against 70.3 +- 2.1 for a single greedy answer: the
extra samples, on their own, are worth nothing, and the +6.2 is the distribution.
`per_task` fits one reliability per expert *and task* instead of one overall,
label-free as ever, since a question's task is part of the query and not of its
answer. It reaches 77.2 +- 1.2 against 76.4 +- 1.4 for the same parameterisation
fitted on labels, so the global-weight restriction is not what the comparison
rests on -- though the margin does narrow, and the paper says so.

Cost is the pool's, as for any rule that hears every witness. `use_rollouts`
costs extra only where the samples are really re-generated -- on GSM8K; on
multiple choice `Expert.sample_batch` draws them from the A/B/C/D distribution of
a single forward pass, so there the soft emission is free and is in fact a
quantised copy of a distribution `Expert._score_mc` currently discards.
"""
from __future__ import annotations

import numpy as np

from ...data import LETTERS, is_correct
from ..aggregator import (FAMILY, answer_key, collapse_to_modes,
                          rollout_distributions)
from ..base import BaseRouter, RoutingContext, RoutingResult, prediction_matrix

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Which experts, if any, share a lure
# --------------------------------------------------------------------------- #
def lure_groups(expert_ids, mode: str):
    """Group index per expert; -1 means the expert has no lure.

    `none`   no lure anywhere -- the model of record.
    `family` one group per pre-training lineage. A lineage with a single member
             is given -1 rather than a group of its own: a family of one shares
             nothing, and leaving it a free lure lets that expert explain away
             whatever it says as its own family's error, which silently
             neutralises its dissent against the largest bloc.
    `pool`   every expert in one group: the question itself has an attractive
             wrong answer that part of the pool falls for.
    """
    n = len(expert_ids)
    if mode == "none":
        return np.full(n, -1), []
    if mode == "pool":
        return np.zeros(n, dtype=int), ["pool"]
    if mode != "family":
        raise ValueError(f"unknown lure mode {mode!r}; expected none, family or pool")
    lab = [FAMILY.get(e, e) for e in expert_ids]
    keep = [f for f in sorted(set(lab)) if lab.count(f) >= 2]
    return np.array([keep.index(f) if f in keep else -1 for f in lab]), keep


# --------------------------------------------------------------------------- #
# One question's testimony
# --------------------------------------------------------------------------- #
class Testimony:
    """The candidate set of a question plus one emission distribution per expert.

    `emit[e]` sums to 1 for an expert that answered and to 0 for one that did
    not, so a missing answer contributes nothing to the likelihood instead of
    being scored as a wrong one. Greedy answers give a one-hot row; rollouts give
    the empirical distribution over the samples. Both go through the same
    likelihood, so the two variants of this router differ in the evidence and not
    in the model.
    """

    __slots__ = ("cands", "emit", "group", "qid", "plur")

    def __init__(self, cands, emit, group, qid, plur=0):
        self.cands = cands
        self.emit = emit
        self.group = group
        self.qid = qid
        self.plur = plur        # index of the greedy plurality answer, for the
                                # selective-override sweep


def build_testimony(ctx: RoutingContext, answers, group, dist=None, fixed_mc=False):
    """One `Testimony` per question of the context, in context row order.

    The candidate set is the answers the pool actually produced, so it is a
    support read off the data. With `fixed_mc` a multiple-choice question keeps
    the task's four options as its support whether or not every option was
    proposed: m = 3 always, an unproposed option is a candidate with zero
    emission mass, and a unanimous question is no longer a singleton, so it
    enters EM. The paper reports this as a control on the support construction.
    """
    expert_ids = list(ctx.expert_ids)
    qids = ctx.df["qid"].to_numpy()
    out = []
    for i, qid in enumerate(qids):
        keys = [a for a in answers[i] if a is not None]
        if dist is not None:
            for e in expert_ids:
                keys += list(dist.get(qid, {}).get(e, {}))
        if fixed_mc and ctx.tasks[i] == "mc":
            keys = [k for k in keys if k in LETTERS] + list(LETTERS)
        cands = sorted(set(keys))
        if not cands:
            out.append(None)
            continue
        idx = {c: j for j, c in enumerate(cands)}
        emit = np.zeros((len(expert_ids), len(cands)))
        for j, e in enumerate(expert_ids):
            if dist is not None:
                for k, v in dist.get(qid, {}).get(e, {}).items():
                    if k in idx:
                        emit[j, idx[k]] += v
            if emit[j].sum() <= 0 and answers[i, j] in idx:
                emit[j, idx[answers[i, j]]] = 1.0   # no samples cached: use the greedy answer
            s = emit[j].sum()
            if s > 0:
                emit[j] /= s
        votes = np.zeros(len(cands))
        for a in answers[i]:
            if a in idx:
                votes[idx[a]] += 1
        out.append(Testimony(cands, emit, group, qid,
                             int(np.argmax(votes)) if votes.max() > 0 else 0))
    return out


# --------------------------------------------------------------------------- #
# Likelihood and posterior
# --------------------------------------------------------------------------- #
def _susceptibility(t: Testimony, lam):
    """Per-expert lure susceptibility, zeroed for experts with no lure group.

    lambda = 0 makes the lure channel inert: the probability of emitting the lure
    collapses to the uniform wrong-answer probability, so a no-lure expert is an
    independent Dawid-Skene expert without needing a separate code path.
    """
    out = np.zeros(len(t.group))
    sel = t.group >= 0
    if sel.any() and len(lam):
        out[sel] = lam[sel]
    return out


def _log_emission(t: Testimony, pi, lam_e, alpha=1.0):
    """log P(a_e | z = v, lure = l) as [n_experts, n_cands, n_cands] -> [e, v, l].

    Three channels: the truth (pi_e), the group's shared lure ((1-pi_e) lambda_e)
    and a uniform wrong answer (whatever is left), each accumulating the expert's
    emission mass that falls on it.
    """
    n_e, n_c = t.emit.shape
    m = max(n_c - 1, 1)
    log_truth = np.log(pi + EPS)[:, None]
    log_lure = np.log((1 - pi) * (lam_e + (1 - lam_e) / m) + EPS)[:, None]
    log_other = np.log((1 - pi) * (1 - lam_e) / m + EPS)[:, None]

    w = t.emit
    tot = w.sum(1)
    log = np.empty((n_e, n_c, n_c))
    for v in range(n_c):
        other = (tot - w[:, v])[:, None] - w        # mass on neither the truth nor the lure
        log[:, v, :] = alpha * (w[:, v][:, None] * log_truth + w * log_lure
                                + other * log_other)
    return log


def posterior(t: Testimony, pi, lam, n_groups: int, alpha=1.0):
    """(gamma, delta): P(z = v | a) and, per group, P(lure = l | z = v).

    Given z the groups' lures are conditionally independent, so the sum over
    lures factors across groups and costs |candidates| terms each -- exact, and
    cheap enough that nothing here is approximated.
    """
    n_c = t.emit.shape[1]
    if n_c == 1:
        return np.ones(1), np.zeros((max(n_groups, 1), 1, 1))

    log_e = _log_emission(t, pi, _susceptibility(t, lam), alpha)
    mask = ~np.eye(n_c, dtype=bool)
    log_prior_lure = -np.log(n_c - 1)

    log_post = np.zeros(n_c)
    delta = np.zeros((max(n_groups, 1), n_c, n_c))

    free = t.group < 0
    if free.any():
        # No lure to sum over: the likelihood does not depend on l, so any column
        # is the value.
        log_post += log_e[free][:, :, 0].sum(0)

    for g in range(n_groups):
        sel = t.group == g
        if not sel.any():
            continue
        lf = np.where(mask, log_e[sel].sum(0) + log_prior_lure, -np.inf)
        mx = lf.max(axis=1, keepdims=True)
        ex = np.exp(lf - mx)
        log_post += mx.squeeze(1) + np.log(ex.sum(1) + EPS)
        delta[g] = ex / (ex.sum(1, keepdims=True) + EPS)

    log_post -= log_post.max()
    gamma = np.exp(log_post)
    return gamma / (gamma.sum() + EPS), delta


# --------------------------------------------------------------------------- #
# EM
# --------------------------------------------------------------------------- #
def fit_em(testimony, n_groups, n_exp, max_iter=200, tol=1e-7, alpha=1.0):
    """Estimate pi (and lambda, when a lure is enabled) with no labels.

    The pi update is the Dawid-Skene one and comes out unusually clean here: the
    uniform wrong-answer channel never emits z, so an expert that said v can only
    have reached it through the truth channel, and its responsibility for having
    told the truth is exactly the posterior mass on what it said.
    """
    pi = np.full(n_exp, 0.6)
    lam = np.full(n_exp, 0.3) if n_groups else np.zeros(0)
    prev = None

    for _ in range(max_iter):
        num_pi, den_pi = np.zeros(n_exp), np.zeros(n_exp)
        num_lam, den_lam = np.zeros(n_exp), np.zeros(n_exp)

        for t in testimony:
            if t is None or len(t.cands) == 1:
                continue
            gamma, delta = posterior(t, pi, lam, n_groups, alpha)
            answered = t.emit.sum(1) > 0
            truth_mass = t.emit @ gamma
            num_pi += np.where(answered, truth_mass, 0.0)
            den_pi += answered.astype(float)

            if not n_groups:
                continue
            m = max(len(t.cands) - 1, 1)
            lam_e = _susceptibility(t, lam)
            share = lam_e / (lam_e + (1 - lam_e) / m + EPS)
            for g in range(n_groups):
                sel = (t.group == g) & answered
                if not sel.any():
                    continue
                lure_mass = np.einsum("v,vl,el->e", gamma, delta[g], t.emit[sel])
                num_lam[sel] += lure_mass * share[sel]
                den_lam[sel] += 1.0 - truth_mass[sel]

        pi = np.clip(num_pi / np.maximum(den_pi, EPS), 0.02, 0.98)
        if n_groups:
            lam = np.clip(num_lam / np.maximum(den_lam, EPS), 0.0, 0.98)
        cur = np.concatenate([pi, lam])
        if prev is not None and np.abs(cur - prev).max() < tol:
            break
        prev = cur
    return pi, lam


# --------------------------------------------------------------------------- #
class JuryRouter(BaseRouter):
    name = "jury"
    kind = "combination"

    # ----------------------------------------------------------------- setup --
    def _setup(self, ctx: RoutingContext):
        cfg = ctx.cfg.get("experiments", {}).get("jury", {}) \
            if hasattr(ctx.cfg, "get") else {}
        self.lure = str(cfg.get("lure", "none"))
        self.max_iter = int(cfg.get("max_iter", 200))
        self.tol = float(cfg.get("tol", 1e-7))
        self.use_rollouts = bool(cfg.get("use_rollouts", False))
        # Two controls, both off by default and both reported in the paper:
        # `collapse_modes` pays for the samples and then discards the
        # distribution, and `per_task` lets one reliability per expert become one
        # per expert and task.
        self.collapse_modes = bool(cfg.get("collapse_modes", False))
        self.per_task = bool(cfg.get("per_task", False))
        # `observed` (the default) reads the support off the pool's answers;
        # `task` fixes it to the four options on multiple choice.
        self.support = str(cfg.get("support", "observed"))
        if self.support not in ("observed", "task"):
            raise ValueError(f"unknown support {self.support!r}; expected observed or task")

        self.dist, self.n_samples = ({}, 1)
        if self.use_rollouts:
            self.dist, self.n_samples = rollout_distributions(
                ctx.cfg["paths"]["cache_dir"], set(ctx.expert_ids))
            if not self.dist:
                print("[jury] no rollout cache -- falling back to greedy answers")
                self.use_rollouts = False
            elif self.collapse_modes:
                self.dist = collapse_to_modes(self.dist)
        # The evidence weight. "n" (the default) is the multinomial likelihood of
        # the n samples behind each expert's distribution, so alpha = n with the
        # rollouts and alpha = 1 on a single greedy answer. A number overrides
        # it; alpha = 1 with the rollouts is the fractional "one witness, one
        # testimony" likelihood, which the paper keeps as an ablation because on
        # pools of three experts it admits a degenerate maximum (Appendix B.5).
        alpha = cfg.get("alpha", "n")
        self.alpha = float(self.n_samples if self.use_rollouts else 1) \
            if isinstance(alpha, str) and alpha.strip().lower() in ("n", "multinomial") \
            else float(alpha)

        tasks = np.repeat(ctx.tasks[:, None], len(ctx.expert_ids), axis=1)
        self._ans = np.vectorize(answer_key, otypes=[object])(tasks, prediction_matrix(ctx))
        group, self.groups = lure_groups(ctx.expert_ids, self.lure)
        self._t = build_testimony(ctx, self._ans, group,
                                  self.dist if self.use_rollouts else None,
                                  fixed_mc=self.support == "task")
        self._row_of = {q: i for i, q in enumerate(ctx.df["qid"].to_numpy())}
        self._task_of = dict(zip(ctx.df["qid"].to_numpy(), ctx.tasks))

    # ------------------------------------------------------------------ fit ---
    def fit(self, ctx: RoutingContext):
        """EM over the training questions' configurations. No labels are read.

        Fitting on the training half only, rather than on every cached question,
        is not required by the model -- it has no labels to leak, so the whole
        cache would be legitimate and scores the same (76.4 +- 1.3 against
        76.4 +- 1.6) -- but it keeps this router on exactly the footing the rest
        of the benchmark is evaluated on.
        """
        self._setup(ctx)
        tr = [i for i in np.where(ctx.train_mask)[0] if self._t[i] is not None]
        rows = [self._t[i] for i in tr]
        self.pi, self.lam = fit_em(rows, len(self.groups), len(ctx.expert_ids),
                                   self.max_iter, self.tol, self.alpha)
        self.pi_task = {}
        if self.per_task:
            # One EM per task, on that task's training configurations only. Still
            # label-free: the task of a question is part of the query, not of its
            # answer, and every rule in the comparison may read it.
            for t in sorted(set(ctx.tasks[tr])):
                sub = [self._t[i] for i in tr if ctx.tasks[i] == t]
                self.pi_task[t] = fit_em(sub, len(self.groups), len(ctx.expert_ids),
                                         self.max_iter, self.tol, self.alpha)[0]
        print(f"[jury] EM on {len(rows)} unlabelled questions "
              f"({len(ctx.expert_ids)} pi"
              + (f" x {len(self.pi_task)} tasks" if self.per_task else "")
              + (f" + {len(ctx.expert_ids)} lambda/{self.lure}" if self.groups else "")
              + (", fixed four-option support" if self.support == "task" else "")
              + f", {'mode' if self.collapse_modes else 'rollout' if self.use_rollouts else 'greedy'}"
              " emissions)")
        return self

    # ---------------------------------------------------------------- verdict --
    def deliberate(self, qid):
        """(candidates, posterior, index of the greedy plurality) for one question.

        The public handle on the posterior, so a caller can sweep an override
        threshold or read a calibrated confidence without reaching into the
        fitted state. Returns None when nobody answered the question.
        """
        t = self._t[self._row_of[qid]]
        if t is None:
            return None
        pi = self.pi_task.get(self._task_of[qid], self.pi) if self.per_task else self.pi
        gamma, _ = posterior(t, pi, self.lam, len(self.groups), self.alpha)
        return t.cands, gamma, t.plur

    def verdict(self, qid, tau=0.0):
        """The jury's answer, or None when nobody answered.

        With `tau > 0` the jury must beat the plurality by that much posterior
        mass before it may overturn it. The paper reports the sweep: it is
        monotone, so the best threshold is no threshold.
        """
        out = self.deliberate(qid)
        if out is None:
            return None
        cands, gamma, plur = out
        pick = int(np.argmax(gamma))
        if tau > 0 and pick != plur and gamma[pick] - gamma[plur] < tau:
            pick = plur
        return cands[pick]

    # -------------------------------------------------------------- predict ---
    def predict(self, ctx: RoutingContext) -> RoutingResult:
        test_qids = ctx.df.loc[ctx.test_mask, "qid"].to_numpy()
        golds = ctx.df.loc[ctx.test_mask, "answer"].to_numpy()
        tasks = ctx.tasks[ctx.test_mask]

        correct = np.zeros(len(test_qids), dtype=int)
        for n, qid in enumerate(test_qids):
            pick = self.verdict(qid)
            if pick is not None:
                correct[n] = is_correct(tasks[n], pick, golds[n])

        cost = np.full(len(test_qids), float(ctx.params_b.sum()), dtype=np.float32)
        if self.use_rollouts:
            # Only the generative third really re-runs the models; see the module
            # docstring and `AggregatorRouter.predict`.
            cost[np.array([t == "numeric" for t in tasks])] *= self.n_samples
        note = ("label-free generative answer selection"
                + (f" ({self.n_samples} samples/expert)" if self.use_rollouts else ""))
        return RoutingResult(self.name, correct, cost, None, note=note)
