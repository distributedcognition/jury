"""Post-hoc aggregation over the agreement pattern (learned answer selection).

`ensemble` fuses the experts' answers with weights it fixes in advance (each
expert's regional success rate). This router asks a different question: given
that the pool has *already* answered and we can see exactly who said what, which
of the answers on the table is the correct one?

The unit of prediction is therefore not an expert but a **candidate answer**. For
each question the distinct answers form the candidate set, and each candidate is
described by the testimony behind it:

  * how many experts support it, and which ones (one indicator per expert);
  * how reliable each supporter is in this region of the embedding space
    (`knn_regional_accuracy`), and how confident it says it is (`selfconf_matrix`);
  * how many distinct model *families* back it -- agreement between two Qwen2.5
    checkpoints is weaker evidence than agreement between Qwen and OLMo, because
    shared pre-training data makes their errors correlated;
  * optionally, how much sampled probability mass it holds across the repeated
    Fugu rollouts (`use_rollouts`).

A logistic regression scores every candidate and the argmax is returned. The gold
answer is the training label; it is never an input feature.

Why this shape. `paper/scripts/headroom_ceiling.py` measures that 29.2% of the
questions are *minority-correct* -- the gold answer is on the table but is not the
plurality -- and that no weighting of a plurality rule can reach them, because a
weighted vote over discrete answers is quotiented onto the set of distinct
answers however the weights move. Scoring candidates directly is the smallest
model class that escapes that quotient. It lifts the minority-correct stratum
from 31.3% to 41.4% (greedy) or 50.5% (with rollouts).

Cost is honest: like `ensemble` this runs every expert, so a question costs the
whole pool. `use_rollouts` costs extra only where the samples are really
re-generated -- on GSM8K. On multiple choice `Expert.sample_batch` draws the
samples from the A/B/C/D distribution of a single forward pass, so there the mass
features are free, and are in fact a quantised copy of a distribution
`Expert._score_mc` currently throws away.

Prior art, and what this router is NOT. Selecting among candidate answers with a
learned scorer is MoRE (Si et al., 2023), which uses inter-expert agreement and
the experts' own confidence. `paper/scripts/aggregator_vs_literature.py` runs
MoRE on the same ten splits and the answer is unambiguous:

    MoRE                  72.1 +- 2.6 %   14.45B
    aggregator            71.8 +- 2.5 %   14.45B  (-0.3, t=-0.60, 2/10 splits)
    MoRE + rollouts       73.8 +- 1.8 %   48.17B
    aggregator + rollouts 74.1 +- 2.3 %   48.17B  (+0.3, t=+0.77, 5/10 splits)

So the two extra feature groups this router adds over MoRE -- the regional
reliability prior and the family diversity of the supporting coalition -- buy
nothing measurable at either cost point. Everything it gains over `ensemble`
(+1.3 greedy, +3.6 with rollouts) comes from the 2023 idea of scoring candidate
answers instead of weighting experts, plus repeated sampling. It is kept in the
benchmark as the implemented reference point for that whole family, not as a
contribution: treat it as MoRE with a wider feature set, and cite MoRE.

Two neighbours cannot be run here at all and are excluded rather than
approximated: LLM-Blender's PairRanker is a cross-encoder over candidate *text*
(a candidate here is one letter or one number) and LENS needs layer-wise hidden
states, which the cache does not store.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ...data import _clean_num, is_correct
from ..base import (BaseRouter, RoutingContext, RoutingResult,
                    knn_regional_accuracy, prediction_matrix, selfconf_matrix)

# Base-corpus lineage, not parameter count: shared pre-training data is what makes
# two experts fail on the same question, so it is what makes their agreement cheap.
FAMILY = {"qwen0.5b": "qwen2.5", "qwen1.5b": "qwen2.5", "qwen3b": "qwen2.5",
          "qwen-math": "qwen2.5", "qwen-coder1.5b": "qwen2.5",
          "qwen3-0.6b": "qwen3", "olmo2-1b": "olmo2",
          "eurollm1.7b": "eurollm", "granite2b": "granite"}


def answer_key(task: str, ans) -> str | None:
    """Canonical key: two answers share a key iff they are the same answer.

    Mirrors `is_correct` -- exact string match for multiple choice, numeric value
    for GSM8K, rounded at the 1e-4 scale that comparison uses.
    """
    if ans is None or (isinstance(ans, float) and np.isnan(ans)):
        return None
    if task == "numeric":
        try:
            return f"{round(float(_clean_num(str(ans))), 4):.4f}"
        except (ValueError, TypeError):
            return None
    s = str(ans).strip()
    return s or None


def rollout_distributions(cache_dir, expert_ids):
    """qid -> expert -> {answer key: probability} from the repeated Fugu samples.

    Returns ({}, 1) when the rollout cache is absent, which silently degrades the
    router to its greedy-only variant rather than failing on an older cache.
    """
    path = Path(cache_dir) / "fugu_rollouts.csv"
    if not path.exists():
        return {}, 1
    roll = pd.read_csv(path)
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    n = defaultdict(lambda: defaultdict(int))
    for r in roll.itertuples():
        if r.expert_id not in expert_ids:
            continue
        n[r.expert_id][r.qid] += 1
        key = answer_key(r.task, r.pred)
        if key is not None:
            counts[r.qid][r.expert_id][key] += 1.0
    dist = {q: {e: {k: c / max(n[e][q], 1) for k, c in ks.items()}
                for e, ks in per_expert.items()}
            for q, per_expert in counts.items()}
    per_q = [v for e in n.values() for v in e.values()]
    return dist, int(np.median(per_q)) if per_q else 1


def collapse_to_modes(dist):
    """Same samples, each expert reduced to the one answer it sampled most often.

    The sample-matched control for the distributional gain. A rule given these
    point masses has paid for the repeated sampling in full but has thrown the
    distribution away, so its distance from the same rule on `dist` is what
    keeping the distribution is worth, and its distance from the same rule on
    greedy answers is what the extra samples are worth on their own. Without it
    the two effects are confounded, since the distributional rules are also the
    ones that re-run the models.

    Ties are broken by sort order, which is deterministic and, on a tie, as
    arbitrary as the mode itself.
    """
    out = {}
    for q, per_expert in dist.items():
        row = {}
        for e, ks in per_expert.items():
            if ks:
                row[e] = {max(sorted(ks), key=lambda k: ks[k]): 1.0}
        out[q] = row
    return out


class AggregatorRouter(BaseRouter):
    name = "aggregator"
    kind = "combination"

    # ----------------------------------------------------------------- setup --
    def _setup(self, ctx: RoutingContext):
        cfg = ctx.cfg.get("experiments", {}).get("aggregator", {}) \
            if hasattr(ctx.cfg, "get") else {}
        self.k = int(cfg.get("k", 15))
        self.C = float(cfg.get("C", 0.5))
        self.use_rollouts = bool(cfg.get("use_rollouts", False))
        self.dist, self.n_samples = ({}, 1)
        if self.use_rollouts:
            self.dist, self.n_samples = rollout_distributions(
                ctx.cfg["paths"]["cache_dir"], set(ctx.expert_ids))
            if not self.dist:
                print("[aggregator] no rollout cache -- falling back to greedy answers")
                self.use_rollouts = False
        tasks = np.repeat(ctx.tasks[:, None], len(ctx.expert_ids), axis=1)
        self._ans = np.vectorize(answer_key, otypes=[object])(tasks,
                                                              prediction_matrix(ctx))
        self._conf = np.nan_to_num(selfconf_matrix(ctx), nan=0.0)
        # Regional reliability for *every* question, fitted on the training half
        # only: it is a feature of the testimony, not of the answer.
        self._reg = knn_regional_accuracy(ctx.Xtr, ctx.Ytr, ctx.emb, self.k)
        self._row_of = {q: i for i, q in enumerate(ctx.df["qid"].to_numpy())}
        self._gold = ctx.df["answer"].to_numpy()

    # ------------------------------------------------------------- features ---
    def candidate_rows(self, ctx: RoutingContext, qids):
        """One row per (question, candidate answer).

        Returns (X, y, question id per row, candidate key per row). `y` is only
        meaningful on training questions; `predict` ignores it.
        """
        n_exp = len(ctx.expert_ids)
        X, y, groups, cands = [], [], [], []
        for qid in qids:
            i = self._row_of[qid]
            task, gold = ctx.tasks[i], self._gold[i]
            support: dict[str, list[int]] = {}
            for j in range(n_exp):
                if self._ans[i, j] is not None:
                    support.setdefault(self._ans[i, j], []).append(j)
            if self.use_rollouts:
                for e in ctx.expert_ids:
                    for key in self.dist.get(qid, {}).get(e, {}):
                        support.setdefault(key, support.get(key, []))
            if not support:
                continue
            top = max(len(v) for v in support.values())
            reg, conf = self._reg[i], self._conf[i]
            for key, js in support.items():
                s = np.zeros(n_exp)
                s[js] = 1.0
                mass = (np.array([self.dist.get(qid, {}).get(e, {}).get(key, 0.0)
                                  for e in ctx.expert_ids])
                        if self.use_rollouts else np.zeros(n_exp))
                feats = [
                    len(js), len(js) / n_exp,                  # raw and relative support
                    float(s @ reg), float(s @ conf),           # trust- and confidence-weighted
                    len({FAMILY.get(ctx.expert_ids[j], ctx.expert_ids[j]) for j in js}),
                    float(reg[js].max()) if js else 0.0,
                    float(reg[js].mean()) if js else 0.0,
                    int(len(js) == top and top > 0),           # is this the plurality?
                    len(support),                              # how split is the pool
                    int(task == "numeric"),
                    float(reg.max()), float(reg.mean()),       # how good the pool is here
                ]
                feats += list(s) + list(s * reg) + list(s * conf)
                if self.use_rollouts:
                    feats += [float(mass.sum()), float(mass.max()),
                              float(mass.mean())] + list(mass)
                X.append(feats)
                y.append(is_correct(task, key, gold))
                groups.append(qid)
                cands.append(key)
        return (np.asarray(X, np.float32), np.asarray(y),
                np.asarray(groups), np.asarray(cands, dtype=object))

    # ------------------------------------------------------------------ fit ---
    def fit(self, ctx: RoutingContext):
        from sklearn.linear_model import LogisticRegression

        self._setup(ctx)
        train_qids = ctx.df.loc[ctx.train_mask, "qid"].to_numpy()
        X, y, _, _ = self.candidate_rows(ctx, train_qids)
        self.model = LogisticRegression(max_iter=2000, C=self.C).fit(X, y)
        print(f"[aggregator] fitted on {len(X)} candidates from {len(train_qids)} "
              f"questions ({X.shape[1]} features, "
              f"{'with' if self.use_rollouts else 'no'} rollouts)")
        return self

    # -------------------------------------------------------------- predict ---
    def predict(self, ctx: RoutingContext) -> RoutingResult:
        test_qids = ctx.df.loc[ctx.test_mask, "qid"].to_numpy()
        X, y, groups, _ = self.candidate_rows(ctx, test_qids)
        scores = self.model.predict_proba(X)[:, 1]

        correct = np.zeros(len(test_qids), dtype=int)
        for i, qid in enumerate(test_qids):
            m = groups == qid
            if m.any():
                correct[i] = y[m][int(np.argmax(scores[m]))]
        cost = np.full(len(test_qids), float(ctx.params_b.sum()), dtype=np.float32)
        if self.use_rollouts:
            # Repeated sampling is not uniformly n_samples times the work. For
            # multiple choice the samples are multinomial draws from the A/B/C/D
            # distribution that ONE forward pass already produced
            # (`Expert.sample_batch`), so the mass features are free; only the
            # generative numeric half actually runs the model n_samples times.
            numeric = np.array([t == "numeric" for t in ctx.tasks[ctx.test_mask]])
            cost[numeric] *= self.n_samples
        note = ("learned answer selection over the agreement pattern"
                + (f" ({self.n_samples} samples/expert)" if self.use_rollouts else ""))
        return RoutingResult(self.name, correct, cost, None, note=note)
