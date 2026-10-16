"""Checks for the jury and the supervised answer scorer. Plain python, no pytest:

    python tests/test_jury.py            # everything
    python tests/test_jury.py jury       # only checks whose name matches

Everything runs against the committed `cache/`: no LLM, no GPU, no network.
A run takes about a minute.

These are the assertions the paper points to: the jury reads no gold answer during
fitting (per-task reliabilities included) and neither do OW-L and OW-I, its greedy rule is exactly the log-odds
weighted vote of Proposition 1, the distributional M-step is the equation in the
paper, the shared-error channel never raises accuracy, and collapsing the samples to
their mode keeps every sample and discards only the distribution.
"""
from __future__ import annotations

import copy
import dataclasses
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router_lab.techniques import REGISTRY                     # noqa: E402
from router_lab.techniques.aggregator import AggregatorRouter  # noqa: E402
from router_lab.techniques.base import build_context           # noqa: E402
from router_lab.techniques.aggregator import (collapse_to_modes,  # noqa: E402
                                              rollout_distributions)
from router_lab.techniques.jury import (JuryRouter, lure_groups, # noqa: E402
                                        posterior)
from router_lab.utils import load_config, set_seed             # noqa: E402

SLOW = set()
ESCALATING = set()

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _ctx():
    cfg = load_config(str(Path(__file__).resolve().parents[1] / "config.yaml"))
    cache = Path(cfg["paths"]["cache_dir"])
    df = pd.read_csv(cache / "questions.csv").fillna("")
    preds = pd.read_csv(cache / "predictions.csv")
    return cfg, build_context(cfg, df, preds, cache)


def _with_Y(ctx, Y):
    """A shallow copy of the context with the correctness matrix replaced."""
    c = copy.copy(ctx)
    c.Y = Y
    return c


# --------------------------------------------------------------------------- #
# contract: every registered router
# --------------------------------------------------------------------------- #
@check
def cached_embeddings_need_no_question_text(cfg, ctx, slow):
    from tempfile import TemporaryDirectory
    from router_lab.features import compute_embeddings

    without_text = ctx.df.copy()
    without_text[["question", "A", "B", "C", "D"]] = ""
    cached = compute_embeddings(cfg, without_text, Path(cfg["paths"]["cache_dir"]))
    assert np.array_equal(cached, ctx.emb)
    with TemporaryDirectory() as directory:
        try:
            compute_embeddings(cfg, without_text, Path(directory))
        except ValueError as error:
            assert "Question text is omitted" in str(error)
        else:
            raise AssertionError("Missing text must prevent uncached embedding generation")


@check
def registry_contract(cfg, ctx, slow):
    n_test = int(ctx.test_mask.sum())
    for name, cls in sorted(REGISTRY.items()):
        if name in SLOW and not slow:
            continue
        set_seed(cfg["seed"])
        res = cls(cfg).fit(ctx).predict(ctx)
        assert res.name == name, f"{name}: RoutingResult.name is {res.name!r}"
        assert res.correct.shape == (n_test,), f"{name}: correct has shape {res.correct.shape}"
        assert res.cost.shape == (n_test,), f"{name}: cost has shape {res.cost.shape}"
        assert set(np.unique(res.correct)) <= {0, 1, True, False}, f"{name}: correct not binary"
        assert np.all(res.cost > 0), f"{name}: non-positive cost"
        if cls.kind == "selection":
            assert res.chosen is not None, f"{name}: selection router reports no choice"
            assert res.chosen.min() >= 0 and res.chosen.max() < len(ctx.expert_ids), \
                f"{name}: chosen index out of range"
            assert np.array_equal(res.correct, ctx.Yte[np.arange(n_test), res.chosen]), \
                f"{name}: correctness does not match the chosen expert's label"
            # You always pay at least for the expert whose answer you keep.
            assert np.all(res.cost >= ctx.params_b[res.chosen] - 1e-6), \
                f"{name}: cost is below the chosen expert's size"
            if name not in ESCALATING:
                # A single-shot router runs exactly one expert, so its cost per
                # query IS that expert's size -- never a sum, never a blend.
                assert np.allclose(res.cost, ctx.params_b[res.chosen]), \
                    f"{name}: cost does not match the chosen expert's size"


@check
def registry_ignores_test_labels(cfg, ctx, slow):
    """Fitting must not read the held-out labels. Flip every test label and the
    routing decisions have to come out bit-identical -- only the *score* changes."""
    for name, cls in sorted(REGISTRY.items()):
        if name in SLOW and not slow or REGISTRY[name].kind != "selection":
            continue
        set_seed(cfg["seed"])
        a = cls(cfg).fit(ctx).predict(ctx)
        flipped = ctx.Y.copy()
        flipped[ctx.test_mask] = 1 - flipped[ctx.test_mask]
        set_seed(cfg["seed"])
        b = cls(cfg).fit(_with_Y(ctx, flipped)).predict(_with_Y(ctx, flipped))
        assert np.array_equal(a.chosen, b.chosen), \
            f"{name}: routing changed when test labels were flipped -> test leakage"


@check
def aggregator_ignores_test_labels(cfg, ctx, slow):
    """`registry_ignores_test_labels` only covers selection routers, because it
    compares `chosen`, which a combination router does not report. `aggregator`
    still has to satisfy the same property, and it scores candidates rather than
    experts, so the perturbation has to be on the gold *answers*: scramble the
    held-out ones and every candidate score must come out bit-identical.
    """
    set_seed(cfg["seed"])
    router = AggregatorRouter(cfg).fit(ctx)
    test_qids = ctx.df.loc[ctx.test_mask, "qid"].to_numpy()
    X_a, _, _, _ = router.candidate_rows(ctx, test_qids)
    score_a = router.model.predict_proba(X_a)[:, 1]

    scrambled = ctx.df.copy()
    rng = np.random.default_rng(0)
    held = scrambled.index[ctx.test_mask]
    scrambled.loc[held, "answer"] = rng.permutation(scrambled.loc[held, "answer"].to_numpy())
    ctx_b = dataclasses.replace(ctx, df=scrambled)
    set_seed(cfg["seed"])
    router_b = AggregatorRouter(cfg).fit(ctx_b)
    X_b, _, _, _ = router_b.candidate_rows(ctx_b, test_qids)
    score_b = router_b.model.predict_proba(X_b)[:, 1]

    assert np.array_equal(X_a, X_b), "aggregator features depend on the test answers"
    assert np.allclose(score_a, score_b), \
        "aggregator scoring changed when the held-out answers were scrambled -> test leakage"


# --------------------------------------------------------------------------- #
# jury: the label-free generative combiner
# --------------------------------------------------------------------------- #
def _jury(ctx, **over):
    c = copy.copy(ctx)
    c.cfg = copy.deepcopy(ctx.cfg)
    c.cfg.raw.setdefault("experiments", {}).setdefault("jury", {}).update(over)
    return c


@check
def ow_estimators_read_no_labels(cfg, ctx, slow):
    """OW-L and OW-I (Ai et al.), the two published label-free counterparts of the
    jury's reliability estimate, are compared with it in the appendix on the
    multiple-choice questions. The comparison is only fair if they are as blind
    as the jury: scrambling every gold answer must leave their estimated
    accuracies and their verdicts unchanged.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from aggregator_vs_literature import (_ow_pick, answers_matrix,   # noqa: E402
                                          run_ow_i, run_ow_l)
    A = answers_matrix(ctx)
    tr, te = np.where(ctx.train_mask)[0], np.where(ctx.test_mask)[0]
    scrambled = ctx.df.copy()
    scrambled["answer"] = np.random.default_rng(0).permutation(scrambled["answer"].to_numpy())
    ctx_b = dataclasses.replace(ctx, df=scrambled)
    for name, fn in (("OW-L", run_ow_l), ("OW-I", run_ow_i)):
        _, _, x_a = fn(ctx, A, tr, te)
        _, _, x_b = fn(ctx_b, A, tr, te)
        assert np.allclose(x_a, x_b), f"{name}: estimated accuracies depend on the gold answers"
        pa, pb = _ow_pick(ctx, A, x_a), _ow_pick(ctx_b, A, x_b)
        assert [pa(i) for i in te] == [pb(i) for i in te], \
            f"{name}: verdicts changed when every label was scrambled -> label leakage"
        assert 1e-3 <= x_a.min() and x_a.max() <= 1 - 1e-3, f"{name}: accuracies out of range"


@check
def jury_reads_no_labels_at_all(cfg, ctx, slow):
    """`aggregator` may not read the *held-out* labels. `jury` may not read any.

    Its whole claim is that the pool's own agreement identifies the experts'
    reliabilities, so scrambling every gold answer -- training half included --
    must leave both the fitted parameters and the verdicts bit-identical. Only
    the score this is graded against may move.
    """
    c = _jury(ctx, use_rollouts=False)
    set_seed(cfg["seed"])
    a = JuryRouter(c.cfg).fit(c)

    scrambled = c.df.copy()
    rng = np.random.default_rng(0)
    scrambled["answer"] = rng.permutation(scrambled["answer"].to_numpy())
    c_b = dataclasses.replace(c, df=scrambled)
    set_seed(cfg["seed"])
    b = JuryRouter(c_b.cfg).fit(c_b)

    assert np.allclose(a.pi, b.pi), "jury's fitted reliabilities depend on the gold answers"
    qids = c.df.loc[c.test_mask, "qid"].to_numpy()
    assert [a.verdict(q) for q in qids] == [b.verdict(q) for q in qids], \
        "jury's verdicts changed when every label was scrambled -> label leakage"


@check
def jury_greedy_is_the_log_odds_weighted_vote(cfg, ctx, slow):
    """The reduction the paper states, made executable.

    With one greedy answer per expert the emission is one-hot, so

        log P(a | z = v) = const + sum_{e : a_e = v} log( pi_e m / (1 - pi_e) )

    and the posterior argmax is exactly the Nitzan-Paroush log-odds weighted
    plurality, with m = |candidates| - 1 fixing the weights per question. If this
    ever stops holding, the paper's claim that greedy `jury` is a weighted vote --
    and therefore confined to the answers the experts proposed -- is wrong.
    """
    c = _jury(ctx, use_rollouts=False, lure="none")
    set_seed(cfg["seed"])
    r = JuryRouter(c.cfg).fit(c)

    for qid in c.df.loc[c.test_mask, "qid"].to_numpy():
        t = r._t[r._row_of[qid]]
        if t is None or len(t.cands) == 1:
            continue
        m = len(t.cands) - 1
        w = np.log(r.pi * m / (1 - r.pi))
        score = np.array([float(w @ (t.emit[:, v] > 0)) for v in range(len(t.cands))])
        assert t.cands[int(np.argmax(score))] == r.verdict(qid), \
            f"q{qid}: greedy jury departed from the log-odds weighted vote"


@check
def jury_posterior_is_a_distribution(cfg, ctx, slow):
    """Every lure mode must leave a normalised posterior over the candidates."""
    for mode in ("none", "family", "pool"):
        c = _jury(ctx, use_rollouts=False, lure=mode)
        set_seed(cfg["seed"])
        r = JuryRouter(c.cfg).fit(c)
        assert np.all((r.pi > 0) & (r.pi < 1)), f"{mode}: pi outside (0, 1)"
        for t in r._t:
            if t is None:
                continue
            g, _ = posterior(t, r.pi, r.lam, len(r.groups))
            assert g.shape == (len(t.cands),), f"{mode}: posterior has the wrong support"
            assert abs(g.sum() - 1.0) < 1e-9, f"{mode}: posterior does not sum to 1"
            assert np.all(g >= 0), f"{mode}: negative posterior mass"


@check
def jury_lure_channel_never_helps(cfg, ctx, slow):
    """The negative result the paper reports, pinned.

    Two statements, from Appendix C. Under the fractional evidence weight
    (alpha = 1) EM drives every susceptibility to zero on the rollout emissions
    (below 0.01 here, since EM stops at a tolerance) and the verdicts are
    identical to the model without a lure; that is checked on the cache's own
    split and is fast. Under the multinomial default
    (alpha = n) the susceptibilities are fitted away from zero and the channel
    costs accuracy on average over the paper's ten splits (-0.6 points by
    lineage, on no split positive, and -0.4 pool-wide); that is the claim
    `scripts/testimony_structure.py` measures, and it is re-checked here only
    with --slow because it fits the jury thirty times.
    """
    singletons, _ = lure_groups(ctx.expert_ids, "family")
    assert (singletons < 0).any(), \
        "family grouping gave a lure to every expert; single-member lineages must not get one"

    base = _jury(ctx, use_rollouts=True, lure="none", alpha=1.0)
    set_seed(cfg["seed"])
    ref = JuryRouter(base.cfg).fit(base).predict(base)
    for mode in ("family", "pool"):
        c = _jury(ctx, use_rollouts=True, lure=mode, alpha=1.0)
        set_seed(cfg["seed"])
        router = JuryRouter(c.cfg).fit(c)
        assert np.all(np.asarray(router.lam) < 0.01), \
            f"{mode}, alpha = 1: a susceptibility survived EM (max {np.max(router.lam):.3g})"
        assert np.array_equal(router.predict(c).correct, ref.correct), \
            f"{mode}, alpha = 1: the verdicts differ from the model without a lure"

    if not slow:
        return
    from router_lab.techniques.multiseed import _resplit
    diff = {"family": [], "pool": []}
    for seed in range(1, 11):
        d = ctx.df.copy()
        d["split"] = _resplit(d, cfg["dataset"]["train_frac"], seed)
        c0 = build_context(cfg, d, ctx.preds, Path(cfg["paths"]["cache_dir"]))
        base = _jury(c0, use_rollouts=True, lure="none")
        set_seed(seed)
        ref = JuryRouter(base.cfg).fit(base).predict(base).correct.mean()
        for mode in diff:
            c = _jury(c0, use_rollouts=True, lure=mode)
            set_seed(seed)
            diff[mode].append(JuryRouter(c.cfg).fit(c).predict(c).correct.mean() - ref)
    for mode, v in diff.items():
        assert np.mean(v) <= 0, \
            f"{mode}: the lure channel raised mean accuracy by {100*np.mean(v):+.2f} points over ten splits"
    assert max(diff["family"]) <= 0, "family: the lure channel was positive on some split"


@check
def jury_rollouts_cost_only_the_generative_half(cfg, ctx, slow):
    """Repeated sampling is not uniformly n_samples times the work: on multiple
    choice the samples come from a distribution one forward pass already
    produced. Cost must reflect that, or the frontier figure is wrong."""
    c = _jury(ctx, use_rollouts=True)
    set_seed(cfg["seed"])
    res = JuryRouter(c.cfg).fit(c).predict(c)
    pool = float(ctx.params_b.sum())
    numeric = np.array([t == "numeric" for t in ctx.tasks[ctx.test_mask]])
    if not numeric.any():
        return
    assert np.allclose(res.cost[~numeric], pool), \
        "multiple-choice questions were charged for re-sampling that never happened"
    assert np.all(res.cost[numeric] > pool), "generative questions were not charged for sampling"


@check
def distributional_m_step_matches_the_equation_in_the_paper(cfg, ctx, slow):
    """Equation (5): pi_e = (1/|Q_e|) sum_q sum_v gamma_q(v) p_{e,q}(v).

    The paper states the M-step for distributional emissions, which is the regime
    the headline number is in. Recomputing it from the equation, by hand, at the
    fitted point must return the fitted parameters: if the code ever drifts from
    the printed update -- a stray alpha, a renormalisation, a different treatment
    of experts that did not answer -- the method in the paper stops being the
    method that produced the number.
    """
    c = _jury(ctx, use_rollouts=True, lure="none")
    set_seed(cfg["seed"])
    r = JuryRouter(c.cfg).fit(c)
    if not r.use_rollouts:
        return

    rows = [r._t[i] for i in np.where(c.train_mask)[0] if r._t[i] is not None]
    num = np.zeros(len(r.pi))
    den = np.zeros(len(r.pi))
    for t in rows:
        if len(t.cands) == 1:            # unanimous: constrains no parameter
            continue
        gamma, _ = posterior(t, r.pi, r.lam, 0, r.alpha)
        answered = t.emit.sum(1) > 0
        num += np.where(answered, t.emit @ gamma, 0.0)
        den += answered.astype(float)
    by_hand = np.clip(num / np.maximum(den, 1e-12), 0.02, 0.98)

    assert np.allclose(r.pi, by_hand, atol=1e-6), \
        f"fitted pi departs from equation (5) by {np.abs(r.pi - by_hand).max():.2e}"


@check
def collapsing_to_modes_keeps_the_samples_and_drops_the_distribution(cfg, ctx, slow):
    """The sample-matched control of the paper's decomposition table.

    Each expert must end up with a point mass on the answer it sampled most
    often, so a rule given this has paid for the repeated sampling in full and
    kept nothing of the distribution. If the collapse silently dropped experts or
    picked something other than the mode, the control would understate what the
    extra samples are worth and the attribution of the +6.2 would be wrong in our
    favour.
    """
    dist, _ = rollout_distributions(cfg["paths"]["cache_dir"], set(ctx.expert_ids))
    if not dist:
        return
    modes = collapse_to_modes(dist)
    assert set(modes) == set(dist), "collapsing to modes lost questions"
    for q, per_expert in dist.items():
        assert set(modes[q]) == {e for e, ks in per_expert.items() if ks}, \
            f"q{q}: collapsing to modes lost an expert that had samples"
        for e, ks in per_expert.items():
            if not ks:
                continue
            (key, mass), = modes[q][e].items()
            assert mass == 1.0, f"q{q}/{e}: collapsed emission is not a point mass"
            assert ks[key] == max(ks.values()), \
                f"q{q}/{e}: collapsed onto {key!r}, which is not the mode"


@check
def jury_task_support_keeps_all_four_options(cfg, ctx, slow):
    """The support control of the paper: with `support: task` a multiple-choice
    question keeps the task's four options as its candidate set, so m = 3
    everywhere on multiple choice, an unproposed option is a zero-mass
    candidate, and a unanimous question is no longer a singleton and enters EM.
    GSM8K keeps the observed support. Still label-free."""
    from router_lab.data import LETTERS
    a = JuryRouter(_jury(ctx, use_rollouts=False, support="observed").cfg).fit(
        _jury(ctx, use_rollouts=False, support="observed"))
    c = _jury(ctx, use_rollouts=False, support="task")
    set_seed(cfg["seed"])
    b = JuryRouter(c.cfg).fit(c)
    for i, task in enumerate(ctx.tasks):
        t = b._t[i]
        if task == "mc":
            assert t is not None and list(t.cands) == LETTERS, f"q{i}: support {t and t.cands}"
        elif t is not None:
            assert list(t.cands) == list(a._t[i].cands), f"q{i}: numeric support changed"
    tr = np.where(ctx.train_mask)[0]
    singles = lambda r: sum(1 for i in tr if r._t[i] is not None and len(r._t[i].cands) == 1)
    assert singles(a) > 0 and singles(b) < singles(a), \
        "no unanimous question moved from singleton to four-option configuration"

    scrambled = c.df.copy()
    scrambled["answer"] = np.random.default_rng(0).permutation(scrambled["answer"].to_numpy())
    set_seed(cfg["seed"])
    d = JuryRouter(c.cfg).fit(dataclasses.replace(c, df=scrambled))
    assert np.allclose(b.pi, d.pi), "the four-option fit depends on the gold answers"


@check
def jury_per_task_still_reads_no_labels(cfg, ctx, slow):
    """Per-task reliabilities are one parameter per expert and task, and the task
    is part of the query. Fitting them must still be blind to every gold answer,
    or the specialisation control compares a supervised rule with another
    supervised rule."""
    c = _jury(ctx, use_rollouts=True, per_task=True)
    set_seed(cfg["seed"])
    a = JuryRouter(c.cfg).fit(c)
    assert len(a.pi_task) >= 2, "per_task fitted a single group -- the control is vacuous"

    scrambled = c.df.copy()
    rng = np.random.default_rng(0)
    scrambled["answer"] = rng.permutation(scrambled["answer"].to_numpy())
    c_b = dataclasses.replace(c, df=scrambled)
    set_seed(cfg["seed"])
    b = JuryRouter(c_b.cfg).fit(c_b)
    for t, pi in a.pi_task.items():
        assert np.allclose(pi, b.pi_task[t]), \
            f"per-task reliabilities for {t} depend on the gold answers"


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    slow = "--slow" in sys.argv
    cfg, ctx = _ctx()
    selected = [c for c in CHECKS if not args or any(a in c.__name__ for a in args)]
    if not selected:
        print(f"no check matches {args}")
        return 1

    failures = []
    for fn in selected:
        t0 = time.time()
        try:
            fn(cfg, ctx, slow)
            print(f"  PASS  {fn.__name__:<42} {time.time()-t0:5.1f}s")
        except Exception as exc:
            failures.append((fn.__name__, exc))
            print(f"  FAIL  {fn.__name__:<42} {time.time()-t0:5.1f}s")
            traceback.print_exc()

    print(f"\n{len(selected)-len(failures)}/{len(selected)} checks passed"
          )
    for name, exc in failures:
        print(f"  - {name}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
