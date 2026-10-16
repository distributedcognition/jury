"""A resampling distribution for the headline comparisons that refits inside it.

    python paper/scripts/refit_bootstrap.py [B]     # from the repo root

`run_paper_results.py` reports paired bootstrap intervals over questions, computed
from predictions that ten fitted models produced on ten overlapping splits. Those
intervals are *conditional on those ten fits*: resampling questions moves which
questions are scored, never which questions any weight was estimated from, so the
variability contributed by the choice of training half is missing from them. The
ten splits share all 360 questions, so that omission is not obviously small.

This script measures the unconditional thing. One replica is a whole pipeline:

    draw a fresh stratified train/test split
      -> refit every method on its training half (EM for the jury, labels for the
         supervised baselines, the scorer's logistic regression, MoRE's selector)
      -> score the held-out half
      -> resample that half's questions with replacement, once

so the spread over replicas carries both sources at once -- which questions were
available to fit, and which questions were scored. The percentile interval of the
per-replica paired difference is then a genuine interval for "the jury minus X",
not for "the jury minus X given these ten fits".

Nothing here reruns a language model; a replica is a second of arithmetic over the
committed cache, which is why the honest interval is affordable at all.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]          # this supplementary folder
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from aggregator_vs_literature import (run_majority, run_more,        # noqa: E402
                                      run_soft_equal, run_soft_ow)
import router_lab.techniques.aggregator as agg_mod                   # noqa: E402
import router_lab.techniques.jury as jury_mod                        # noqa: E402
from router_lab.techniques.aggregator import (answer_key,            # noqa: E402
                                              rollout_distributions)
from router_lab.techniques.base import (build_context,               # noqa: E402
                                        prediction_matrix)
from router_lab.techniques.multiseed import _resplit                 # noqa: E402
from router_lab.techniques.runner import fit_router                  # noqa: E402
from router_lab.utils import load_config                             # noqa: E402

REF = "jury+rollouts"
BASE_SEED = 100_000          # disjoint from the ten reporting splits


def bootstrap_dataset(df, preds, emb, dist, boot, split):
    """A resampled copy of the benchmark, with every duplicate given its own id.

    `boot` indexes the original questions with replacement. Duplicates cannot be
    left sharing a qid -- every router keys its per-question state by it -- so each
    drawn row gets a fresh synthetic id and the cached predictions and sampled
    distributions are carried across to it. `split` is assigned over the
    *original* questions and inherited, so all copies of one question land on
    the same side and a supervised baseline cannot be fitted on a question it is
    then scored on.
    """
    n_exp = len(preds) // len(df)
    d = df.iloc[boot].copy()
    d["orig_qid"] = d["qid"].to_numpy()
    d["qid"] = np.arange(len(d))
    d["split"] = split[boot]
    d = d.reset_index(drop=True)

    # preds is 9 consecutive rows per question once sorted, so the resample is one
    # fancy-index rather than 360 concatenations.
    P = preds.sort_values(["qid", "expert_id"]).reset_index(drop=True)
    idx = (np.asarray(boot)[:, None] * n_exp + np.arange(n_exp)).ravel()
    p2 = P.iloc[idx].copy()
    p2["qid"] = np.repeat(np.arange(len(d)), n_exp)

    dist2 = {k: dist.get(q, {}) for k, q in enumerate(d["orig_qid"].to_numpy())}
    return d, p2.reset_index(drop=True), emb[boot], dist2


def replica(cfg, df, preds, cache, dist, n_samples, seed, rng, emb=None,
            nested=False):
    """One split, refit everywhere, scored on the half held out.

    With `nested`, the 360 questions are first resampled with replacement, so the
    replica varies the sample of questions as well as the split and the fit; the
    cached embeddings are then indexed by hand, because `compute_embeddings`
    reuses its cache on any frame of the same length and would otherwise hand back
    the original order.
    """
    d = df.copy()
    d["split"] = _resplit(d, cfg["dataset"]["train_frac"], seed)
    cfg.raw["seed"] = seed
    if nested:
        boot = rng.integers(0, len(df), size=len(df))
        d, preds, emb_b, dist = bootstrap_dataset(df, preds, emb,
                                                  dist, boot, d["split"].to_numpy())
        ctx = dataclasses.replace(build_context(cfg, d, preds, cache), emb=emb_b)
        # Both routers reload the rollouts from the cache themselves, keyed by
        # qid. Under resampling the qids are synthetic, so an unpatched reload
        # pairs each question with another question's distribution -- silently,
        # and it costs about sixty points. They are pointed at the resampled
        # distributions for the duration of the replica instead.
        stub = lambda *_a, **_k: (dist, n_samples)
        agg_mod.rollout_distributions = stub
        jury_mod.rollout_distributions = stub
    else:
        ctx = build_context(cfg, d, preds, cache)
    A = np.vectorize(answer_key, otypes=[object])(
        np.repeat(ctx.tasks[:, None], len(ctx.expert_ids), axis=1),
        prediction_matrix(ctx))
    tr, te = np.where(ctx.train_mask)[0], np.where(ctx.test_mask)[0]

    out = {}
    out["majority"] = run_majority(ctx, A, tr, te)[0]
    out["soft-equal"] = run_soft_equal(ctx, A, tr, te, dist, n_samples)[0]
    out["soft-OW"] = run_soft_ow(ctx, A, tr, te, dist, n_samples)[0]
    out["MoRE+rollouts"] = run_more(ctx, A, tr, te, dist, n_samples)[0]

    cfg.raw["experiments"]["aggregator"]["use_rollouts"] = True
    out["aggregator+rollouts"] = fit_router("aggregator", cfg, ctx, seed) \
        .predict(ctx).correct

    cfg.raw["experiments"]["jury"].update(use_rollouts=True, lure="none",
                                          collapse_modes=False, per_task=False)
    out[REF] = fit_router("jury", cfg, ctx, seed).predict(ctx).correct

    # Two statistics per replica. `plain` is the accuracy on the half that was
    # actually held out, and it is the one we report: each replica already draws a
    # fresh 180 questions to score, so the spread over replicas carries the
    # sampling of questions and the sampling of the training half at once.
    # `resampled` additionally bootstraps those 180, which counts the question
    # sampling a second time and is therefore conservative; we compute it only to
    # show how much of the width is that double count.
    idx = rng.integers(0, len(te), size=len(te))
    return ({k: float(np.asarray(v).mean()) for k, v in out.items()},
            {k: float(np.asarray(v)[idx].mean()) for k, v in out.items()})


LABEL = {"majority": "Majority vote",
         "soft-equal": "Equal-weight pool",
         "soft-OW": "Supervised linear fusion",
         "MoRE+rollouts": "MoRE + repeated sampling",
         "aggregator+rollouts": "Answer scoring + repeated sampling"}


def table(cfg) -> None:
    """Three intervals side by side, adding one source of variability at a time.

    Left: the paired bootstrap of the main table, which resamples questions and
    holds the ten fitted models fixed. Middle: fresh split and refit, on the same
    360 questions. Right: the nested bootstrap -- questions resampled with
    replacement as well -- which is the only one of the three that is a confidence
    interval for the difference on new questions rather than on these.
    """
    res = Path(cfg["paths"]["results_dir"])
    cond = json.loads((res / "paper_results.json").read_text())["bootstrap_vs_jury"]
    refit = json.loads((res / "refit_bootstrap.json").read_text())
    nest = json.loads((res / "nested_bootstrap.json").read_text())

    def ci(d):
        return (f"{100*d['delta']:+.1f}~[{100*d.get('lo', d.get('lo95')):+.1f}, "
                f"{100*d.get('hi', d.get('hi95')):+.1f}]")

    lines = ["\\begin{tabular}{lrrr}", "\\toprule",
             "& \\multicolumn{3}{c}{$\\Delta$ vs.\\ jury, $95\\%$ interval} \\\\",
             "\\cmidrule(lr){2-4}",
             "Resampled & Questions & Split and fit & Both \\\\",
             "\\midrule"]
    for k in ("soft-OW", "MoRE+rollouts", "aggregator+rollouts", "soft-equal",
              "majority"):
        lines.append(f"{LABEL[k]} & {ci(cond[k])} & {ci(refit['delta_vs_jury'][k])}"
                     f" & {ci(nest['delta_vs_jury'][k])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    out = ROOT / "tables" / "refit.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"[tables] -> {out}  ({refit['replicas']} + {nest['replicas']} replicas)")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--table":
        table(load_config(ROOT / "config.yaml"))
        return
    args = [a for a in sys.argv[1:] if a != "--nested"]
    for flag in ("--cache", "--jobs"):             # drop the flag and its value
        if flag in args:
            i = args.index(flag)
            del args[i:i + 2]
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 1
    nested = "--nested" in sys.argv
    B = int(args[0]) if args else 2000
    cfg = load_config(ROOT / "config.yaml")
    if "--cache" in sys.argv:                      # extended-benchmark branch
        cfg.raw["paths"]["cache_dir"] = sys.argv[sys.argv.index("--cache") + 1]
    cache = Path(cfg["paths"]["cache_dir"])
    df = pd.read_csv(cache / "questions.csv").fillna("")
    preds = pd.read_csv(cache / "predictions.csv")
    dist, n_samples = rollout_distributions(cache, {e["id"] for e in cfg["experts"]})
    emb = np.load(cache / "embeddings.npy") if nested else None

    acc, acc_rs = {}, {}
    if jobs > 1:
        # `--jobs J` runs the replicas in J forked processes. Each chunk draws
        # its bootstrap samples from its own generator, so the stream differs
        # from the serial run's but every replica is still an independent draw.
        from multiprocessing import Pool
        global _WORK
        _WORK = (cfg, df, preds, cache, dist, n_samples, emb, nested)
        chunks = [list(range(B))[c::jobs] for c in range(jobs)]
        done = 0
        with Pool(jobs) as pool:
            for results in pool.imap_unordered(_run_chunk, list(enumerate(chunks))):
                for plain, resampled in results:
                    for k, v in plain.items():
                        acc.setdefault(k, []).append(v)
                    for k, v in resampled.items():
                        acc_rs.setdefault(k, []).append(v)
                done += len(results)
                print(f"[refit] {done}/{B} replicas", flush=True)
    else:
        rng = np.random.default_rng(0)
        for b in range(B):
            plain, resampled = replica(cfg, df, preds, cache, dist, n_samples,
                                       BASE_SEED + b, rng, emb, nested)
            for k, v in plain.items():
                acc.setdefault(k, []).append(v)
            for k, v in resampled.items():
                acc_rs.setdefault(k, []).append(v)
            if (b + 1) % 100 == 0:
                print(f"[refit] {b+1}/{B} replicas", flush=True)

    A = {k: np.asarray(v) for k, v in acc.items()}
    R = {k: np.asarray(v) for k, v in acc_rs.items()}

    def intervals(M):
        out = {}
        for k, v in M.items():
            if k == REF:
                continue
            d = M[REF] - v
            lo95, hi95 = np.percentile(d, [2.5, 97.5])
            lo90, hi90 = np.percentile(d, [5, 95])
            p = 2 * min(float((d <= 0).mean()), float((d >= 0).mean()))
            out[k] = dict(delta=float(d.mean()), lo95=float(lo95),
                          hi95=float(hi95), lo90=float(lo90), hi90=float(hi90),
                          p=max(p, 1.0 / B),
                          equivalence=float(max(abs(lo90), abs(hi90))))
        return out

    summary = {"replicas": B, "base_seed": BASE_SEED,
               "accuracy": {k: dict(mean=float(v.mean()), sd=float(v.std(ddof=1)))
                            for k, v in A.items()},
               "delta_vs_jury": intervals(A),
               "delta_vs_jury_resampled": intervals(R)}

    summary["nested"] = nested
    name = "nested_bootstrap.json" if nested else "refit_bootstrap.json"
    out = Path(cfg["paths"]["results_dir"]) / name
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n[refit] -> {out}")
    for k, v in sorted(summary["delta_vs_jury"].items(),
                       key=lambda kv: kv[1]["delta"]):
        print(f"  {k:<22} {100*v['delta']:+5.2f}  "
              f"95% [{100*v['lo95']:+.1f}, {100*v['hi95']:+.1f}]  p={v['p']:.4f}")


_WORK = None


def _run_chunk(job):
    """One forked worker: its replicas, its own generator, the routers' logging silenced."""
    import contextlib
    import io
    chunk_id, bs = job
    cfg, df, preds, cache, dist, n_samples, emb, nested = _WORK
    rng = np.random.default_rng(1000 + chunk_id)
    out = []
    with contextlib.redirect_stdout(io.StringIO()):
        for b in bs:
            out.append(replica(cfg, df, preds, cache, dist, n_samples,
                               BASE_SEED + b, rng, emb, nested))
    return out


if __name__ == "__main__":
    main()
