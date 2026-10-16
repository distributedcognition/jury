"""Robustness of the EM fit to its initialisation: multi-start on the main pool.

    python scripts/em_multistart.py

The paper fits the jury's reliabilities with a single EM run from pi_e = 0.6 for
every expert and no restarts. This script re-runs the same EM, on the same ten
splits of the same nine-expert pool, from a set of other starts and compares the
runs on three things: the final marginal log-likelihood of the training
configurations (the objective EM climbs), the fitted reliability vector, and the
test accuracy of the resulting verdicts.

Starts. The paper's constant 0.6; constants 0.3, 0.5 and 0.9 (0.3 is below
1/|V(q)| on much of the benchmark, so it starts every expert on the "vote
against" side of the log-odds); a data-driven start that sets pi_e to the mass
each expert puts on the plurality answer; and twenty random vectors drawn
uniformly on [0.05, 0.95]. Everything else -- the stopping rule, the clip to
[0.02, 0.98], the evidence weight -- is the paper's.

Both evidence regimes of Table 3 are run: the distributional jury with the
multinomial weight alpha = 8 (the paper's main result) and the greedy jury with
one answer per expert (alpha = 1). Nothing here runs an LLM; the answers and
samples come from the cache. The per-run results go to
results/em_multistart.json (reused on the next call unless --recompute is
given) and the summary table to tables/multistart.tex. Means and standard
deviations over splits follow the paper's convention (population sd, as in
`run_paper_results.py`).
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]          # this supplementary folder
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from run_paper_results import SEEDS, TABLEDIR, write                   # noqa: E402
from router_lab.techniques.base import build_context                   # noqa: E402
from router_lab.techniques.jury import EPS, _log_emission              # noqa: E402
from router_lab.techniques.multiseed import _resplit                   # noqa: E402
from router_lab.techniques.runner import fit_router                    # noqa: E402
from router_lab.utils import Config, load_config                       # noqa: E402

N_RANDOM = 20
CONSTANTS = (0.6, 0.3, 0.5, 0.9)
REGIMES = (("rollouts", True), ("greedy", False))


# --------------------------------------------------------------------------- #
# The EM of `router_lab.techniques.jury.fit_em`, no lure, with a free start and
# the objective reported alongside.
# --------------------------------------------------------------------------- #
def _log_lik_terms(t, pi, alpha):
    """log P(a_q | z = v) summed over experts, one entry per candidate v."""
    log_e = _log_emission(t, pi, np.zeros(len(pi)), alpha)
    return log_e[:, :, 0].sum(0)


def log_likelihood(testimony, pi, alpha):
    """Marginal log-likelihood of the configurations under a uniform prior on z.

    Singleton questions contribute a constant that does not depend on pi, so
    they are skipped, as they are in the EM.
    """
    total = 0.0
    for t in testimony:
        if t is None or len(t.cands) == 1:
            continue
        ll = _log_lik_terms(t, pi, alpha)
        mx = ll.max()
        total += mx + np.log(np.exp(ll - mx).sum()) - np.log(len(t.cands))
    return float(total)


def fit_em_from(testimony, pi0, max_iter=200, tol=1e-7, alpha=1.0):
    """-> (pi, iterations). Same updates and stopping rule as `fit_em`."""
    pi = np.array(pi0, dtype=float)
    prev = None
    for it in range(1, max_iter + 1):
        num, den = np.zeros(len(pi)), np.zeros(len(pi))
        for t in testimony:
            if t is None or len(t.cands) == 1:
                continue
            ll = _log_lik_terms(t, pi, alpha)
            ll -= ll.max()
            gamma = np.exp(ll)
            gamma /= gamma.sum() + EPS
            answered = t.emit.sum(1) > 0
            num += np.where(answered, t.emit @ gamma, 0.0)
            den += answered.astype(float)
        pi = np.clip(num / np.maximum(den, EPS), 0.02, 0.98)
        if prev is not None and np.abs(pi - prev).max() < tol:
            break
        prev = pi
    return pi, it


def starts(testimony, n_exp, seed):
    """name -> initial pi, in the order the paper's default comes first."""
    out = {f"const_{c}": np.full(n_exp, c) for c in CONSTANTS}
    # Data-driven: each expert's average mass on the plurality answer.
    num, den = np.zeros(n_exp), np.zeros(n_exp)
    for t in testimony:
        if t is None:
            continue
        answered = t.emit.sum(1) > 0
        num += np.where(answered, t.emit[:, t.plur], 0.0)
        den += answered
    out["plurality"] = np.clip(num / np.maximum(den, 1), 0.02, 0.98)
    rng = np.random.default_rng(1000 + seed)
    for r in range(N_RANDOM):
        out[f"random_{r:02d}"] = rng.uniform(0.05, 0.95, size=n_exp)
    return out


# --------------------------------------------------------------------------- #
def run_seed(args):
    cfg_raw, cache, seed = args
    import contextlib
    import io
    cfg = Config(cfg_raw)
    cache = Path(cache)
    df = pd.read_csv(cache / "questions.csv").fillna("")
    preds = pd.read_csv(cache / "predictions.csv")
    d = df.copy()
    d["split"] = _resplit(d, cfg["dataset"]["train_frac"], seed)
    cfg.raw["seed"] = seed
    ctx = build_context(cfg, d, preds, cache)

    out = {}
    for regime, use_roll in REGIMES:
        cfg.raw["experiments"]["jury"].update(use_rollouts=use_roll, lure="none", alpha="n")
        with contextlib.redirect_stdout(io.StringIO()):
            router = fit_router("jury", cfg, ctx, seed)      # the paper's fit, from 0.6
        tr = [i for i in np.where(ctx.train_mask)[0] if router._t[i] is not None]
        rows = [router._t[i] for i in tr]
        n_exp = len(ctx.expert_ids)
        runs = {}
        for name, pi0 in starts(rows, n_exp, seed).items():
            pi, its = fit_em_from(rows, pi0, router.max_iter, router.tol, router.alpha)
            router.pi = pi
            acc = float(router.predict(ctx).correct.mean())
            runs[name] = dict(pi0=[float(x) for x in pi0], pi=[float(x) for x in pi],
                              iters=its, loglik=log_likelihood(rows, pi, router.alpha),
                              acc=acc)
        out[regime] = dict(alpha=router.alpha, n_train=len(rows), runs=runs)
    return seed, out


def summarise(per_seed, regime):
    """Across seeds: how far every start lands from the paper's default."""
    names = list(next(iter(per_seed.values()))[regime]["runs"])
    rows = {}
    for name in names:
        d_ll, d_pi, d_acc, acc, its = [], [], [], [], []
        for seed, res in per_seed.items():
            r = res[regime]["runs"]
            ref = r["const_0.6"]
            d_ll.append(r[name]["loglik"] - ref["loglik"])
            d_pi.append(float(np.abs(np.array(r[name]["pi"]) - np.array(ref["pi"])).max()))
            d_acc.append(100 * (r[name]["acc"] - ref["acc"]))
            acc.append(100 * r[name]["acc"])
            its.append(r[name]["iters"])
        same = [abs(x) < 1e-3 for x in d_ll]           # same optimum as the default start
        rows[name] = dict(acc_mean=float(np.mean(acc)), acc_sd=float(np.std(acc)),
                          d_loglik=[float(x) for x in d_ll],
                          d_loglik_max=float(np.max(np.abs(d_ll))),
                          d_pi_max=float(np.max(d_pi)),
                          d_pi_max_same=float(max([p for p, s in zip(d_pi, same) if s], default=0.0)),
                          d_acc_max=float(np.max(np.abs(d_acc))),
                          n_same=int(sum(same)), n_splits=len(d_ll),
                          iters_mean=float(np.mean(its)), iters_max=int(np.max(its)))
    return rows


# --------------------------------------------------------------------------- #
# The appendix table: one block per regime, the starts grouped as in the text.
# --------------------------------------------------------------------------- #
GROUPS = (("Constant $0.6$ (default)", ("const_0.6",)),
          ("Constant $0.3$", ("const_0.3",)),
          ("Constant $0.5$", ("const_0.5",)),
          ("Constant $0.9$", ("const_0.9",)),
          ("Plurality agreement", ("plurality",)),
          ("Random, $20$ draws", tuple(f"random_{r:02d}" for r in range(N_RANDOM))))


def table(summary, per_seed):
    def block(regime, title):
        s = summary[regime]
        out = [f"\\multicolumn{{5}}{{l}}{{\\emph{{{title}}}}} \\\\"]
        for label, names in GROUPS:
            n_runs = sum(s[n]["n_splits"] for n in names)
            n_same = sum(s[n]["n_same"] for n in names)
            d_ll = np.array([x for n in names for x in s[n]["d_loglik"]])
            other = d_ll[np.abs(d_ll) > 1e-3]
            if len(other) == 0:
                ll_cell = "--"
            elif other.min() > 0:
                ll_cell = f"$+{other.min():.0f}$ to $+{other.max():.0f}$"
            elif other.max() < 0:
                ll_cell = f"${other.max():,.0f}$ to ${other.min():,.0f}$".replace(",", "{,}")
            else:
                ll_cell = f"${other.min():,.0f}$ to $+{other.max():,.0f}$".replace(",", "{,}")
            accs = [100 * per_seed[seed][regime]["runs"][n]["acc"]
                    for n in names for seed in per_seed]
            out.append(f"{label} & {n_same}/{n_runs} & {ll_cell} & "
                       f"{min(accs):.1f} & {max(accs):.1f} \\\\")
        return out

    lines = ["\\begin{tabular}{lrrrr}", "\\toprule",
             "Start & Same solution & $\\Delta$ log-lik.\\ of the others & "
             "\\multicolumn{2}{c}{Test accuracy (\\%)} \\\\",
             " & as default & (nats) & min & max \\\\", "\\midrule"]
    lines += block("rollouts", "Sampled distributions, $\\alpha = 8$")
    lines += ["\\midrule"]
    lines += block("greedy", "Greedy answers, $\\alpha = 1$")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(TABLEDIR / "multistart.tex", "\n".join(lines))


def main() -> None:
    cfg = load_config(ROOT / "config.yaml")
    if "--cache" in sys.argv:
        cfg.raw["paths"]["cache_dir"] = sys.argv[sys.argv.index("--cache") + 1]
    cache = Path(cfg["paths"]["cache_dir"])
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else len(SEEDS)
    out = Path(cfg["paths"]["results_dir"]) / "em_multistart.json"

    if out.exists() and "--recompute" not in sys.argv:
        per_seed = json.loads(out.read_text())["per_seed"]
        print(f"[multistart] reusing {out}; pass --recompute to re-run the EM")
    else:
        with Pool(min(jobs, len(SEEDS))) as pool:
            per_seed = dict(pool.map(run_seed, [(cfg.raw, str(cache), s) for s in SEEDS]))
    per_seed = {int(k): v for k, v in sorted(per_seed.items(), key=lambda kv: int(kv[0]))}

    summary = {regime: summarise(per_seed, regime) for regime, _ in REGIMES}
    out.write_text(json.dumps({"seeds": SEEDS, "starts": list(summary["rollouts"]),
                               "per_seed": per_seed, "summary": summary}, indent=1))
    table(summary, per_seed)

    for regime, _ in REGIMES:
        alpha = per_seed[SEEDS[0]][regime]["alpha"]
        print(f"\n== {regime} (alpha = {alpha:g}) -- differences are to the paper's start 0.6, "
              f"max over {len(SEEDS)} splits ==")
        print(f"{'start':<12}{'acc':>14}{'|d loglik|':>12}{'max|d pi|':>11}{'|d acc|':>9}{'iters':>12}")
        for name, s in summary[regime].items():
            print(f"{name:<12}{s['acc_mean']:>8.1f} +- {s['acc_sd']:<4.1f}"
                  f"{s['d_loglik_max']:>11.2e}{s['d_pi_max']:>11.2e}{s['d_acc_max']:>9.2f}"
                  f"{s['iters_mean']:>8.0f} ({s['iters_max']})")
    print(f"[results] -> {out}")


if __name__ == "__main__":
    main()
