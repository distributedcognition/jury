"""SkillAggregation against the jury, on the multiple-choice questions only.

    python paper/scripts/skillaggregation_mc.py --cache results/cache_ext

SkillAggregation (Sun et al., ACL 2025) is the closest reference-free neighbour of
the jury, so the paper measures it. What is run here is not the authors' code, and
the differences are these.

  * The released implementation (github.com/BriansIDP/Crowdsourcing, branch
    `crowdlayer_dev`) is binary: a 2-way bottleneck and two skills per judge. The
    multi-class method exists only as the derivation of their Appendix F, which
    this file implements: a C-way bottleneck s_n = softmax(W f(X_n) + b), one
    C x C confusion matrix per judge, the prediction sum_c s_{n,c} C^k_{c,b} of
    judge k's estimate, cross-entropy against that estimate, the regulariser
    sum_{b<C} sum_{c<C} (C^k_{c,b} - C^k_{C,b})^2, and the conditional-independence
    posterior for the group estimate. SkillAggregation-X makes the confusion
    matrices a function of the context through one linear layer.
  * Everything else follows the release: GPT-2 as the context encoder, trained
    end to end and read at the final hidden state of the last token, batch 8,
    learning rate 1e-4, cosine schedule with 3% warm-up, no weight decay, every
    judge starting at skill 0.7, lambda = 0.1. The context is the question followed
    by its four options. A frozen sentence encoder was tried first and is not a
    fair test: the correct letter of a multiple-choice question is not a function
    of its text, so the bottleneck stays at chance and nothing anchors the skills.
  * Three adaptations. (i) Confusion rows are a softmax, so they are probabilities
    by construction; the release clamps an unconstrained linear layer. (ii) We
    train for 40 epochs where the release trains for 10: on 1440 items ten epochs
    are 1800 updates, several times fewer than on their datasets, and the
    per-epoch accuracy was still rising; at 40 it has flattened. (iii) Appendix F
    writes the posterior for hard judgments, prod_k C^k_{c, b_k}; for a judge that
    reports a distribution p_k we use prod_k prod_z (C^k_{c,z})^{p_k(z)}, which is
    that posterior when p_k is one-hot and the fractional likelihood the jury uses
    otherwise. It is fed the same eight-sample distributions as the jury.

Protocol. Theirs, which is more generous than the jury's: the model is fitted on
the judgments of all 1440 questions, test questions included -- "no distinction
between train and test sets" -- and no label is read in doing so. It therefore
does not depend on the split and is trained once. They choose lambda and the
checkpoint on 250 labelled development items; we report that (`dev`, the items
drawn from the training half of each split) and the fully label-free alternative
(`nodev`: the released lambda, the last epoch).

Every other rule is the one of the main comparison -- fitted on the full training
half of the same ten splits -- and is only *scored* on the multiple-choice test
questions, so nothing about the main benchmark changes. GSM8K is left out because
the method needs a fixed label space and an open numeric answer has none.

Fine-tuning GPT-2 is neither instant (about ten minutes a run on one GPU, sixteen
runs) nor bit-reproducible, so the per-epoch predictions of every run are kept in
results/skillaggregation_runs.npz and reused; `--retrain` ignores them.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]          # this supplementary folder
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from run_paper_results import (BOOT, SEEDS, TABLEDIR, bootstrap,   # noqa: E402
                               fmt, write)
from aggregator_vs_literature import (run_isp, run_majority,      # noqa: E402
                                      run_ow_i, run_ow_l,
                                      run_soft_equal, run_soft_ow)
from router_lab.techniques.aggregator import (answer_key,         # noqa: E402
                                              rollout_distributions)
from router_lab.techniques.base import (build_context,            # noqa: E402
                                        prediction_matrix)
from router_lab.techniques.multiseed import _resplit              # noqa: E402
from router_lab.techniques.runner import fit_router               # noqa: E402
from router_lab.utils import load_config                          # noqa: E402

LETTERS = ["A", "B", "C", "D"]
LAMBDAS = [0.0, 0.01, 0.1, 1.0]
LAMBDA_DEFAULT = 0.1         # `--reg_factor 0.1` in the released train.sh
# The released recipe (train.sh): GPT-2, batch 8, lr 1e-4, 10 epochs, cosine
# schedule with 3% warm-up, no weight decay, the whole encoder trained.
ENCODER, EPOCHS, BATCH, LR, WARMUP, MAX_LEN = "gpt2", 40, 8, 1e-4, 0.03, 512
N_DEV = 250                  # labelled development items, as in the paper
REF = "jury+rollouts"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# The model of Appendix F
# --------------------------------------------------------------------------- #
class SkillAggregation(torch.nn.Module):
    def __init__(self, n_judges, n_classes, contextual):
        super().__init__()
        from transformers import AutoModel
        self.llm = AutoModel.from_pretrained(ENCODER)
        dim = self.llm.config.hidden_size
        self.bottleneck = torch.nn.Linear(dim, n_classes)
        # The release starts every judge at skill 0.7; with C classes the other
        # 0.3 is spread over the C - 1 wrong answers.
        init = torch.full((n_classes, n_classes), 0.3 / (n_classes - 1))
        init.fill_diagonal_(0.7)
        self.theta = torch.nn.Parameter(torch.log(init).repeat(n_judges, 1, 1))
        self.skill = None
        if contextual:                                  # SkillAggregation-X
            self.skill = torch.nn.Linear(dim, n_judges * n_classes * n_classes)
            torch.nn.init.zeros_(self.skill.weight)
            torch.nn.init.zeros_(self.skill.bias)

    def encode(self, ids, att):
        """The final hidden state of the last real token, as in the release."""
        h = self.llm(input_ids=ids, attention_mask=att).last_hidden_state
        return h[torch.arange(len(h)), att.sum(-1) - 1]

    def forward(self, ids, att):
        h = self.encode(ids, att)
        s = torch.softmax(self.bottleneck(h), dim=-1)               # [n, C]
        theta = self.theta.unsqueeze(0)
        if self.skill is not None:
            # as in the release, the skill head reads the context but does not
            # train the encoder
            theta = theta + self.skill(h.detach()).view(len(h), *self.theta.shape)
        Cm = torch.softmax(theta, dim=-1).expand(len(h), -1, -1, -1)  # [n, K, C, C]
        return s, Cm, torch.einsum("nc,nkcb->nkb", s, Cm)           # judge estimates

    def loss(self, ids, att, Y, mask, lam):
        _, Cm, pred = self(ids, att)
        ce = -(Y * torch.log(pred.clamp_min(1e-8))).sum(-1)         # [n, K]
        ce = (ce * mask).sum() / mask.sum()
        reg = ((Cm[:, :, :-1, :-1] - Cm[:, :, -1:, :-1]) ** 2).sum((-1, -2)).mean()
        return ce + lam * reg

    @torch.no_grad()
    def posterior(self, ids, att, Y, mask):
        s, Cm, _ = self(ids, att)
        ll = torch.einsum("nkz,nkcz->nkc", Y, torch.log(Cm.clamp_min(1e-8)))
        return (torch.log(s.clamp_min(1e-8)) + (ll * mask.unsqueeze(-1)).sum(1)).argmax(-1)


def tokenise(texts):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ENCODER)
    tok.pad_token = tok.eos_token
    tok.truncation_side = "left"                        # keep the options, which come last
    return [tok(t, truncation=True, max_length=MAX_LEN)["input_ids"] for t in texts]


def _batch(tokens, rows):
    n = max(len(tokens[i]) for i in rows)
    ids = torch.zeros((len(rows), n), dtype=torch.long)
    att = torch.zeros_like(ids)
    for b, i in enumerate(rows):
        ids[b, :len(tokens[i])] = torch.tensor(tokens[i])
        att[b, :len(tokens[i])] = 1
    return ids.to(DEVICE), att.to(DEVICE)


def fit_skillaggregation(tokens, Y, mask, fit_rows, contextual, lam, seed):
    """-> [EPOCHS, n] predicted class for every row after each epoch.

    Trained on the judgments of `fit_rows` and on nothing else: no label is read.
    """
    from transformers import get_cosine_schedule_with_warmup
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Yt = torch.tensor(Y, dtype=torch.float32, device=DEVICE)
    Mt = torch.tensor(mask, dtype=torch.float32, device=DEVICE)
    model = SkillAggregation(Y.shape[1], Y.shape[2], contextual).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    total = EPOCHS * int(np.ceil(len(fit_rows) / BATCH))
    sched = get_cosine_schedule_with_warmup(opt, int(WARMUP * total), total)
    order_all = np.argsort([len(t) for t in tokens])    # length-sorted, for inference
    out = np.zeros((EPOCHS, len(tokens)), dtype=np.int64)
    for epoch in range(EPOCHS):
        model.train()
        perm = rng.permutation(fit_rows)
        for k in range(0, len(perm), BATCH):
            rows = perm[k:k + BATCH]
            ids, att = _batch(tokens, rows)
            opt.zero_grad()
            with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                loss = model.loss(ids, att, Yt[rows], Mt[rows], lam)
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        for k in range(0, len(order_all), 64):
            rows = order_all[k:k + 64]
            ids, att = _batch(tokens, rows)
            with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                out[epoch, rows] = model.posterior(ids, att, Yt[rows], Mt[rows]).cpu().numpy()
    return out


# --------------------------------------------------------------------------- #
def judgments(ctx, A, dist, rows):
    """[n, K, 4] hard and soft judge estimates over A-D, and the answered mask."""
    qids = ctx.df["qid"].to_numpy()
    ids = list(ctx.expert_ids)
    hard = np.zeros((len(rows), len(ids), 4))
    soft = np.zeros_like(hard)
    for n, i in enumerate(rows):
        for j, e in enumerate(ids):
            if A[i, j] in LETTERS:
                hard[n, j, LETTERS.index(A[i, j])] = 1.0
            for k, v in dist.get(qids[i], {}).get(e, {}).items():
                if k in LETTERS:
                    soft[n, j, LETTERS.index(k)] += v
            if soft[n, j].sum() <= 0:                   # no samples: the greedy answer
                soft[n, j] = hard[n, j]
            if soft[n, j].sum() > 0:
                soft[n, j] /= soft[n, j].sum()
    return hard, soft, (hard.sum(-1) > 0).astype(float)


def main() -> None:
    cfg = load_config(ROOT / "config.yaml")
    if "--cache" in sys.argv:
        cfg.raw["paths"]["cache_dir"] = sys.argv[sys.argv.index("--cache") + 1]
    cache = Path(cfg["paths"]["cache_dir"])
    df = pd.read_csv(cache / "questions.csv").fillna("")
    preds = pd.read_csv(cache / "predictions.csv")
    roll_dist, n_samples = rollout_distributions(cache, {e["id"] for e in cfg["experts"]})
    seeds = SEEDS[:int(sys.argv[sys.argv.index("--seeds") + 1])] if "--seeds" in sys.argv else SEEDS

    # Fine-tuning GPT-2 is the one thing in this folder that is neither
    # instant nor bit-reproducible, so every run's per-epoch predictions are kept
    # and reused; `--retrain` ignores them.
    runs_path = Path(cfg["paths"]["results_dir"]) / "skillaggregation_runs.npz"
    runs = {} if "--retrain" in sys.argv or not runs_path.exists() \
        else dict(np.load(runs_path))

    def run(key, *args):
        nonlocal tokens
        if key not in runs:
            if tokens is None:
                q = df.loc[df["task"] == "mc"]
                if q[["question", "A", "B", "C", "D"]].eq("").any().any():
                    raise ValueError(
                        "Question and choice text is omitted from the public "
                        "HellaSwag cache. Keep the shipped skillaggregation_runs.npz "
                        "or supply complete, lawfully obtained text with --cache "
                        "before retraining. See THIRD_PARTY_NOTICES.md."
                    )
                tokens = tokenise([
                    f"{r.question}\nA. {r.A}\nB. {r.B}\nC. {r.C}\nD. {r.D}\nAnswer:"
                    for r in q.itertuples()
                ])
            print(f"[skillaggregation] training {key}", flush=True)
            runs[key] = fit_skillaggregation(tokens, *args)
            np.savez_compressed(runs_path, **runs)
        return runs[key]

    perq = defaultdict(lambda: defaultdict(list))
    acc = defaultdict(list)
    estimated, true_acc = defaultdict(list), defaultdict(list)   # OW-L / OW-I accuracies
    chosen = defaultdict(list)
    tokens = None
    folds, families = [], {}

    for seed in seeds:
        d = df.copy()
        d["split"] = _resplit(d, cfg["dataset"]["train_frac"], seed)
        cfg.raw["seed"] = seed
        ctx = build_context(cfg, d, preds, cache)
        n_exp = len(ctx.expert_ids)
        A = np.vectorize(answer_key, otypes=[object])(
            np.repeat(ctx.tasks[:, None], n_exp, axis=1), prediction_matrix(ctx))
        tr, te = np.where(ctx.train_mask)[0], np.where(ctx.test_mask)[0]
        qids = ctx.df["qid"].to_numpy()
        is_mc = ctx.tasks == "mc"
        te_mc = is_mc[te]                                # which test rows are scored

        def record(name, correct, mc_only=False):
            c = np.asarray(correct) if mc_only else np.asarray(correct)[te_mc]
            acc[name].append(float(c.mean()))
            for q, y in zip(qids[te][te_mc], c):
                perq[name][q].append(int(y))

        # The rules of the main comparison, fitted exactly as there.
        record("majority", run_majority(ctx, A, tr, te)[0])
        record("ISP", run_isp(ctx, A, tr, te)[0])
        # Ai et al.'s two label-free accuracy estimators feeding Optimal Weight,
        # fitted on the unlabelled training half like ISP and the jury, and
        # (|transductive) on all 1440 questions as in their own protocol.
        for name, fn in (("OW-L", run_ow_l), ("OW-I", run_ow_i)):
            correct, _, x = fn(ctx, A, tr, te)
            record(name, correct)
            estimated[name].append([float(v) for v in x])
            record(name + "|transductive", fn(ctx, A, tr, te, fit=np.arange(len(A)))[0])
        true_acc["mc"].append([float(v) for v in ctx.Y[[i for i in tr if is_mc[i]]].mean(0)])
        record("soft-equal", run_soft_equal(ctx, A, tr, te, roll_dist, n_samples)[0])
        record("soft-OW", run_soft_ow(ctx, A, tr, te, roll_dist, n_samples)[0])
        for tag, use_roll in (("", False), ("+rollouts", True)):
            cfg.raw["experiments"]["jury"]["use_rollouts"] = use_roll
            cfg.raw["experiments"]["jury"]["lure"] = "none"
            record("jury" + tag, fit_router("jury", cfg, ctx, seed).predict(ctx).correct)

        # SkillAggregation, on the multiple-choice rows (the same rows, in the same
        # order, on every split).
        mc = np.where(is_mc)[0]
        pos = {i: n for n, i in enumerate(mc)}
        hard, soft, mask = judgments(ctx, A, roll_dist, mc)
        gold = np.array([LETTERS.index(g) for g in ctx.df["answer"].to_numpy()[mc]])
        tr_mc = np.array([pos[i] for i in tr if is_mc[i]])
        te_pos = np.array([pos[i] for i in te if is_mc[i]])
        folds.append((te_pos, qids[te][te_mc]))
        dev = np.random.default_rng(seed).choice(tr_mc, size=N_DEV, replace=False)

        protocols = [("transductive", np.arange(len(mc)), 0)]      # fitted once, no split
        for inp, Y in (("hard", hard), ("soft", soft)):
            for ctxl in (False, True):
                for proto, fit_rows, tseed in protocols:
                    name = f"SA{'-X' if ctxl else ''}|{inp}|{proto}"
                    P = {lam: run(f"{name}|lam{lam}|ep{EPOCHS}|seed{tseed}", Y, mask,
                                  fit_rows, ctxl, lam, tseed) for lam in LAMBDAS}
                    # no development set: the released lambda, the last epoch
                    record(name + "|nodev", (P[LAMBDA_DEFAULT][-1] == gold)[te_pos],
                           mc_only=True)
                    # the paper's selection: lambda and epoch on 250 labelled items
                    lam, ep = max(((l, e) for l in LAMBDAS for e in range(EPOCHS)),
                                  key=lambda le: ((P[le[0]][le[1]] == gold)[dev].mean(),
                                                  -le[1]))
                    record(name + "|dev", (P[lam][ep] == gold)[te_pos], mc_only=True)
                    chosen[name].append((lam, ep + 1))
                    families[name] = P
        print(f"[seed {seed}] " + "  ".join(
            f"{k}={100*v[-1]:.1f}" for k, v in acc.items()
            if k in ("soft-equal", REF, "SA|soft|transductive|dev",
                     "SA-X|soft|transductive|dev")), flush=True)

    # The best of the 40 epochs: lambda and the epoch chosen *on the test
    # questions*, one choice for all ten splits. No protocol could deliver this;
    # it answers "what is the best this model ever did here", and the table
    # reports it as such (`Labels` = test).
    posthoc = {}
    for name, P in families.items():
        lam, ep = max(((l, e) for l in LAMBDAS for e in range(EPOCHS)),
                      key=lambda le: np.mean([(P[le[0]][le[1]] == gold)[t].mean()
                                              for t, _ in folds]))
        for t, q in folds:
            c = (P[lam][ep] == gold)[t]
            acc[name + "|posthoc"].append(float(c.mean()))
            for qq, y in zip(q, c):
                perq[name + "|posthoc"][qq].append(int(y))
        posthoc[name] = (lam, ep + 1)

    # A rule fitted per split records one score per test appearance of a question;
    # bootstrap() averages them, so every question counts once.
    boot = bootstrap(perq, REF)
    # The greedy label-free rules against the greedy jury: same input, so the
    # difference is the reliability estimator alone.
    greedy = ("majority", "ISP", "OW-L", "OW-I", "OW-L|transductive", "OW-I|transductive")
    boot_greedy = bootstrap({k: perq[k] for k in greedy + ("jury",)}, "jury")
    print(f"\n=== multiple-choice questions only ({len(perq[REF])} questions) ===\n")
    for k in acc:
        b = boot.get(k)
        tail = f"   jury {100*b['delta']:+.1f} [{100*b['lo']:+.1f}, {100*b['hi']:+.1f}]" if b else ""
        if k in boot_greedy:
            g = boot_greedy[k]
            tail += f"   greedy jury {100*g['delta']:+.1f} [{100*g['lo']:+.1f}, {100*g['hi']:+.1f}]"
        print(f"{k:<36}{100*np.mean(acc[k]):6.1f} +- {100*np.std(acc[k]):.1f}{tail}")

    # ----------------------------------------------------------------- table
    def row(label, labels, inp, key, oracle=False):
        if key not in acc:
            return None
        b = boot.get(key)
        delta = ""
        if b and oracle:
            # lambda and the epoch were selected on these very questions, so a
            # post-selection interval would not be a normal one: the point
            # difference alone, as a best-observed reference.
            delta = f"{100*b['delta']:+.1f}"
        elif b:
            star = "$^{*}$" if b["p"] < 0.05 else ""
            delta = (f"{100*b['delta']:+.1f}{star}~[{100*b['lo']:+.1f}, "
                     f"{100*b['hi']:+.1f}]")
        return (f"{label} & {labels} & {inp} & {fmt(np.mean(acc[key]), np.std(acc[key]))}"
                f" & {delta} \\\\")

    sa = "\\cite{sun-etal-2025-skillaggregation}"
    lines = ["\\begin{tabular}{lllrr}", "\\toprule",
             "Method & Labels & Input & Accuracy & $\\Delta$ vs.\\ jury~[95\\% CI] \\\\",
             "\\midrule",
             row("Majority vote", "no", "greedy", "majority"),
             row("Inverse Surprising Popularity (ISP)", "no", "greedy", "ISP"),
             row("Optimal Weight, accuracies learned (OW-L)", "no", "greedy", "OW-L"),
             row("Optimal Weight, ISP pseudo-labels (OW-I)", "no", "greedy", "OW-I"),
             row("Equal-weight pool", "no", "distributions", "soft-equal"),
             "\\midrule",
             row(f"SkillAggregation~{sa}", "no", "greedy", "SA|hard|transductive|nodev"),
             row("\\quad + answer distributions", "no", "distributions",
                 "SA|soft|transductive|nodev"),
             row("\\quad\\quad + development set", "250", "distributions",
                 "SA|soft|transductive|dev"),
             row("\\quad\\quad + best observed epoch and $\\lambda$ (test oracle)",
                 "test", "distributions", "SA|soft|transductive|posthoc", oracle=True),
             row(f"SkillAggregation-X~{sa}", "no", "distributions",
                 "SA-X|soft|transductive|nodev"),
             row("\\quad + development set", "250", "distributions",
                 "SA-X|soft|transductive|dev"),
             row("\\quad + best observed epoch and $\\lambda$ (test oracle)",
                 "test", "distributions", "SA-X|soft|transductive|posthoc", oracle=True),
             "\\midrule",
             row("Supervised linear fusion", "yes", "distributions", "soft-OW"),
             row("\\textbf{Jury (ours), greedy answers}", "\\textbf{no}", "greedy", "jury"),
             row("\\textbf{\\quad + repeated sampling}", "\\textbf{no}", "distributions", REF),
             "\\bottomrule", "\\end{tabular}"]
    write(TABLEDIR / "skillaggregation.tex", "\n".join(l for l in lines if l))

    out = Path(cfg["paths"]["results_dir"]) / "skillaggregation_mc.json"
    out.write_text(json.dumps({
        "seeds": seeds, "n_questions": len(perq[REF]), "boot": BOOT,
        "lambdas": LAMBDAS, "epochs": EPOCHS, "encoder": ENCODER, "n_dev": N_DEV,
        "accuracy": dict(acc), "bootstrap_vs_jury": boot,
        "bootstrap_vs_greedy_jury": boot_greedy,
        "experts": [e["id"] for e in cfg["experts"]],
        "ow_estimated_accuracies": dict(estimated),
        "train_accuracies_mc": true_acc["mc"],
        "chosen": {k: v for k, v in chosen.items()}, "posthoc": posthoc}, indent=2))
    print(f"[results] -> {out}")


if __name__ == "__main__":
    main()
