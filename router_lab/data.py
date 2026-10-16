"""Build a MULTI-TASK benchmark that mixes two domains with genuinely
complementary expert strengths:

  * mmlu   - knowledge/humanities multiple-choice questions (general model wins)
  * gsm8k  - grade-school math word problems, numeric answer (math model wins)

Each row carries a `task` ("mc" or "numeric") so experts know how to answer it
and how correctness is scored. The `split` column marks router-train vs benchmark.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

LETTERS = ["A", "B", "C", "D"]


def _clean_num(s: str) -> str:
    return s.replace(",", "").replace("$", "").strip().rstrip(".")


def build_dataset(cfg, cache_dir: Path) -> pd.DataFrame:
    from datasets import load_dataset  # only needed to rebuild the cache
    out_path = cache_dir / "questions.csv"
    if out_path.exists():
        print(f"[data] reusing cached dataset -> {out_path}")
        return pd.read_csv(out_path).fillna("")

    dcfg = cfg["dataset"]
    rng = np.random.default_rng(cfg["seed"])
    rows = []

    # ---- MMLU knowledge (multiple choice) --------------------------------- #
    mmlu = dcfg["tasks"]["mmlu"]
    for subject in mmlu["subjects"]:
        print(f"[data] downloading MMLU subject: {subject}")
        ds = load_dataset("cais/mmlu", subject, split="test")
        n = min(mmlu["max_per_subject"], len(ds))
        for i in rng.choice(len(ds), size=n, replace=False):
            ex = ds[int(i)]
            if len(ex["choices"]) != 4:
                continue
            c = ex["choices"]
            rows.append({"task": "mc", "subject": subject, "question": ex["question"],
                         "A": c[0], "B": c[1], "C": c[2], "D": c[3],
                         "answer": LETTERS[ex["answer"]]})

    # ---- GSM8K math (numeric answer) -------------------------------------- #
    gsm = dcfg["tasks"]["gsm8k"]
    print("[data] downloading GSM8K")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    n = min(gsm["max_examples"], len(ds))
    for i in rng.choice(len(ds), size=n, replace=False):
        ex = ds[int(i)]
        gold = _clean_num(ex["answer"].split("####")[-1])
        rows.append({"task": "numeric", "subject": "gsm8k", "question": ex["question"],
                     "A": "", "B": "", "C": "", "D": "", "answer": gold})

    # ---- ARC-Challenge and HellaSwag (multiple choice, same A-D shape) ----- #
    # Both are four-way multiple choice with a binary score, which is what lets
    # them share the benchmark's `mc` task, its prompt and its scorer. ARC keeps
    # only the four-choice items -- a handful carry three or five, and reshaping
    # them would change the answer space the emission model assumes.
    arc = dcfg["tasks"].get("arc")
    if arc:
        print("[data] downloading ARC-Challenge")
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        picked = 0
        for i in rng.permutation(len(ds)):
            ex = ds[int(i)]
            ch, lab = ex["choices"]["text"], ex["choices"]["label"]
            if len(ch) != 4 or ex["answerKey"] not in lab:
                continue
            rows.append({"task": "mc", "subject": "arc", "question": ex["question"],
                         "A": ch[0], "B": ch[1], "C": ch[2], "D": ch[3],
                         "answer": LETTERS[lab.index(ex["answerKey"])]})
            picked += 1
            if picked >= arc["max_examples"]:
                break

    hs = dcfg["tasks"].get("hellaswag")
    if hs:
        print("[data] downloading HellaSwag")
        ds = load_dataset("Rowan/hellaswag", split="validation")
        picked = 0
        for i in rng.permutation(len(ds)):
            ex = ds[int(i)]
            if len(ex["endings"]) != 4 or not str(ex["label"]).isdigit():
                continue
            rows.append({"task": "mc", "subject": "hellaswag",
                         "question": ex["ctx"] + " ...", "A": ex["endings"][0],
                         "B": ex["endings"][1], "C": ex["endings"][2],
                         "D": ex["endings"][3],
                         "answer": LETTERS[int(ex["label"])]})
            picked += 1
            if picked >= hs["max_examples"]:
                break

    df = pd.DataFrame(rows).reset_index(drop=True)
    df["qid"] = df.index

    # Stratified split per subject into router-train / benchmark-test.
    train_frac = dcfg["train_frac"]
    split = np.empty(len(df), dtype=object)
    for _, grp in df.groupby("subject"):
        ids = grp.index.to_numpy().copy()
        rng.shuffle(ids)
        cut = int(round(len(ids) * train_frac))
        split[ids[:cut]] = "train"
        split[ids[cut:]] = "test"
    df["split"] = split

    df.to_csv(out_path, index=False)
    by_sub = df.groupby("subject").size().to_dict()
    print(f"[data] built {len(df)} questions "
          f"({(df.task == 'mc').sum()} mc + {(df.task == 'numeric').sum()} numeric; "
          f"{by_sub}) "
          f"| {(df.split == 'train').sum()} train / {(df.split == 'test').sum()} test -> {out_path}")
    return df


def format_prompt(row) -> str:
    if row["task"] == "numeric":
        return (
            "Solve the following math problem. Think step by step, then on the last "
            "line write 'The final answer is <number>'.\n\n"
            f"Problem: {row['question']}"
        )
    return (
        "Answer the following multiple-choice question. "
        "Reply with ONLY the letter (A, B, C, or D) of the correct answer.\n\n"
        f"Question: {row['question']}\n"
        f"A. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}\n"
        "Answer:"
    )


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_number(text: str) -> str | None:
    text = text.replace(",", "")
    m = re.findall(r"answer is[^\d\-]*(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        return m[-1]
    nums = _NUM_RE.findall(text)
    return nums[-1] if nums else None


def is_correct(task: str, pred, gold) -> int:
    if pred is None or (isinstance(pred, float) and np.isnan(pred)):
        return 0
    if task == "numeric":
        try:
            return int(abs(float(_clean_num(str(pred))) - float(_clean_num(str(gold)))) < 1e-4)
        except (ValueError, TypeError):
            return 0
    return int(str(pred).strip() == str(gold).strip())
