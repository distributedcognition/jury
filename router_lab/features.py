"""Shared features used by every technique: question embeddings and the cached
expert-correctness matrix. Kept separate so no single technique owns them."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def compute_embeddings(cfg, df: pd.DataFrame, cache_dir: Path) -> np.ndarray:
    """Sentence-embedding of each question (+ its options), cached to disk."""
    emb_path = cache_dir / "embeddings.npy"
    if emb_path.exists():
        emb = np.load(emb_path)
        if len(emb) == len(df):
            print(f"[features] reusing cached embeddings -> {emb_path}")
            return emb

    if df["question"].fillna("").str.strip().eq("").any():
        raise ValueError(
            "Question text is omitted from the public HellaSwag cache. Keep the "
            "shipped embeddings.npy or supply complete, lawfully obtained text "
            "before regenerating embeddings. See THIRD_PARTY_NOTICES.md."
        )

    from sentence_transformers import SentenceTransformer

    print(f"[features] embedding {len(df)} questions with {cfg['router']['embedding_model']}")
    model = SentenceTransformer(cfg["router"]["embedding_model"])
    texts = [f"{r['question']} {r['A']} {r['B']} {r['C']} {r['D']}" for _, r in df.iterrows()]
    emb = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    np.save(emb_path, emb)
    return np.asarray(emb, dtype=np.float32)


def correctness_matrix(preds: pd.DataFrame, expert_ids: list[str], df: pd.DataFrame) -> np.ndarray:
    """Y[n_questions, n_experts] of 0/1 correctness, aligned with df order."""
    wide = preds.pivot_table(index="qid", columns="expert_id", values="correct")
    wide = wide.reindex(index=df["qid"].values, columns=expert_ids)
    return wide.to_numpy(dtype=np.float32)
