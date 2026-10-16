"""Shared plumbing for routing experiments.

Every experiment is a self-contained package exposing a `BaseRouter` subclass.
They all consume the same `RoutingContext` (embeddings + cached expert-correctness
matrix + metadata) and return a `RoutingResult` (per-test-question correctness and
compute cost), so the runner can compare them on equal footing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features import compute_embeddings, correctness_matrix


@dataclass
class RoutingContext:
    expert_ids: list
    params_b: np.ndarray            # [n_experts]
    emb: np.ndarray                 # [n_questions, d]
    Y: np.ndarray                   # [n_questions, n_experts] correctness 0/1
    train_mask: np.ndarray
    test_mask: np.ndarray
    tasks: np.ndarray               # per-question task type
    subjects: np.ndarray
    df: pd.DataFrame
    preds: pd.DataFrame
    cfg: dict

    # convenience views
    @property
    def Xtr(self): return self.emb[self.train_mask]
    @property
    def Ytr(self): return self.Y[self.train_mask]
    @property
    def Xte(self): return self.emb[self.test_mask]
    @property
    def Yte(self): return self.Y[self.test_mask]
    @property
    def tasks_te(self): return self.tasks[self.test_mask]


@dataclass
class RoutingResult:
    name: str
    correct: np.ndarray             # 0/1 per test question
    cost: np.ndarray                # params-B "consumed" per test question
    chosen: np.ndarray = None       # chosen expert index per test question (selection routers)
    note: str = ""
    calls: np.ndarray = None        # worker calls per question (workflow routers)


class BaseRouter:
    name = "base"
    kind = "selection"              # "selection" or "combination"

    def __init__(self, cfg):
        self.cfg = cfg
        self.rcfg = cfg.get("experiments", {}).get(self.name, {}) if hasattr(cfg, "get") else {}

    def fit(self, ctx: RoutingContext):
        raise NotImplementedError

    def predict(self, ctx: RoutingContext) -> RoutingResult:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def build_context(cfg, df, preds, cache_dir) -> RoutingContext:
    expert_ids = [e["id"] for e in cfg["experts"]]
    params_b = np.asarray([float(e["params_b"]) for e in cfg["experts"]], dtype=np.float32)
    emb = compute_embeddings(cfg, df, cache_dir)
    Y = correctness_matrix(preds, expert_ids, df)
    return RoutingContext(
        expert_ids=expert_ids,
        params_b=params_b,
        emb=np.asarray(emb, dtype=np.float32),
        Y=Y,
        train_mask=(df["split"] == "train").to_numpy(),
        test_mask=(df["split"] == "test").to_numpy(),
        tasks=df["task"].to_numpy(),
        subjects=df["subject"].to_numpy(),
        df=df,
        preds=preds,
        cfg=cfg,
    )


def l2norm(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def fit_rbf_centroids(Xtr, n_clusters, seed=0):
    """KMeans centroids on L2-normalised train embeddings (for RBF features)."""
    from sklearn.cluster import KMeans
    n_clusters = min(int(n_clusters), len(Xtr))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(l2norm(Xtr))
    return l2norm(km.cluster_centers_)


def rbf_features(X, C, gamma):
    """RBF-to-centroid features: phi(x)_c = exp(-gamma*(1 - cos(x, mu_c)))."""
    return np.exp(-gamma * (1.0 - l2norm(X) @ C.T)).astype(np.float32)


def knn_regional_accuracy(Xtr, Ytr, Xq, k) -> np.ndarray:
    """Predict each expert's success prob for queries as its mean correctness
    over the k nearest training questions (cosine similarity)."""
    A, B = l2norm(Xtr), l2norm(Xq)
    sim = B @ A.T
    k = min(k, A.shape[0])
    idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    return Ytr[idx].mean(axis=1)


def prediction_matrix(ctx: RoutingContext) -> np.ndarray:
    """[n_questions, n_experts] of raw expert answers (as strings), df-aligned."""
    wide = ctx.preds.pivot_table(index="qid", columns="expert_id", values="pred",
                                 aggfunc="first")
    wide = wide.reindex(index=ctx.df["qid"].values, columns=ctx.expert_ids)
    return wide.to_numpy(dtype=object)


def reward_matrix(ctx: RoutingContext) -> np.ndarray:
    """[n_questions, n_experts] worker rewards used to train soft routers.

    A fresh Fugu rollout pass stores the mean reward from repeated samples in
    ``reward_mean``.  Every other pipeline remains compatible with the original
    one-answer cache: when that column is absent, binary correctness is the
    single observed reward.
    """
    if "reward_mean" not in ctx.preds.columns:
        return ctx.Y.astype(np.float32, copy=True)
    wide = ctx.preds.pivot_table(index="qid", columns="expert_id", values="reward_mean",
                                 aggfunc="first")
    wide = wide.reindex(index=ctx.df["qid"].values, columns=ctx.expert_ids)
    out = wide.to_numpy(dtype=np.float32)
    return np.where(np.isnan(out), ctx.Y, out).astype(np.float32)


def selfconf_matrix(ctx: RoutingContext) -> np.ndarray:
    """[n_questions, n_experts] of each expert's post-hoc self-confidence, df-aligned.

    NaN where unavailable (e.g. numeric questions, or if the `confidence` column
    was never cached). Callers must handle NaN (e.g. fall back to a regional estimate).
    """
    if "confidence" not in ctx.preds.columns:
        return np.full((len(ctx.df), len(ctx.expert_ids)), np.nan, dtype=np.float32)
    wide = ctx.preds.pivot_table(index="qid", columns="expert_id", values="confidence",
                                 aggfunc="first")
    wide = wide.reindex(index=ctx.df["qid"].values, columns=ctx.expert_ids)
    return wide.to_numpy(dtype=np.float32)
