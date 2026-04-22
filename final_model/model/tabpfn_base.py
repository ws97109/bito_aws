"""TabPFN v2 as a 4th/5th base learner in the stacking ensemble.

TabPFN's sweet spot is <=10k rows — for our 40k training set we subsample
per-fold, preserving all positives and stratifying the negatives. This also
aligns with the paper's recommendation for extreme imbalance (keep the rare
class intact, downsample the common class).

Design choices:
  - subsample size: up to 8,000 per fold (TabPFN v2.5 limit ≈ 10k but we leave
    slack for memory; benchmarks show the gain plateaus around 5k–10k anyway)
  - all positives are kept; negatives are sampled without replacement
  - CPU fallback is fine — a 5-fold OOF on 8k×features completes in a few minutes
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def _subsample_for_tabpfn(
    X: np.ndarray, y: np.ndarray, max_rows: int = 8000, seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a stratified subsample keeping all positives.

    Returns (X_sub, y_sub, idx_sub) where idx_sub are indices into the
    original arrays (handy if the caller wants to track which rows were kept).
    """
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    neg_budget = max(0, max_rows - len(pos_idx))
    if neg_budget < len(neg_idx):
        neg_sample = rng.choice(neg_idx, size=neg_budget, replace=False)
    else:
        neg_sample = neg_idx

    idx = np.concatenate([pos_idx, neg_sample])
    rng.shuffle(idx)
    return X[idx], y[idx], idx


def tabpfn_oof_probabilities(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    max_rows_per_fold: int = 8000,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, dict]:
    """5-fold OOF probabilities from TabPFN with per-fold subsampling."""
    from tabpfn import TabPFNClassifier

    y = np.asarray(y).astype(int)
    oof = np.zeros(len(y), dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)

    fold_scores = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val = X[val_idx]

        # Subsample training partition
        X_tr_sub, y_tr_sub, _ = _subsample_for_tabpfn(
            X_tr, y_tr, max_rows=max_rows_per_fold, seed=fold,
        )

        scaler = StandardScaler().fit(X_tr_sub)
        X_tr_sub_s = scaler.transform(X_tr_sub)
        X_val_s = scaler.transform(X_val)

        clf = TabPFNClassifier(
            device="cpu",
            ignore_pretraining_limits=True,
            random_state=fold,
        )
        if verbose:
            print(f"  TabPFN fold {fold+1}/{n_splits}: "
                  f"fitting on {len(X_tr_sub)} rows "
                  f"(pos={int(y_tr_sub.sum())}) ...")
        clf.fit(X_tr_sub_s, y_tr_sub)
        proba = clf.predict_proba(X_val_s)[:, 1]
        oof[val_idx] = proba

        from sklearn.metrics import average_precision_score
        fold_auc = float(average_precision_score(y[val_idx], proba))
        fold_scores.append(fold_auc)
        if verbose:
            print(f"    fold AUC-PR={fold_auc:.4f}")

    info = {
        "n_splits": n_splits,
        "max_rows_per_fold": max_rows_per_fold,
        "fold_auc_pr": fold_scores,
        "mean_auc_pr": float(np.mean(fold_scores)),
    }
    if verbose:
        print(f"  TabPFN mean AUC-PR = {info['mean_auc_pr']:.4f}")
    return oof, info


def tabpfn_fit_predict_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    max_rows: int = 8000,
    seed: int = 0,
) -> np.ndarray:
    """Fit once on subsampled train, predict on test."""
    from tabpfn import TabPFNClassifier

    X_sub, y_sub, _ = _subsample_for_tabpfn(X_train, y_train,
                                             max_rows=max_rows, seed=seed)
    scaler = StandardScaler().fit(X_sub)
    clf = TabPFNClassifier(
        device="cpu",
        ignore_pretraining_limits=True,
        random_state=seed,
    )
    clf.fit(scaler.transform(X_sub), y_sub)
    return clf.predict_proba(scaler.transform(X_test))[:, 1].astype(np.float32)
