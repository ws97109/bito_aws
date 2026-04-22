"""PU learning base learner for the BitoGuard ensemble.

Treats the training set's negatives as *unlabelled* (a mix of true negatives
and un-labelled positives), rather than trusting them as clean negatives.
This is a principled response to the realistic AML setting where blacklist
labels only cover users who were actively investigated — the rest may still
contain hidden fraudsters.

We use sklearn-compatible PU wrappers from `pulearn`:
  - BaggingPuClassifier: re-samples unlabelled data per bag, aggregates.
    Robust on moderate datasets and the best default for this task.
  - ElkanotoPuClassifier: the classical Elkan-Noto 2008 method, assumes
    SCAR (selected completely at random) — less robust but a good diversity
    source in a stacked ensemble.

Output: OOF probabilities (5-fold) aligned with the original labels, so they
can be fed into the StackingEnsemble meta-learner alongside XGB/LGB/Cat.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def _build_pu_base() -> BaseEstimator:
    """Inner classifier used inside the PU wrapper. RF is the usual default
    because it copes with the asymmetric label noise PU induces better than a
    linear model, and is fast enough for bagging."""
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=0,
    )


def pu_oof_probabilities(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "bagging",
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, dict]:
    """Produce out-of-fold PU probabilities for every row in X.

    The labelling convention expected by `pulearn`:
      +1  = positive (observed blacklist)
      -1  = unlabelled (everything else)

    We translate our {0, 1} labels into {-1, +1} per-fold, run the PU
    classifier, and convert the scores back to a [0, 1] range for stacking.
    """
    from pulearn import BaggingPuClassifier, ElkanotoPuClassifier

    y = np.asarray(y).astype(int)
    oof = np.zeros(len(y), dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)

    fold_scores = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr = X[tr_idx]
        y_tr = y[tr_idx].copy()
        # Transform to {-1, +1}. In PU, "0" (unknown) is treated as -1.
        y_tr_pu = np.where(y_tr == 1, 1, -1).astype(int)

        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_val_s = scaler.transform(X[val_idx])

        if method == "bagging":
            clf = BaggingPuClassifier(
                estimator=_build_pu_base(),
                n_estimators=15,
                n_jobs=-1,
                random_state=fold,
            )
        elif method == "elkanoto":
            clf = ElkanotoPuClassifier(
                estimator=_build_pu_base(),
                hold_out_ratio=0.2,
            )
        else:
            raise ValueError(f"unknown PU method: {method!r}")

        clf.fit(X_tr_s, y_tr_pu)
        # `predict_proba` returns a 2-column array for both wrappers; we take
        # the positive-class column.
        try:
            raw = clf.predict_proba(X_val_s)
            scores = raw[:, 1] if raw.ndim == 2 else raw
        except (AttributeError, NotImplementedError):
            # Older Elkanoto exposes only `decision_function`
            scores = clf.predict_proba(X_val_s)

        # Ensure [0, 1] range (Elkanoto can return slightly out-of-range
        # values after its p-correction).
        scores = np.clip(scores, 0.0, 1.0)
        oof[val_idx] = scores

        fold_auc = _fold_pr_auc(y[val_idx], scores)
        fold_scores.append(fold_auc)
        if verbose:
            print(f"  PU({method}) fold {fold+1}/{n_splits}: "
                  f"AUC-PR={fold_auc:.4f}")

    info = {
        "method": method,
        "n_splits": n_splits,
        "fold_auc_pr": [float(s) for s in fold_scores],
        "mean_auc_pr": float(np.mean(fold_scores)),
    }
    if verbose:
        print(f"  PU({method}) mean AUC-PR = {info['mean_auc_pr']:.4f}")
    return oof, info


def _fold_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def pu_full_fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    method: str = "bagging",
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit once on all training data, produce test predictions.

    Used at inference time (after OOF for stacking is produced separately).
    Returns probability arrays on train and test.
    """
    from pulearn import BaggingPuClassifier, ElkanotoPuClassifier

    y_pu = np.where(np.asarray(y_train) == 1, 1, -1).astype(int)
    scaler = StandardScaler().fit(X_train)
    X_tr_s = scaler.transform(X_train)
    X_te_s = scaler.transform(X_test)

    if method == "bagging":
        clf = BaggingPuClassifier(
            estimator=_build_pu_base(),
            n_estimators=15,
            n_jobs=-1,
            random_state=random_state,
        )
    elif method == "elkanoto":
        clf = ElkanotoPuClassifier(
            estimator=_build_pu_base(),
            hold_out_ratio=0.2,
        )
    else:
        raise ValueError(f"unknown PU method: {method!r}")

    clf.fit(X_tr_s, y_pu)
    try:
        train_scores = clf.predict_proba(X_tr_s)
        test_scores = clf.predict_proba(X_te_s)
        if train_scores.ndim == 2:
            train_scores = train_scores[:, 1]
            test_scores = test_scores[:, 1]
    except (AttributeError, NotImplementedError):
        train_scores = clf.predict_proba(X_tr_s)
        test_scores = clf.predict_proba(X_te_s)

    return (
        np.clip(train_scores, 0.0, 1.0).astype(np.float32),
        np.clip(test_scores, 0.0, 1.0).astype(np.float32),
    )
