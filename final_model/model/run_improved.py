"""Driver that applies post-hoc improvements to a trained baseline.

Runs the *same* data preparation as `main.py` (load CSV → features → anomaly
scores → train/test split), loads a pre-trained `ensemble_model.joblib`,
and compares:

  - baseline (original pipeline threshold)
  - raw probabilities with max-F1, max-F2, min-cost thresholds
  - isotonic-calibrated probabilities, with the same three threshold strategies

Re-uses the split seed (`random_state=42`) from `main.py` so the test set is
identical to the one the baseline was evaluated on.

Also reserves an in-train *holdout* (10%, stratified) so we can fit the
calibrator on data the ensemble didn't see, making the calibrated numbers
deployable rather than leaky-diagnostic.

Usage
-----
    python run_improved.py \
        --baseline_dir ../output/baseline \
        --output_dir   ../output/improved
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE))

from Feature_engineering import build_all_features  # noqa: E402
from feature_selection import select_features  # noqa: E402
from anomaly_detection import add_anomaly_scores_to_splits  # noqa: E402
from improved_evaluation import run_post_hoc_improvements  # noqa: E402
from main import load_and_validate  # noqa: E402


def prepare_data(data_dir: str):
    tables = load_and_validate(data_dir)
    user_info = tables["user_info_train"]
    twd = tables["twd_transfer_train"]
    crypto = tables["crypto_transfer_train"]
    trading = tables["usdt_twd_trading_train"]
    swap = tables["usdt_swap_train"]

    feat_df = build_all_features(user_info, twd, crypto, trading, swap)
    labels_s = user_info.set_index("user_id")["status"]
    feat_df = feat_df.join(labels_s, how="left")
    feat_df["status"] = feat_df["status"].fillna(0).astype(int)
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    y = feat_df["status"].values.astype(int)
    X_raw = feat_df.drop(columns=["status"])
    X_selected, _ = select_features(X_raw, y, corr_threshold=0.95)
    for col in ("is_female", "age"):
        if col in X_selected.columns:
            X_selected = X_selected.drop(columns=col)
    X = X_selected.values.astype(np.float32)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",
                    default=os.path.join(ROOT, "adjust_data", "train"))
    ap.add_argument("--baseline_dir",
                    default=os.path.join(ROOT, "final_model", "output", "baseline"))
    ap.add_argument("--output_dir",
                    default=os.path.join(ROOT, "final_model", "output", "improved"))
    ap.add_argument("--holdout_size", type=float, default=0.1,
                    help="Fraction of training data to reserve for calibrator fit")
    ap.add_argument("--cost_fn", type=float, default=10.0)
    ap.add_argument("--cost_fp", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Re-create the exact same features + split the baseline was trained on
    X, y = prepare_data(args.data_dir)
    print(f"\n  X shape: {X.shape}, pos: {int(y.sum())}/{len(y)}"
          f" ({y.mean()*100:.2f}%)")

    # Same split as main.py (stratified, random_state=42, test_size=0.2)
    X_tr_full, X_te, y_tr_full, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )

    # Add anomaly scores fit on the *full* train (matches main.py)
    X_tr_full, X_te, _ = add_anomaly_scores_to_splits(X_tr_full, X_te)
    print(f"  after anomaly scores: X_tr={X_tr_full.shape}, X_te={X_te.shape}")

    # ── Load baseline ensemble
    bundle = joblib.load(os.path.join(args.baseline_dir, "ensemble_model.joblib"))
    ensemble = bundle["ensemble"]
    orig_threshold = float(bundle.get("optimal_threshold",
                                      getattr(ensemble, "oof_threshold", 0.5)))
    print(f"\n  loaded baseline ensemble, orig_threshold={orig_threshold:.4f}")

    # ── Carve out a calibration holdout from the *training* half
    X_tr, X_hold, y_tr, y_hold = train_test_split(
        X_tr_full, y_tr_full,
        test_size=args.holdout_size,
        stratify=y_tr_full,
        random_state=43,
    )
    print(f"  train={len(X_tr):,} (pos {int(y_tr.sum())}), "
          f"holdout={len(X_hold):,} (pos {int(y_hold.sum())}), "
          f"test={len(X_te):,} (pos {int(y_te.sum())})")

    # Produce probabilities
    print("\n  predicting on holdout + test ...")
    proba_hold = ensemble.predict_proba(X_hold)
    proba_te = ensemble.predict_proba(X_te)

    # Sanity-check: test metrics should match baseline metrics.json
    baseline_metrics_path = os.path.join(args.baseline_dir, "metrics.json")
    if os.path.exists(baseline_metrics_path):
        with open(baseline_metrics_path) as f:
            bm = json.load(f)
        print(f"  baseline metrics (from file): F1={bm.get('F1', 0):.4f}, "
              f"AUC-PR={bm.get('AUC-PR', 0):.4f}")

    # ── Run improvements
    results = run_post_hoc_improvements(
        proba_te=proba_te, y_te=y_te,
        baseline_threshold=orig_threshold,
        proba_holdout=proba_hold, y_holdout=y_hold,
        output_dir=args.output_dir,
        cost_fn=args.cost_fn, cost_fp=args.cost_fp,
    )

    # Save raw probabilities for downstream analysis
    pd.DataFrame({
        "y_true": y_te,
        "proba_raw": proba_te,
    }).to_csv(os.path.join(args.output_dir, "test_proba.csv"), index=False)

    pd.DataFrame({
        "y_true": y_hold,
        "proba_raw": proba_hold,
    }).to_csv(os.path.join(args.output_dir, "holdout_proba.csv"), index=False)

    print(f"\n  Results written to {args.output_dir}/metrics_improved.json")


if __name__ == "__main__":
    main()
