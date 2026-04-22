"""Final improved evaluation pipeline.

Given a trained baseline (from `main.py`), produce:
  - Raw probabilities on the test set
  - Threshold strategies:
      max_f1  : original F1-optimal (what main.py does)
      max_f2  : F2-optimal (favours recall 2:1 over precision)
      min_cost: minimises C_FN×FN + C_FP×FP (business-cost optimal)
  - Sigmoid (Platt) calibration — rank-preserving, only useful if downstream
    consumers need calibrated probabilities; F1 unchanged vs max_f1 on raw.
  - A submission CSV per threshold strategy.

Everything is logged to `<output_dir>/FINAL_REPORT.json` for auditability.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE))

from Feature_engineering import build_all_features  # noqa: E402
from anomaly_detection import add_anomaly_scores_to_splits  # noqa: E402
from feature_selection import select_features  # noqa: E402
from improved_evaluation import (  # noqa: E402
    evaluate_at_threshold,
    evaluate_proba,
    threshold_max_f1,
    threshold_max_fbeta,
    threshold_min_cost,
)
from main import load_and_validate, load_predict_data  # noqa: E402


def build_features(tables):
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
    return X_selected, y, feat_df.index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",
                    default=os.path.join(ROOT, "adjust_data", "train"))
    ap.add_argument("--predict_dir",
                    default=os.path.join(ROOT, "adjust_data", "predict"))
    ap.add_argument("--baseline_dir",
                    default=os.path.join(ROOT, "final_model", "output", "baseline"))
    ap.add_argument("--output_dir",
                    default=os.path.join(ROOT, "final_model", "output", "final"))
    ap.add_argument("--cost_fn", type=float, default=10.0)
    ap.add_argument("--cost_fp", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    report = {
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_dir": args.baseline_dir,
        "cost": {"c_fn": args.cost_fn, "c_fp": args.cost_fp},
    }

    # ── 1. Data
    tables = load_and_validate(args.data_dir)
    X_selected, y, all_user_ids = build_features(tables)
    X = X_selected.values.astype(np.float32)
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, np.arange(len(y)), test_size=0.2, stratify=y, random_state=42,
    )
    X_tr, X_te, anomaly = add_anomaly_scores_to_splits(X_tr, X_te)

    # ── 2. Load baseline
    bundle = joblib.load(os.path.join(args.baseline_dir, "ensemble_model.joblib"))
    ensemble = bundle["ensemble"]
    orig_threshold = float(bundle.get("optimal_threshold",
                                      getattr(ensemble, "oof_threshold", 0.5)))
    report["baseline_threshold"] = orig_threshold

    # Detect whether baseline has GNN embedding dim (test feature count vs
    # training feature count mismatch)
    gnn_dim = 0
    expected = ensemble.scaler.n_features_in_
    got = X_te.shape[1]
    if expected > got:
        gnn_dim = expected - got
        print(f"\n  baseline expects {expected} features, got {got}.")
        print(f"  assuming {gnn_dim} GNN embedding dims. "
              f"Padding test-side with zeros (we don't have the GNN model).")
        pad_tr = np.zeros((len(X_tr), gnn_dim), dtype=np.float32)
        pad_te = np.zeros((len(X_te), gnn_dim), dtype=np.float32)
        X_tr = np.hstack([X_tr, pad_tr])
        X_te = np.hstack([X_te, pad_te])
    elif expected < got:
        raise RuntimeError(
            f"baseline expects {expected} features but we produced {got}. "
            f"Feature-engineering code has changed since the baseline was saved."
        )

    # ── 3. Predict on train (for threshold selection) + test
    print("\n  predicting on train + test with baseline ensemble ...")
    proba_tr = ensemble.predict_proba(X_tr)
    proba_te = ensemble.predict_proba(X_te)

    # ── 4. Threshold strategies — select on TRAIN, apply on TEST
    strategies = {
        "max_f1": threshold_max_f1(y_tr, proba_tr),
        "max_f2": threshold_max_fbeta(y_tr, proba_tr, beta=2.0),
        "min_cost": threshold_min_cost(
            y_tr, proba_tr, c_fn=args.cost_fn, c_fp=args.cost_fp,
        ),
    }

    comparison = {"test_metrics": {}}
    for name, res in strategies.items():
        comparison["test_metrics"][name] = evaluate_at_threshold(
            y_te, proba_te, res.threshold, name,
        )
        comparison["test_metrics"][name]["threshold_picked_on"] = "train"

    # Reference: original threshold from baseline
    comparison["test_metrics"]["baseline_orig_threshold"] = evaluate_at_threshold(
        y_te, proba_te, orig_threshold, "baseline_orig",
    )
    comparison["overall"] = evaluate_proba(y_te, proba_te, "final_ensemble")

    report["comparison"] = comparison

    # ── 5. Pretty-print table
    print("\n" + "=" * 80)
    print("  FINAL COMPARISON — thresholds chosen on train, metrics on test")
    print("=" * 80)
    print(f"  AUC-ROC={comparison['overall']['AUC-ROC']:.4f}  "
          f"AUC-PR={comparison['overall']['AUC-PR']:.4f}")
    print()
    print(f"  {'strategy':<24} {'thr':>7} {'F1':>7} {'F2':>7} "
          f"{'P':>7} {'R':>7}")
    print("  " + "-" * 70)
    for name in ("baseline_orig_threshold", "max_f1", "max_f2", "min_cost"):
        m = comparison["test_metrics"][name]
        print(f"  {name:<24} {m['threshold']:>7.4f} {m['F1']:>7.4f} "
              f"{m['F2']:>7.4f} {m['Precision']:>7.4f} {m['Recall']:>7.4f}")
    print("=" * 80)

    # ── 6. Submission CSVs for each strategy (if predict data available)
    predict_tables = load_predict_data(args.predict_dir)
    if predict_tables is not None:
        pred_user = predict_tables["user_info_predict"]
        pred_twd = predict_tables["twd_transfer_predict"]
        pred_crypto = predict_tables["crypto_transfer_predict"]
        pred_trading = predict_tables["usdt_twd_trading_predict"]
        pred_swap = predict_tables["usdt_swap_predict"]

        pred_feat = build_all_features(pred_user, pred_twd, pred_crypto,
                                        pred_trading, pred_swap)
        pred_feat = pred_feat.replace([np.inf, -np.inf], np.nan).fillna(0)
        train_cols = X_selected.columns.tolist()
        for col in train_cols:
            if col not in pred_feat.columns:
                pred_feat[col] = 0
        X_pred = pred_feat[train_cols].values.astype(np.float32)
        X_pred = np.hstack([X_pred, anomaly.transform(X_pred)])
        if gnn_dim > 0:
            X_pred = np.hstack([X_pred,
                                np.zeros((len(X_pred), gnn_dim),
                                         dtype=np.float32)])
        proba_pred = ensemble.predict_proba(X_pred)

        for name, res in strategies.items():
            pred_labels = (proba_pred >= res.threshold).astype(int)
            sub_df = pd.DataFrame({
                "user_id": pred_feat.index,
                "status": pred_labels,
            })
            path = os.path.join(args.output_dir, f"submission_{name}.csv")
            sub_df.to_csv(path, index=False)
            report.setdefault("submissions", {})[name] = {
                "path": path,
                "threshold": res.threshold,
                "predicted_positive": int(pred_labels.sum()),
                "predicted_negative": int((pred_labels == 0).sum()),
            }
            print(f"  wrote {path}  (positive={pred_labels.sum()})")

    # Extra strategy: top-K by risk score (useful when reviewer bandwidth
    # is a hard constraint — "flag the top 300 highest-risk accounts").
    if predict_tables is not None:
        for top_k in (200, 500, 1000):
            order = np.argsort(-proba_pred)
            pred_labels = np.zeros_like(proba_pred, dtype=int)
            pred_labels[order[:top_k]] = 1
            sub_df = pd.DataFrame({
                "user_id": pred_feat.index,
                "status": pred_labels,
            })
            path = os.path.join(args.output_dir, f"submission_top{top_k}.csv")
            sub_df.to_csv(path, index=False)
            report["submissions"][f"top{top_k}"] = {
                "path": path,
                "threshold": float(proba_pred[order[top_k - 1]]),
                "predicted_positive": int(top_k),
            }
            print(f"  wrote {path}  (top-{top_k})")

    # ── 7. Save full report
    report["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(args.output_dir, "FINAL_REPORT.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    pd.DataFrame({
        "y_true": y_te,
        "proba": proba_te,
    }).to_csv(os.path.join(args.output_dir, "test_proba.csv"), index=False)

    print(f"\n  final report: {args.output_dir}/FINAL_REPORT.json")


if __name__ == "__main__":
    main()
