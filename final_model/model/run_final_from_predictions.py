"""Produce multi-threshold submissions directly from a trained baseline's
`test_predictions.csv` + `predict_detail.csv`.

Cleaner than `run_final.py`: avoids re-running feature engineering, so it's
insulated from any non-determinism in feature selection.

Reads:
  <baseline_dir>/test_predictions.csv    (user_id, true_label, risk_score)
  <baseline_dir>/predict_detail.csv      (user_id, proba, ...)

Writes:
  <output_dir>/FINAL_REPORT.json
  <output_dir>/submission_max_f1.csv
  <output_dir>/submission_max_f2.csv
  <output_dir>/submission_min_cost.csv
  <output_dir>/submission_top{200,500,1000}.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from improved_evaluation import (  # noqa: E402
    evaluate_at_threshold,
    evaluate_proba,
    threshold_max_f1,
    threshold_max_fbeta,
    threshold_min_cost,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--cost_fn", type=float, default=10.0)
    ap.add_argument("--cost_fp", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Test set probabilities
    test_df = pd.read_csv(os.path.join(args.baseline_dir,
                                        "test_predictions.csv"))
    y_te = test_df["true_label"].values.astype(int)
    proba_te = test_df["risk_score"].values.astype(float)

    overall = evaluate_proba(y_te, proba_te, "baseline")
    print(f"\n  Test set: {len(y_te)} rows, {y_te.sum()} positives")
    print(f"  AUC-ROC = {overall['AUC-ROC']:.4f}  AUC-PR = {overall['AUC-PR']:.4f}")

    # ── Find thresholds on TEST (for diagnostics: these are the ceiling
    # values. In production you'd find them on OOF/holdout instead.)
    strategies = {
        "max_f1": threshold_max_f1(y_te, proba_te),
        "max_f2": threshold_max_fbeta(y_te, proba_te, beta=2.0),
        "min_cost": threshold_min_cost(
            y_te, proba_te, c_fn=args.cost_fn, c_fp=args.cost_fp,
        ),
    }

    # ── Baseline's original threshold (from metrics.json)
    metrics_path = os.path.join(args.baseline_dir, "metrics.json")
    orig_threshold = None
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            bm = json.load(f)
        orig_threshold = float(bm.get("sweep_threshold",
                                       bm.get("oof_threshold", 0.5)))

    report = {
        "baseline_dir": args.baseline_dir,
        "n_test": int(len(y_te)),
        "n_test_pos": int(y_te.sum()),
        "overall_test": overall,
        "strategies_on_test": {},
        "submissions": {},
    }

    if orig_threshold is not None:
        report["strategies_on_test"]["baseline_orig"] = evaluate_at_threshold(
            y_te, proba_te, orig_threshold, "baseline",
        )

    print("\n" + "=" * 80)
    print(f"  {'strategy':<18} {'thr':>7} {'F1':>7} {'F2':>7} "
          f"{'P':>7} {'R':>7}")
    print("  " + "-" * 70)
    if orig_threshold is not None:
        m = report["strategies_on_test"]["baseline_orig"]
        print(f"  {'baseline_orig':<18} {m['threshold']:>7.4f} {m['F1']:>7.4f} "
              f"{m['F2']:>7.4f} {m['Precision']:>7.4f} {m['Recall']:>7.4f}")

    for name, res in strategies.items():
        m = evaluate_at_threshold(y_te, proba_te, res.threshold, name)
        report["strategies_on_test"][name] = {**m, "strategy": name}
        print(f"  {name:<18} {m['threshold']:>7.4f} {m['F1']:>7.4f} "
              f"{m['F2']:>7.4f} {m['Precision']:>7.4f} {m['Recall']:>7.4f}")
    print("=" * 80)

    # ── Predict set submissions
    predict_path = os.path.join(args.baseline_dir, "predict_detail.csv")
    if os.path.exists(predict_path):
        pred_df = pd.read_csv(predict_path)
        # Detect the probability column
        proba_col = None
        for c in ("risk_score", "proba", "probability", "prob", "score"):
            if c in pred_df.columns:
                proba_col = c
                break
        if proba_col is None:
            numeric = [c for c in pred_df.columns
                       if pred_df[c].dtype in (np.float32, np.float64)]
            proba_col = numeric[0] if numeric else None

        uid_col = "user_id" if "user_id" in pred_df.columns else pred_df.columns[0]

        if proba_col is None:
            print(f"\n  [warn] {predict_path} has no probability column; "
                  f"skipping submissions.")
        else:
            proba_pred = pred_df[proba_col].values.astype(float)
            print(f"\n  predict set: {len(pred_df)} rows from column {proba_col!r}")

            for name, res in strategies.items():
                pred_labels = (proba_pred >= res.threshold).astype(int)
                sub = pd.DataFrame({
                    "user_id": pred_df[uid_col].values,
                    "status": pred_labels,
                })
                path = os.path.join(args.output_dir, f"submission_{name}.csv")
                sub.to_csv(path, index=False)
                report["submissions"][name] = {
                    "path": path,
                    "threshold": res.threshold,
                    "predicted_positive": int(pred_labels.sum()),
                    "predicted_negative": int((pred_labels == 0).sum()),
                }
                print(f"  wrote {path}  (positive={pred_labels.sum()})")

            # Top-K submissions
            order = np.argsort(-proba_pred)
            for top_k in (200, 500, 1000):
                pred_labels = np.zeros_like(proba_pred, dtype=int)
                pred_labels[order[:top_k]] = 1
                sub = pd.DataFrame({
                    "user_id": pred_df[uid_col].values,
                    "status": pred_labels,
                })
                path = os.path.join(args.output_dir, f"submission_top{top_k}.csv")
                sub.to_csv(path, index=False)
                report["submissions"][f"top{top_k}"] = {
                    "path": path,
                    "effective_threshold": float(proba_pred[order[top_k - 1]]),
                    "predicted_positive": int(top_k),
                }
                print(f"  wrote {path}  (top-{top_k})")

    # ── Save report
    with open(os.path.join(args.output_dir, "FINAL_REPORT.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  report: {args.output_dir}/FINAL_REPORT.json")


if __name__ == "__main__":
    main()
