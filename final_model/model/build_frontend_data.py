"""Build all frontend-required CSVs from a final_model pipeline output.

The main pipeline produces:
  shap_values_all.csv    (63,770 × 79 — one row per user, all SHAP values)
  test_predictions.csv   (test-set predictions with true_label)
  predict_detail.csv     (predict-set probabilities)
  user_risk_scores.csv   (train-set scores)

The frontend additionally needs:
  all_user_risk_scores.csv    (train + predict risk scores merged)
  shap_values_all_top10.csv   (each row: keep only top-10 |SHAP|, others blank)
  shap_values_blacklist.csv   (rows filtered to status=1 users)
  shap_top10_fp.csv           (top-10 features by SHAP contribution on FP cases)
  shap_top10_fn.csv           (top-10 features by SHAP contribution on FN cases)

Usage:
    python build_frontend_data.py --input_dir ../output/baseline_loo \
                                   --output_dir ../output/baseline_loo
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


RISK_LEVEL_THRESHOLDS = [
    (0.90, "極高風險"),
    (0.70, "高風險"),
    (0.40, "中風險"),
    (0.20, "低風險"),
    (0.00, "極低風險"),
]


def risk_level(score: float) -> str:
    for t, label in RISK_LEVEL_THRESHOLDS:
        if score >= t:
            return label
    return RISK_LEVEL_THRESHOLDS[-1][1]


def build_all_user_risk_scores(
    test_predictions_path: str,
    user_risk_path: str,
    predict_detail_path: str,
    output_path: str,
):
    """Merge train-side risk scores with predict-side predictions."""
    # Train-side (from full training set, with true labels)
    train_df = pd.read_csv(user_risk_path)
    # Normalise columns: user_risk_scores.csv has at least user_id + risk_score
    # and sometimes true_label / predicted / risk_level.
    if "true_label" not in train_df.columns:
        # Derive from test_predictions.csv if present
        if os.path.exists(test_predictions_path):
            test_df = pd.read_csv(test_predictions_path)
            tl = dict(zip(test_df["user_id"], test_df["true_label"]))
            train_df["true_label"] = train_df["user_id"].map(tl)
        else:
            train_df["true_label"] = np.nan
    if "predicted_blacklist" not in train_df.columns:
        # Use threshold from predict_detail if we can, else 0.5
        train_df["predicted_blacklist"] = (train_df["risk_score"] >= 0.5).astype(int)
    train_df["risk_level"] = train_df["risk_score"].apply(risk_level)
    train_df["data_source"] = "train"

    # Predict side
    pred_df = pd.read_csv(predict_detail_path)
    pred_df = pred_df.rename(columns={
        "proba": "risk_score",
        "probability": "risk_score",
    })
    if "status" in pred_df.columns and "predicted_blacklist" not in pred_df.columns:
        pred_df["predicted_blacklist"] = pred_df["status"].astype(int)
    pred_df["risk_level"] = pred_df["risk_score"].apply(risk_level)
    pred_df["data_source"] = "predict"
    pred_df["true_label"] = np.nan

    cols = ["user_id", "true_label", "risk_score",
            "predicted_blacklist", "risk_level", "data_source"]
    merged = pd.concat(
        [train_df[cols], pred_df[cols]],
        ignore_index=True,
    )
    merged = merged.sort_values("risk_score", ascending=False)
    # Cast true_label to clean int strings — the frontend compares against
    # "0" / "1" string literals, not "0.0" / "1.0".
    merged["true_label"] = merged["true_label"].map(
        lambda x: "" if pd.isna(x) else str(int(float(x))))
    merged.to_csv(output_path, index=False)
    print(f"  ✓ {output_path}  {len(merged):,} rows")


def build_shap_top10(
    shap_all_path: str, output_path: str,
):
    """Per-row top-10 SHAP magnitudes; blank out the rest."""
    df = pd.read_csv(shap_all_path)
    user_ids = df["user_id"]
    feat_cols = [c for c in df.columns if c != "user_id"]
    values = df[feat_cols].values.astype(float)

    # For each row, keep top-10 |value|; zero out the rest
    top10_mask = np.zeros_like(values, dtype=bool)
    for i in range(len(values)):
        abs_vals = np.abs(values[i])
        top_idx = np.argpartition(-abs_vals, 10)[:10] if abs_vals.size > 10 else np.arange(abs_vals.size)
        top10_mask[i, top_idx] = True

    masked = np.where(top10_mask, values, np.nan)
    out = pd.DataFrame(masked, columns=feat_cols)
    out.insert(0, "user_id", user_ids.values)
    # Write with NaN as empty string (matches old format)
    out.to_csv(output_path, index=False, na_rep="")
    print(f"  ✓ {output_path}  {len(out):,} rows × {len(feat_cols)} feat cols")


def build_shap_blacklist(
    shap_all_path: str,
    test_predictions_path: str,
    user_risk_path: str,
    output_path: str,
):
    """SHAP rows for users with true_label == 1 (confirmed blacklist).

    We reconstruct the blacklist user_ids from user_risk_scores.csv
    (which covers only the train set — predict set has no true_label).
    """
    # Figure out blacklist user_ids
    bl_ids = set()
    if os.path.exists(test_predictions_path):
        tp = pd.read_csv(test_predictions_path)
        bl_ids.update(tp.loc[tp["true_label"] == 1, "user_id"].tolist())
    # train_label: we can infer from user_risk_scores if it has labels
    ur = pd.read_csv(user_risk_path)
    for cand in ("true_label", "status"):
        if cand in ur.columns:
            bl_ids.update(ur.loc[ur[cand] == 1, "user_id"].tolist())
            break

    shap_df = pd.read_csv(shap_all_path)
    filtered = shap_df[shap_df["user_id"].isin(bl_ids)]
    filtered.to_csv(output_path, index=False)
    print(f"  ✓ {output_path}  {len(filtered):,} blacklist rows")


def build_top10_fp_fn(
    shap_all_path: str,
    test_predictions_path: str,
    output_fp: str,
    output_fn: str,
):
    """Aggregated top-10 features driving FP / FN predictions on test set."""
    tp = pd.read_csv(test_predictions_path)
    # Robust column detection
    pred_col = "pred_label" if "pred_label" in tp.columns else "predicted_label"
    fp_ids = tp.loc[(tp["true_label"] == 0) & (tp[pred_col] == 1), "user_id"]
    fn_ids = tp.loc[(tp["true_label"] == 1) & (tp[pred_col] == 0), "user_id"]

    shap_df = pd.read_csv(shap_all_path).set_index("user_id")
    feat_cols = list(shap_df.columns)

    def _aggregate(ids, label: str, out_path: str):
        subset = shap_df.loc[shap_df.index.intersection(ids)]
        if subset.empty:
            # Write an empty file with header so frontend doesn't crash
            pd.DataFrame(columns=[
                "排名", "feature", "中文名稱", "SHAP值",
                "SHAP均值", "方向", "佔比", "頻率次數", "累積%",
            ]).to_csv(out_path, index=False)
            print(f"  ✓ {out_path}  (0 {label} users)")
            return

        mean_abs = subset.abs().mean()
        total = mean_abs.sum()
        mean_signed = subset.mean()

        rows = []
        cum = 0.0
        for rank, (feat, mean_val) in enumerate(
            mean_abs.sort_values(ascending=False).head(10).items(), start=1,
        ):
            pct = mean_val / total * 100 if total > 0 else 0
            cum += pct
            direction = "+" if mean_signed[feat] >= 0 else "-"
            rows.append({
                "排名": rank,
                "feature": feat,
                "中文名稱": feat,  # fallback: same name (frontend has a map)
                "SHAP值": float(mean_val),
                "SHAP均值": float(mean_signed[feat]),
                "方向": direction,
                "佔比": f"{pct:.2f}%",
                "頻率次數": int(len(subset)),
                "累積%": f"{cum:.2f}%",
            })
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"  ✓ {out_path}  ({len(subset)} {label} users)")

    _aggregate(fp_ids, "FP", output_fp)
    _aggregate(fn_ids, "FN", output_fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    shap_all = os.path.join(args.input_dir, "shap_values_all.csv")
    test_preds = os.path.join(args.input_dir, "test_predictions.csv")
    user_risk = os.path.join(args.input_dir, "user_risk_scores.csv")
    predict_detail = os.path.join(args.input_dir, "predict_detail.csv")

    print("\nBuilding derived frontend CSVs …\n")
    build_all_user_risk_scores(
        test_preds, user_risk, predict_detail,
        os.path.join(args.output_dir, "all_user_risk_scores.csv"),
    )
    build_shap_top10(
        shap_all,
        os.path.join(args.output_dir, "shap_values_all_top10.csv"),
    )
    build_shap_blacklist(
        shap_all, test_preds, user_risk,
        os.path.join(args.output_dir, "shap_values_blacklist.csv"),
    )
    build_top10_fp_fn(
        shap_all, test_preds,
        os.path.join(args.output_dir, "shap_top10_fp.csv"),
        os.path.join(args.output_dir, "shap_top10_fn.csv"),
    )
    print("\nDone.\n")


if __name__ == "__main__":
    main()
