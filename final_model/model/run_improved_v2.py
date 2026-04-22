"""End-to-end improved pipeline (v2): baseline ensemble + PU learning +
F-beta / cost-sensitive thresholds.

What it does:
  1. Re-creates the same train/test split as main.py.
  2. Loads the pre-trained ensemble from the baseline run.
  3. Produces PU-learning OOF on the training set (new base learner).
  4. Produces PU-learning test predictions.
  5. Re-trains a lightweight meta-learner on
       [xgb_oof, lgb_oof, cat_oof, pu_oof, xgb_test, lgb_test, cat_test, pu_test]
     — actually only on OOF during fit; at inference uses averaged preds.
  6. Applies the F-beta + cost-sensitive thresholds from `improved_evaluation`.

Writes everything to `--output_dir` for change-log auditability.
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

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
from main import load_and_validate  # noqa: E402
from pu_learning import pu_full_fit_predict, pu_oof_probabilities  # noqa: E402


def prepare(data_dir: str):
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


def get_base_oof_and_test(ensemble, X_tr, y_tr, X_te, n_splits=5):
    """Re-produce OOF probabilities for XGB/LGB/Cat using the *already-fit*
    fold models stored on the ensemble.

    The ensemble keeps K fold-models for each base learner. Reproducing a
    proper OOF from stored models requires knowing which rows each fold saw,
    which isn't persisted. Shortcut: re-run the same StratifiedKFold seed to
    recover fold indices, and predict with the fold-model on its validation
    rows. This matches main.py exactly because both use random_state=42.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    X_tr_s = ensemble.scaler.transform(X_tr)
    X_te_s = ensemble.scaler.transform(X_te)

    xgb_oof = np.zeros(len(y_tr))
    lgb_oof = np.zeros(len(y_tr))
    cat_oof = np.zeros(len(y_tr))

    for fold, (_, val_idx) in enumerate(skf.split(X_tr_s, y_tr)):
        x_val = X_tr_s[val_idx]

        xgb_m = ensemble.xgb_models[fold]
        lgb_m = ensemble.lgb_models[fold]
        cat_m = ensemble.cat_models[fold]

        xgb_oof[val_idx] = xgb_m.predict_proba(x_val)[:, 1]

        # LightGBM native (Focal Loss) returns raw scores; sklearn API returns probs.
        if hasattr(lgb_m, "predict_proba"):
            lgb_oof[val_idx] = lgb_m.predict_proba(x_val)[:, 1]
        else:
            raw = lgb_m.predict(x_val)
            lgb_oof[val_idx] = 1.0 / (1.0 + np.exp(-raw))

        cat_oof[val_idx] = cat_m.predict_proba(x_val)[:, 1]

    # Test: average across all fold models
    xgb_te = np.mean([m.predict_proba(X_te_s)[:, 1]
                      for m in ensemble.xgb_models], axis=0)
    if hasattr(ensemble.lgb_models[0], "predict_proba"):
        lgb_te = np.mean([m.predict_proba(X_te_s)[:, 1]
                          for m in ensemble.lgb_models], axis=0)
    else:
        lgb_te = np.mean([1.0 / (1.0 + np.exp(-m.predict(X_te_s)))
                          for m in ensemble.lgb_models], axis=0)
    cat_te = np.mean([m.predict_proba(X_te_s)[:, 1]
                      for m in ensemble.cat_models], axis=0)

    return xgb_oof, lgb_oof, cat_oof, xgb_te, lgb_te, cat_te


def build_meta_features(*columns):
    stack = np.column_stack(columns)
    return np.column_stack([
        stack,
        stack.max(axis=1),
        stack.min(axis=1),
        stack.std(axis=1),
        stack.mean(axis=1),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",
                    default=os.path.join(ROOT, "adjust_data", "train"))
    ap.add_argument("--baseline_dir",
                    default=os.path.join(ROOT, "final_model", "output", "baseline"))
    ap.add_argument("--output_dir",
                    default=os.path.join(ROOT, "final_model", "output", "improved_v2"))
    ap.add_argument("--pu_method", choices=("bagging", "elkanoto"),
                    default="bagging")
    ap.add_argument("--cost_fn", type=float, default=10.0)
    ap.add_argument("--cost_fp", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log = {"start_time": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": []}

    # ── 1. Prepare data (same as main.py)
    t0 = time.time()
    X, y = prepare(args.data_dir)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )
    X_tr, X_te, _ = add_anomaly_scores_to_splits(X_tr, X_te)
    log["stages"].append({"name": "prepare_data",
                          "duration_s": round(time.time() - t0, 1),
                          "X_tr": X_tr.shape, "X_te": X_te.shape})
    print(f"\n  data ready  tr={X_tr.shape} te={X_te.shape}")

    # ── 2. Load baseline ensemble
    t0 = time.time()
    bundle = joblib.load(os.path.join(args.baseline_dir, "ensemble_model.joblib"))
    ensemble = bundle["ensemble"]
    log["stages"].append({"name": "load_baseline",
                          "duration_s": round(time.time() - t0, 1),
                          "orig_threshold": float(bundle.get("optimal_threshold",
                                                             0.5))})

    # ── 3. Reproduce base-learner OOF + test predictions
    t0 = time.time()
    print("\n  reproducing XGB/LGB/Cat OOF ...")
    xgb_oof, lgb_oof, cat_oof, xgb_te, lgb_te, cat_te = get_base_oof_and_test(
        ensemble, X_tr, y_tr, X_te,
    )
    log["stages"].append({"name": "base_oof",
                          "duration_s": round(time.time() - t0, 1)})

    # ── 4. PU learning base learner
    t0 = time.time()
    print(f"\n  training PU({args.pu_method}) OOF on train ...")
    pu_oof, pu_info = pu_oof_probabilities(
        X_tr, y_tr, method=args.pu_method, n_splits=5,
    )
    print(f"  predicting PU({args.pu_method}) on test ...")
    _, pu_te = pu_full_fit_predict(X_tr, y_tr, X_te, method=args.pu_method)
    log["stages"].append({"name": "pu_learning",
                          "duration_s": round(time.time() - t0, 1),
                          **pu_info})

    # ── 5. Meta-learner (three variants for ablation)
    variants = {
        "ensemble_baseline(3)":  (xgb_oof, lgb_oof, cat_oof),
        "ensemble_with_pu(4)":   (xgb_oof, lgb_oof, cat_oof, pu_oof),
    }
    variant_test = {
        "ensemble_baseline(3)":  (xgb_te, lgb_te, cat_te),
        "ensemble_with_pu(4)":   (xgb_te, lgb_te, cat_te, pu_te),
    }

    all_metrics = {}
    for name, cols in variants.items():
        meta_X_tr = build_meta_features(*cols)
        meta_X_te = build_meta_features(*variant_test[name])

        scaler = StandardScaler().fit(meta_X_tr)
        meta = LogisticRegression(C=1.0, class_weight="balanced",
                                  max_iter=1000, random_state=42)
        meta.fit(scaler.transform(meta_X_tr), y_tr)
        proba_tr = meta.predict_proba(scaler.transform(meta_X_tr))[:, 1]
        proba_te = meta.predict_proba(scaler.transform(meta_X_te))[:, 1]

        # Find threshold on TRAIN OOF, apply on TEST — this is the honest
        # "deployable" protocol.
        tr_threshold_f1 = threshold_max_f1(y_tr, proba_tr).threshold
        tr_threshold_f2 = threshold_max_fbeta(y_tr, proba_tr, beta=2.0).threshold
        tr_threshold_cost = threshold_min_cost(
            y_tr, proba_tr, c_fn=args.cost_fn, c_fp=args.cost_fp,
        ).threshold

        m = {
            "variant": name,
            "meta_coef": {
                f"f{i}": float(c) for i, c in enumerate(meta.coef_[0])
            },
            "auc_pr_te": float(evaluate_proba(y_te, proba_te, name)["AUC-PR"]),
            "auc_roc_te": float(evaluate_proba(y_te, proba_te, name)["AUC-ROC"]),
            "tr_thresholds": {
                "max_f1": tr_threshold_f1,
                "max_f2": tr_threshold_f2,
                "min_cost": tr_threshold_cost,
            },
            "test_at_tr_f1": evaluate_at_threshold(
                y_te, proba_te, tr_threshold_f1, name,
            ),
            "test_at_tr_f2": evaluate_at_threshold(
                y_te, proba_te, tr_threshold_f2, name,
            ),
            "test_at_tr_cost": evaluate_at_threshold(
                y_te, proba_te, tr_threshold_cost, name,
            ),
            # Also the *diagnostic* test-set ceilings (upper bound if we could
            # perfectly pick a threshold):
            "test_ceiling_f1": {
                **evaluate_proba(y_te, proba_te, name),
                "max_f1_at": vars(threshold_max_f1(y_te, proba_te)),
                "max_f2_at": vars(threshold_max_fbeta(y_te, proba_te, beta=2.0)),
            },
        }
        all_metrics[name] = m

    # ── 6. Save artefacts
    with open(os.path.join(args.output_dir, "metrics_v2.json"), "w",
              encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False, default=str)

    pd.DataFrame({
        "y_true": y_te,
        "xgb": xgb_te, "lgb": lgb_te, "cat": cat_te, "pu": pu_te,
    }).to_csv(os.path.join(args.output_dir, "test_base_probas.csv"),
              index=False)

    log["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(args.output_dir, "run_log.json"), "w",
              encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    # ── 7. Print comparison
    print("\n" + "=" * 80)
    print("  Variant comparison (thresholds chosen on TRAIN OOF, reported on TEST)")
    print("=" * 80)
    header = (f"  {'variant':<24} {'thr_strat':<10} {'thr':>6} "
              f"{'F1':>6} {'F2':>6} {'P':>6} {'R':>6} {'AUC-PR':>7}")
    print(header)
    print("  " + "-" * 76)
    for name, m in all_metrics.items():
        for strat_key, label in (("test_at_tr_f1", "max_f1"),
                                  ("test_at_tr_f2", "max_f2"),
                                  ("test_at_tr_cost", "min_cost")):
            s = m[strat_key]
            print(f"  {name:<24} {label:<10} {s['threshold']:>6.3f} "
                  f"{s['F1']:>6.4f} {s['F2']:>6.4f} {s['Precision']:>6.4f} "
                  f"{s['Recall']:>6.4f} {m['auc_pr_te']:>7.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
