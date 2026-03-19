"""
Module 7: XGBoost Fraud Classifier
Concatenates GNN embeddings (64-dim) with tabular features (~43-dim) and
optional CATE scores, then trains an XGBoost classifier.

Feature assembly (controlled by config['xgboost']):
  use_gnn_embeddings  → gnn_embeddings.npy  + gnn_user_id_map.csv
  use_tabular_features→ user_features.csv
  use_arm_features    → arm_features.csv
  use_cate_scores     → cate_scores.csv

Input : fraud labels CSV  (user_id, is_fraud)
Output:
  results/models/xgb_model.pkl
  results/features/fraud_scores.csv    — user_id, fraud_prob, prediction
  results/reports/shap_importance.csv  — feature importance via SHAP
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

from utils.utils import (
    load_config,
    save_pickle,
    compute_metrics,
    print_metrics,
    create_output_dirs,
    set_seed,
)


# ─────────────────────────────────────────────────────────────────────────────
# Feature assembly
# ─────────────────────────────────────────────────────────────────────────────

def load_gnn_embeddings(emb_path: str, map_path: str) -> pd.DataFrame:
    """Load embeddings and attach user_id column."""
    emb = np.load(emb_path)                    # [N, 64]
    uid_map = pd.read_csv(map_path)             # node_idx, user_id

    n_dims  = emb.shape[1]
    emb_df  = pd.DataFrame(emb, columns=[f'gnn_{i}' for i in range(n_dims)])
    emb_df['user_id'] = uid_map['user_id'].values
    return emb_df


def build_feature_matrix(
    config: Dict_,
    emb_path: str,
    map_path: str,
    tabular_path: str,
    arm_path: str,
    cate_path: str,
    labels_path: str,
    label_col: str = 'is_fraud',
) -> tuple:
    """
    Merge feature sources according to config['xgboost'] flags.

    Returns:
        X        — numpy float32 array [N, F]
        y        — numpy int array [N]
        user_ids — numpy array [N]
        feat_names — list of feature column names
    """
    cfg = config['xgboost']
    frames = []

    # GNN embeddings (anchor frame — always loaded to get user_ids)
    gnn_df = load_gnn_embeddings(emb_path, map_path)
    if cfg.get('use_gnn_embeddings', True):
        frames.append(gnn_df)
        print(f"  GNN embeddings : {gnn_df.shape[1] - 1} dims")
    else:
        # Keep user_id only for merging
        frames.append(gnn_df[['user_id']])

    # Tabular features
    if cfg.get('use_tabular_features', True) and Path(tabular_path).exists():
        tab = pd.read_csv(tabular_path)
        tab_feat = [c for c in tab.columns if c != 'user_id']
        frames.append(tab[['user_id'] + tab_feat])
        print(f"  Tabular features: {len(tab_feat)} dims")

    # ARM features
    if cfg.get('use_arm_features', False) and arm_path and Path(arm_path).exists():
        arm = pd.read_csv(arm_path)
        arm_feat = [c for c in arm.columns if c != 'user_id']
        frames.append(arm[['user_id'] + arm_feat])
        print(f"  ARM features    : {len(arm_feat)} dims")

    # CATE scores
    if cfg.get('use_cate_scores', True) and cate_path and Path(cate_path).exists():
        cate = pd.read_csv(cate_path)
        frames.append(cate[['user_id', 'cate_score']])
        print(f"  CATE scores     : 1 dim")

    # Merge all frames on user_id
    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on='user_id', how='inner')

    # Attach labels
    labels_df = pd.read_csv(labels_path)
    merged = merged.merge(labels_df[['user_id', label_col]], on='user_id', how='inner')

    user_ids  = merged['user_id'].values
    y         = merged[label_col].values.astype(int)
    feat_cols = [c for c in merged.columns if c not in ('user_id', label_col)]
    X         = merged[feat_cols].values.astype(np.float32)

    print(f"\n  Final feature matrix: {X.shape}  (positive rate={y.mean():.1%})")
    return X, y, user_ids, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost training
# ─────────────────────────────────────────────────────────────────────────────

def train_xgboost(
    X_train, y_train,
    X_val,   y_val,
    config: dict,
) -> xgb.XGBClassifier:
    cfg  = config['xgboost']
    neg  = int((y_train == 0).sum())
    pos  = int((y_train == 1).sum())
    spw  = neg / pos if pos > 0 else 1.0
    print(f"\n  scale_pos_weight = {spw:.1f}  (neg={neg}, pos={pos})")

    model = xgb.XGBClassifier(
        n_estimators        = cfg.get('n_estimators', 1000),
        learning_rate       = cfg.get('learning_rate', 0.05),
        max_depth           = cfg.get('max_depth', 6),
        min_child_weight    = cfg.get('min_child_weight', 5),
        subsample           = cfg.get('subsample', 0.8),
        colsample_bytree    = cfg.get('colsample_bytree', 0.8),
        reg_alpha           = cfg.get('reg_alpha', 0.1),
        reg_lambda          = cfg.get('reg_lambda', 1.0),
        scale_pos_weight    = spw,
        eval_metric         = cfg.get('eval_metric', 'aucpr'),
        early_stopping_rounds = cfg.get('early_stopping_rounds', 30),
        random_state        = cfg.get('random_state', 42),
        tree_method         = 'hist',
        device              = 'cuda' if config.get('device') == 'cuda' else 'cpu',
        verbosity           = 1,
    )

    model.fit(
        X_train, y_train,
        eval_set      = [(X_val, y_val)],
        verbose       = 50,
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# SHAP analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_importance(
    model: xgb.XGBClassifier,
    X: np.ndarray,
    feat_names: list,
    top_k: int = 20,
    out_path: str = None,
) -> pd.DataFrame:
    try:
        import shap
    except ImportError:
        print("  SHAP not installed — pip install shap")
        return pd.DataFrame()

    print("\nComputing SHAP values...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    importance = pd.DataFrame({
        'feature':         feat_names,
        'mean_abs_shap':   np.abs(shap_values).mean(axis=0),
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

    print(f"\nTop {top_k} features by SHAP importance:")
    print(importance.head(top_k).to_string(index=False))

    if out_path:
        importance.to_csv(out_path, index=False)
        print(f"\n  SHAP importance saved → {out_path}")

    return importance


# ─────────────────────────────────────────────────────────────────────────────
# Type alias (avoid importing typing at module level for compatibility)
# ─────────────────────────────────────────────────────────────────────────────
Dict_ = dict   # local alias


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Module 7 — XGBoost Fraud Classifier')
    parser.add_argument('--config',   default='arm_xgboost/configs/config.yaml')
    parser.add_argument('--emb',      required=True, help='gnn_embeddings.npy from Module 6')
    parser.add_argument('--emb_map',  required=True, help='gnn_user_id_map.csv from Module 6')
    parser.add_argument('--tabular',  required=True, help='user_features.csv from Module 1')
    parser.add_argument('--labels',   required=True, help='Fraud labels CSV (user_id, is_fraud)')
    parser.add_argument('--arm',      default=None,  help='arm_features.csv (optional)')
    parser.add_argument('--cate',     default=None,  help='cate_scores.csv (optional)')
    parser.add_argument('--label_col',default='is_fraud')
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config['seed'])
    create_output_dirs(config)

    print("\n" + "=" * 80)
    print("MODULE 7: XGBOOST FRAUD CLASSIFIER")
    print("=" * 80)

    # ── Feature assembly ───────────────────────────────────────────────────
    print("\nAssembling feature matrix...")
    X, y, user_ids, feat_names = build_feature_matrix(
        config       = config,
        emb_path     = args.emb,
        map_path     = args.emb_map,
        tabular_path = args.tabular,
        arm_path     = args.arm,
        cate_path    = args.cate,
        labels_path  = args.labels,
        label_col    = args.label_col,
    )

    # ── Train / test split ─────────────────────────────────────────────────
    test_size = config['xgboost'].get('test_size', 0.2)
    X_trainval, X_test, y_trainval, y_test, uid_trainval, uid_test = train_test_split(
        X, y, user_ids,
        test_size    = test_size,
        stratify     = y,
        random_state = config['xgboost'].get('random_state', 42),
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size    = 0.15,
        stratify     = y_trainval,
        random_state = config['xgboost'].get('random_state', 42),
    )

    print(f"\n  Train: {X_train.shape[0]:,}  Val: {X_val.shape[0]:,}  Test: {X_test.shape[0]:,}")

    # ── Train ──────────────────────────────────────────────────────────────
    model = train_xgboost(X_train, y_train, X_val, y_val, config)

    # ── Evaluate on test set ───────────────────────────────────────────────
    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs > 0.5).astype(int)
    test_metrics = compute_metrics(y_test, test_preds, test_probs)
    print_metrics(test_metrics, title="TEST SET METRICS")

    # ── Inference on ALL users ─────────────────────────────────────────────
    all_probs = model.predict_proba(X)[:, 1]
    all_preds = (all_probs > 0.5).astype(int)

    # ── Save outputs ───────────────────────────────────────────────────────
    out_dir    = Path(config['data']['output_dir'])
    feat_dir   = out_dir / 'features'
    model_dir  = out_dir / 'models'
    report_dir = out_dir / 'reports'
    for d in [feat_dir, model_dir, report_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Model
    model_path = model_dir / config['data']['xgb_model']
    save_pickle(model, model_path)

    # Fraud scores
    scores_df = pd.DataFrame({
        'user_id':    user_ids,
        'fraud_prob': all_probs,
        'prediction': all_preds,
        'true_label': y,
    }).sort_values('fraud_prob', ascending=False).reset_index(drop=True)

    scores_path = feat_dir / config['data']['fraud_scores']
    scores_df.to_csv(scores_path, index=False)
    print(f"\nFraud scores saved → {scores_path}")

    # SHAP importance
    if config['xgboost'].get('compute_shap', True):
        shap_path = report_dir / 'shap_importance.csv'
        compute_shap_importance(
            model      = model,
            X          = X,
            feat_names = feat_names,
            top_k      = config['xgboost'].get('shap_top_k', 20),
            out_path   = str(shap_path),
        )

    # Test metrics report
    metrics_path = report_dir / 'xgb_test_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)
    print(f"Metrics report   → {metrics_path}")

    print(f"\n{'=' * 80}")
    print("MODULE 7 COMPLETE")
    print(f"  Features used  : {X.shape[1]}")
    print(f"  Test PR-AUC    : {test_metrics.get('pr_auc', 0):.4f}")
    print(f"  Test ROC-AUC   : {test_metrics.get('roc_auc', 0):.4f}")
    print(f"  Test F1        : {test_metrics.get('f1_score', 0):.4f}")
    print(f"  Precision@100  : {test_metrics.get('precision_at_100', 0):.4f}")
    print(f"{'=' * 80}\n")


if __name__ == '__main__':
    main()
