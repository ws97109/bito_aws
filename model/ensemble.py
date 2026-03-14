"""
Ensemble Model Module
論文級集成策略：XGBoost + Isolation Forest + GNN 三模型融合
使用 Stacking 與 Calibration
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict
import xgboost as xgb
from sklearn.ensemble import IsolationForest, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    classification_report,
)
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# XGBoost（主力分類器）
# ─────────────────────────────────────────────

def build_xgboost(scale_pos_weight: float = 50.0) -> xgb.XGBClassifier:
    """
    scale_pos_weight: 黑名單樣本稀少，設定正類權重補償
    論文參數：Tree depth 受限以提升可解釋性
    """
    return xgb.XGBClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,
        reg_alpha=0.1,         # L1 正則：稀疏特徵選擇
        reg_lambda=1.0,        # L2 正則
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="aucpr",   # AUC-PR 更適合不平衡問題
        tree_method="hist",    # 使用 CPU 版本的 hist 方法
        device="cpu",
        random_state=42,
        early_stopping_rounds=50,
    )


# ─────────────────────────────────────────────
# Isolation Forest（無監督異常偵測）
# ─────────────────────────────────────────────

def build_isolation_forest() -> IsolationForest:
    return IsolationForest(
        n_estimators=300,
        max_samples="auto",
        contamination=0.02,    # 預設 2% 異常比例
        random_state=42,
        n_jobs=-1,
    )


# ─────────────────────────────────────────────
# Stacking 集成
# ─────────────────────────────────────────────

class StackingEnsemble:
    """
    論文架構：
      Base learners: XGBoost, Isolation Forest, GNN
      Meta learner:  Logistic Regression（可解釋）
      校正:          Platt Scaling（輸出校準概率）
    """

    def __init__(self, n_splits: int = 5):
        self.n_splits    = n_splits
        self.xgb_model   = None
        self.iso_model   = None
        self.meta_model  = None
        self.scaler      = StandardScaler()
        self.is_fitted   = False

    def _get_xgb_oof(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, xgb.XGBClassifier]:
        """XGBoost OOF 預測（Out-of-Fold）"""
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        oof = np.zeros(len(y))
        models = []

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            n_neg = (y_tr == 0).sum()
            n_pos = (y_tr == 1).sum()
            spw   = n_neg / max(n_pos, 1)

            model = build_xgboost(scale_pos_weight=spw)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            oof[val_idx] = model.predict_proba(X_val)[:, 1]
            models.append(model)
            print(f"  Fold {fold+1}: AUC-PR = {average_precision_score(y_val, oof[val_idx]):.4f}")

        # 選最佳 fold 模型
        best = max(models, key=lambda m: m.best_score)
        return oof, best

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        gnn_proba: np.ndarray = None,
    ) -> "StackingEnsemble":
        print("[1/3] 訓練 XGBoost OOF ...")
        X_scaled = self.scaler.fit_transform(X)
        xgb_oof, self.xgb_model = self._get_xgb_oof(X_scaled, y)

        print("[2/3] 訓練 Isolation Forest ...")
        self.iso_model = build_isolation_forest()
        self.iso_model.fit(X_scaled)
        iso_score = -self.iso_model.decision_function(X_scaled)   # 越高越異常
        iso_score = (iso_score - iso_score.min()) / (iso_score.max() - iso_score.min() + 1e-9)

        print("[3/3] 訓練 Meta Learner ...")
        meta_X = np.column_stack([xgb_oof, iso_score])
        if gnn_proba is not None:
            meta_X = np.column_stack([meta_X, gnn_proba])

        self.meta_model = CalibratedClassifierCV(
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=500),
            method="sigmoid",
            cv=3,
        )
        self.meta_model.fit(meta_X, y)
        self.use_gnn    = gnn_proba is not None
        self.is_fitted  = True
        return self

    def predict_proba(
        self,
        X: np.ndarray,
        gnn_proba: np.ndarray = None,
    ) -> np.ndarray:
        X_scaled  = self.scaler.transform(X)
        xgb_p     = self.xgb_model.predict_proba(X_scaled)[:, 1]
        iso_score = -self.iso_model.decision_function(X_scaled)
        iso_score = (iso_score - iso_score.min()) / (iso_score.max() - iso_score.min() + 1e-9)

        meta_X = np.column_stack([xgb_p, iso_score])
        if self.use_gnn and gnn_proba is not None:
            meta_X = np.column_stack([meta_X, gnn_proba])

        return self.meta_model.predict_proba(meta_X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5, **kwargs) -> np.ndarray:
        return (self.predict_proba(X, **kwargs) >= threshold).astype(int)


# ─────────────────────────────────────────────
# 評估函數
# ─────────────────────────────────────────────

def evaluate(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    label: str = "Model",
) -> Dict:
    y_pred = (y_proba >= threshold).astype(int)
    metrics = {
        "AUC-ROC": roc_auc_score(y_true, y_proba),
        "AUC-PR":  average_precision_score(y_true, y_proba),
        "F1":      f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
    }
    print(f"\n{'='*40}")
    print(f"  {label}")
    print(f"{'='*40}")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")
    print(classification_report(y_true, y_pred, target_names=["正常", "黑名單"]))
    return metrics


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """F1 最大化的最佳閾值搜索"""
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        f1 = f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"最佳閾值: {best_t:.2f} (F1={best_f1:.4f})")
    return best_t