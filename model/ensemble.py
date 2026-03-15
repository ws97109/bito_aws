"""
Ensemble Model Module
集成策略：XGBoost + LightGBM + CatBoost + Isolation Forest
融合方式：Soft Voting（直接平均三模型機率）+ Isolation Forest 異常分數加權
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import IsolationForest
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
# XGBoost
# ─────────────────────────────────────────────

def build_xgboost(scale_pos_weight: float = 50.0) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=1500,
        max_depth=5,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.6,
        min_child_weight=3,
        reg_alpha=0.5,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        tree_method="hist",
        device="cpu",
        random_state=42,
        early_stopping_rounds=100,
    )


# ─────────────────────────────────────────────
# LightGBM
# ─────────────────────────────────────────────

def build_lightgbm(scale_pos_weight: float = 50.0) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=1500,
        max_depth=5,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.6,
        min_child_samples=10,
        reg_alpha=0.5,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1,
    )


# ─────────────────────────────────────────────
# CatBoost
# ─────────────────────────────────────────────

def build_catboost(scale_pos_weight: float = 50.0) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=1500,
        depth=5,
        learning_rate=0.01,
        l2_leaf_reg=2.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="PRAUC",
        random_seed=42,
        verbose=0,
        early_stopping_rounds=100,
    )


# ─────────────────────────────────────────────
# Isolation Forest（無監督異常偵測）
# ─────────────────────────────────────────────

def build_isolation_forest() -> IsolationForest:
    return IsolationForest(
        n_estimators=300,
        max_samples="auto",
        contamination=0.02,
        random_state=42,
        n_jobs=-1,
    )


# ─────────────────────────────────────────────
# Soft Voting Ensemble
# ─────────────────────────────────────────────

class StackingEnsemble:
    """
    Base learners: XGBoost, LightGBM, CatBoost, Isolation Forest, (optional) GNN
    融合方式: Soft Voting（加權平均）
    """

    def __init__(self, n_splits: int = 5):
        self.n_splits    = n_splits
        self.xgb_model   = None
        self.lgb_model   = None
        self.cat_model   = None
        self.iso_model   = None
        self.scaler      = StandardScaler()
        self.is_fitted   = False
        # Soft Voting 權重（由 CV 表現決定）
        self.weights     = None

    def _get_oof(
        self,
        X: np.ndarray,
        y: np.ndarray,
        builder_fn,
        model_name: str,
    ) -> Tuple[np.ndarray, object, float]:
        """通用 OOF 預測，回傳 OOF 機率、最佳模型、平均 AUC-PR"""
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        oof = np.zeros(len(y))
        models = []
        scores = []

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            n_neg = (y_tr == 0).sum()
            n_pos = (y_tr == 1).sum()
            spw = n_neg / max(n_pos, 1)

            model = builder_fn(scale_pos_weight=spw)

            if model_name == "XGBoost":
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            elif model_name == "LightGBM":
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
            elif model_name == "CatBoost":
                model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
            else:
                model.fit(X_tr, y_tr)

            oof[val_idx] = model.predict_proba(X_val)[:, 1]
            fold_score = average_precision_score(y_val, oof[val_idx])
            scores.append(fold_score)
            models.append((model, fold_score))
            print(f"  {model_name} Fold {fold+1}: AUC-PR = {fold_score:.4f}")

        best_model = max(models, key=lambda x: x[1])[0]
        avg_score = np.mean(scores)
        print(f"  {model_name} 平均 AUC-PR = {avg_score:.4f}")
        return oof, best_model, avg_score

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        gnn_proba: np.ndarray = None,
    ) -> "StackingEnsemble":
        X_scaled = self.scaler.fit_transform(X)

        print("\n[1/4] 訓練 XGBoost OOF ...")
        xgb_oof, self.xgb_model, xgb_score = self._get_oof(X_scaled, y, build_xgboost, "XGBoost")

        print("\n[2/4] 訓練 LightGBM OOF ...")
        lgb_oof, self.lgb_model, lgb_score = self._get_oof(X_scaled, y, build_lightgbm, "LightGBM")

        print("\n[3/4] 訓練 CatBoost OOF ...")
        cat_oof, self.cat_model, cat_score = self._get_oof(X_scaled, y, build_catboost, "CatBoost")

        print("\n[4/4] 訓練 Isolation Forest ...")
        self.iso_model = build_isolation_forest()
        self.iso_model.fit(X_scaled)

        # 根據 AUC-PR 表現計算 Soft Voting 權重
        total = xgb_score + lgb_score + cat_score
        self.weights = {
            "xgb": xgb_score / total,
            "lgb": lgb_score / total,
            "cat": cat_score / total,
        }
        print(f"\n  Soft Voting 權重: XGB={self.weights['xgb']:.3f}, "
              f"LGB={self.weights['lgb']:.3f}, CAT={self.weights['cat']:.3f}")

        # OOF 加權平均
        oof_avg = (
            self.weights["xgb"] * xgb_oof +
            self.weights["lgb"] * lgb_oof +
            self.weights["cat"] * cat_oof
        )

        # 加入 Isolation Forest 異常分數（微調）
        iso_score = -self.iso_model.decision_function(X_scaled)
        iso_score = (iso_score - iso_score.min()) / (iso_score.max() - iso_score.min() + 1e-9)
        # 異常分數加權混入（佔 5%）
        self.iso_weight = 0.05
        oof_final = (1 - self.iso_weight) * oof_avg + self.iso_weight * iso_score

        self.use_gnn   = gnn_proba is not None
        self.is_fitted = True

        oof_ap = average_precision_score(y, oof_final)
        print(f"\n  Ensemble OOF AUC-PR = {oof_ap:.4f}")

        # 找 OOF 最佳閾值
        best_f1, best_t = 0, 0.5
        for t in np.arange(0.05, 0.9, 0.01):
            f1 = f1_score(y, (oof_final >= t).astype(int))
            if f1 > best_f1:
                best_f1, best_t = f1, t
        self.oof_threshold = best_t
        print(f"  OOF 最佳閾值 = {best_t:.2f} (F1={best_f1:.4f})")

        return self

    def predict_proba(
        self,
        X: np.ndarray,
        gnn_proba: np.ndarray = None,
    ) -> np.ndarray:
        X_scaled = self.scaler.transform(X)

        xgb_p = self.xgb_model.predict_proba(X_scaled)[:, 1]
        lgb_p = self.lgb_model.predict_proba(X_scaled)[:, 1]
        cat_p = self.cat_model.predict_proba(X_scaled)[:, 1]

        avg_p = (
            self.weights["xgb"] * xgb_p +
            self.weights["lgb"] * lgb_p +
            self.weights["cat"] * cat_p
        )

        iso_score = -self.iso_model.decision_function(X_scaled)
        iso_score = (iso_score - iso_score.min()) / (iso_score.max() - iso_score.min() + 1e-9)

        final_p = (1 - self.iso_weight) * avg_p + self.iso_weight * iso_score

        if self.use_gnn and gnn_proba is not None:
            # GNN 機率混入（佔 10%）
            final_p = 0.9 * final_p + 0.1 * gnn_proba

        return final_p

    def predict(self, X: np.ndarray, threshold: float = None, **kwargs) -> np.ndarray:
        if threshold is None:
            threshold = getattr(self, "oof_threshold", 0.5)
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
        "AUC-ROC":   roc_auc_score(y_true, y_proba),
        "AUC-PR":    average_precision_score(y_true, y_proba),
        "F1":        f1_score(y_true, y_pred),
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
    for t in np.arange(0.05, 0.9, 0.01):
        f1 = f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"最佳閾值: {best_t:.2f} (F1={best_f1:.4f})")
    return best_t
