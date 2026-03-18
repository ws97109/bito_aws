"""
Ensemble Model Module (v4)
集成策略：XGBoost + LightGBM (Focal Loss) + CatBoost
融合方式：True Stacking — OOF 機率 → Logistic Regression meta-learner
改進 v4：
  - LightGBM 改用 Focal Loss（非線性降權容易樣本，專注邊界難分樣本）
  - XGBoost / CatBoost 保持 scale_pos_weight（確保 ensemble 多樣性）
  - 可選 Borderline-SMOTE（在每個 CV fold 內部執行）
  - 三模型各用差異化超參數 + 不同損失函數 → 最大化多樣性
  - 保留所有 K-fold 模型做 averaging → 穩定 base predictions
  - PR-curve 精確閾值搜索
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
import xgboost as xgb
import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    precision_recall_curve,
    classification_report,
)
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Focal Loss for LightGBM
# ─────────────────────────────────────────────

def focal_loss_objective(alpha: float = 0.75, gamma: float = 2.0):
    """
    回傳 LightGBM 用的 Focal Loss objective function。

    Focal Loss 核心思想：
    - scale_pos_weight 對所有正樣本一視同仁加權
    - Focal Loss 對「已經分對的容易樣本」大幅降低 loss，
      對「分不清的邊界樣本」保持高 loss，讓模型集中學習決策邊界

    alpha: 正例權重，30:1 場景建議 0.70~0.90
    gamma: focusing 強度，建議 1.0~3.0
    """
    def _focal_obj(y_pred, dtrain):
        y_true = dtrain.get_label()
        p = 1.0 / (1.0 + np.exp(-y_pred))  # sigmoid
        pt = np.where(y_true == 1, p, 1 - p)
        a = np.where(y_true == 1, alpha, 1 - alpha)

        # gradient
        grad = a * (1 - pt) ** gamma * (p - y_true)
        # hessian (approximation)
        hess = a * (1 - pt) ** gamma * p * (1 - p)
        # 確保 hessian > 0
        hess = np.maximum(hess, 1e-7)
        return grad, hess

    return _focal_obj


def focal_loss_eval(y_pred, dtrain):
    """自訂 eval metric: binary logloss（因為 Focal Loss 的 predict 輸出是 raw score）"""
    y_true = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-y_pred))
    p = np.clip(p, 1e-7, 1 - 1e-7)
    loss = -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    return "focal_logloss", loss, False  # False = lower is better


# ─────────────────────────────────────────────
# XGBoost — 深樹 + 強正則
# ─────────────────────────────────────────────

def build_xgboost(scale_pos_weight: float = 50.0) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=2500,
        max_depth=6,
        learning_rate=0.005,
        subsample=0.7,
        colsample_bytree=0.5,
        min_child_weight=40,
        reg_alpha=0.8,
        reg_lambda=0.1,
        gamma=0.3,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        tree_method="hist",
        device="cpu",
        random_state=42,
        early_stopping_rounds=150,
    )


# ─────────────────────────────────────────────
# LightGBM — Focal Loss（不使用 scale_pos_weight）
# ─────────────────────────────────────────────

def build_lightgbm_params() -> dict:
    """回傳 LightGBM 的 native API 參數（搭配 Focal Loss 使用）"""
    return {
        "num_leaves": 63,
        "max_depth": 4,
        "learning_rate": 0.008,
        "subsample": 0.75,
        "colsample_bytree": 0.65,
        "min_child_samples": 25,
        "reg_alpha": 0.1,
        "reg_lambda": 0.5,
        "random_state": 123,
        "verbose": -1,
        # 不設 scale_pos_weight — Focal Loss 的 alpha 已處理不平衡
        # 不設 objective — 由自訂 fobj 接管
    }


def build_lightgbm(scale_pos_weight: float = 50.0) -> LGBMClassifier:
    """Fallback: 不用 Focal Loss 時的 sklearn API 版本"""
    return LGBMClassifier(
        n_estimators=2500,
        max_depth=4,
        learning_rate=0.008,
        subsample=0.75,
        colsample_bytree=0.65,
        min_child_samples=25,
        reg_alpha=0.1,
        reg_lambda=0.5,
        num_leaves=63,
        scale_pos_weight=scale_pos_weight,
        random_state=123,
        verbose=-1,
    )


# ─────────────────────────────────────────────
# CatBoost — 中等深度 + 高 bagging
# ─────────────────────────────────────────────

def build_catboost(scale_pos_weight: float = 50.0) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=2500,
        depth=7,
        learning_rate=0.01,
        l2_leaf_reg=3.0,
        bagging_temperature=0.8,
        random_strength=0.5,
        scale_pos_weight=scale_pos_weight,
        eval_metric="PRAUC",
        random_seed=7,
        verbose=0,
        early_stopping_rounds=150,
    )


# ─────────────────────────────────────────────
# True Stacking Ensemble
# ─────────────────────────────────────────────

class StackingEnsemble:
    """
    Level-0: XGBoost, LightGBM (Focal Loss), CatBoost (K-fold OOF)
    Level-1: Logistic Regression on [base_predictions + statistics]

    v4 改進：
    - LightGBM 使用 Focal Loss（alpha=0.75, gamma=2.0）
    - 可選 Borderline-SMOTE（在每個 CV fold 內部執行）
    - 三模型故意使用不同損失函數，最大化 ensemble 多樣性
    """

    def __init__(
        self,
        n_splits: int = 5,
        use_focal_loss: bool = True,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        use_smote: bool = False,
        smote_strategy: float = 0.3,
    ):
        self.n_splits       = n_splits
        self.use_focal_loss = use_focal_loss
        self.focal_alpha    = focal_alpha
        self.focal_gamma    = focal_gamma
        self.use_smote      = use_smote
        self.smote_strategy = smote_strategy

        self.xgb_model    = None   # best fold for SHAP
        self.xgb_models   = []
        self.lgb_models   = []
        self.cat_models   = []
        self.meta_model   = None   # Level-1 meta-learner
        self.scaler       = StandardScaler()
        self.meta_scaler  = StandardScaler()
        self.is_fitted    = False
        self.weights      = None   # kept for backward compat

    def _apply_smote(self, X_tr, y_tr, fold: int):
        """在 CV fold 內部執行 Borderline-SMOTE"""
        if not self.use_smote:
            return X_tr, y_tr

        try:
            from imblearn.over_sampling import BorderlineSMOTE
            smote = BorderlineSMOTE(
                sampling_strategy=self.smote_strategy,
                k_neighbors=5,
                random_state=42 + fold,
            )
            X_res, y_res = smote.fit_resample(X_tr, y_tr)
            n_new = len(X_res) - len(X_tr)
            if fold == 0:
                print(f"    SMOTE: {len(X_tr)} → {len(X_res)} (+{n_new} 合成正樣本)")
            return X_res, y_res
        except Exception as e:
            print(f"    [警告] SMOTE 失敗 ({e})，使用原始資料")
            return X_tr, y_tr

    def _get_oof(
        self,
        X: np.ndarray,
        y: np.ndarray,
        builder_fn,
        model_name: str,
    ) -> Tuple[np.ndarray, List, object, float]:
        """通用 OOF 預測，回傳 OOF 機率、所有模型列表、最佳模型、平均 AUC-PR"""
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        oof = np.zeros(len(y))
        all_models = []
        scores = []

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            # SMOTE 在 fold 內部（只對訓練集，驗證集保持原始分佈）
            X_tr_sm, y_tr_sm = self._apply_smote(X_tr, y_tr, fold)

            n_neg = (y_tr_sm == 0).sum()
            n_pos = (y_tr_sm == 1).sum()
            spw = n_neg / max(n_pos, 1)

            if model_name == "XGBoost":
                model = builder_fn(scale_pos_weight=spw)
                model.fit(X_tr_sm, y_tr_sm, eval_set=[(X_val, y_val)], verbose=False)
                oof[val_idx] = model.predict_proba(X_val)[:, 1]

            elif model_name == "LightGBM_Focal":
                # 使用 LightGBM native API + Focal Loss
                model = self._train_lgb_focal(X_tr_sm, y_tr_sm, X_val, y_val)
                # native API 的 predict 輸出是 raw score，需要 sigmoid
                raw = model.predict(X_val)
                oof[val_idx] = 1.0 / (1.0 + np.exp(-raw))

            elif model_name == "LightGBM":
                model = builder_fn(scale_pos_weight=spw)
                model.fit(X_tr_sm, y_tr_sm, eval_set=[(X_val, y_val)])
                oof[val_idx] = model.predict_proba(X_val)[:, 1]

            elif model_name == "CatBoost":
                model = builder_fn(scale_pos_weight=spw)
                model.fit(X_tr_sm, y_tr_sm, eval_set=(X_val, y_val))
                oof[val_idx] = model.predict_proba(X_val)[:, 1]
            else:
                model = builder_fn(scale_pos_weight=spw)
                model.fit(X_tr_sm, y_tr_sm)
                oof[val_idx] = model.predict_proba(X_val)[:, 1]

            fold_score = average_precision_score(y_val, oof[val_idx])
            scores.append(fold_score)
            all_models.append(model)
            print(f"  {model_name} Fold {fold+1}: AUC-PR = {fold_score:.4f}")

        best_model = all_models[int(np.argmax(scores))]
        avg_score = np.mean(scores)
        print(f"  {model_name} 平均 AUC-PR = {avg_score:.4f}")
        return oof, all_models, best_model, avg_score

    def _train_lgb_focal(self, X_tr, y_tr, X_val, y_val):
        """用 LightGBM native API 訓練 Focal Loss 模型"""
        params = build_lightgbm_params()
        # 把自訂 objective 放進 params
        params["objective"] = focal_loss_objective(
            alpha=self.focal_alpha, gamma=self.focal_gamma
        )

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=2500,
            valid_sets=[dval],
            feval=focal_loss_eval,
            callbacks=[
                lgb.early_stopping(150, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        return model

    def _predict_avg(self, models: List, X_scaled: np.ndarray, is_focal: bool = False) -> np.ndarray:
        """用所有 fold 模型做平均預測"""
        preds = np.zeros(len(X_scaled))
        for m in models:
            if is_focal:
                # Focal Loss native API: 輸出 raw score → sigmoid
                raw = m.predict(X_scaled)
                preds += 1.0 / (1.0 + np.exp(-raw))
            else:
                preds += m.predict_proba(X_scaled)[:, 1]
        return preds / len(models)

    def _build_meta_features(
        self,
        xgb_p: np.ndarray,
        lgb_p: np.ndarray,
        cat_p: np.ndarray,
    ) -> np.ndarray:
        """組合 Level-1 meta features: [xgb, lgb, cat, max, min, std]"""
        stack = np.column_stack([xgb_p, lgb_p, cat_p])
        meta = np.column_stack([
            stack,
            stack.max(axis=1),
            stack.min(axis=1),
            stack.std(axis=1),
            stack.mean(axis=1),
        ])
        return meta

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        gnn_proba: np.ndarray = None,
    ) -> "StackingEnsemble":
        X_scaled = self.scaler.fit_transform(X)

        if self.use_smote:
            print(f"  [SMOTE] Borderline-SMOTE enabled (strategy={self.smote_strategy})")
        if self.use_focal_loss:
            print(f"  [Focal Loss] LightGBM 使用 Focal Loss (α={self.focal_alpha}, γ={self.focal_gamma})")
        else:
            print(f"  [LightGBM] 使用 scale_pos_weight（無 Focal Loss）")

        # ── Level-0: 三個 base learners 的 OOF ──
        print("\n[1/4] 訓練 XGBoost OOF ...")
        xgb_oof, self.xgb_models, self.xgb_model, xgb_score = self._get_oof(
            X_scaled, y, build_xgboost, "XGBoost"
        )

        if self.use_focal_loss:
            print("\n[2/4] 訓練 LightGBM OOF (Focal Loss) ...")
            lgb_oof, self.lgb_models, self.lgb_model, lgb_score = self._get_oof(
                X_scaled, y, None, "LightGBM_Focal"
            )
        else:
            print("\n[2/4] 訓練 LightGBM OOF ...")
            lgb_oof, self.lgb_models, self.lgb_model, lgb_score = self._get_oof(
                X_scaled, y, build_lightgbm, "LightGBM"
            )

        print("\n[3/4] 訓練 CatBoost OOF ...")
        cat_oof, self.cat_models, self.cat_model, cat_score = self._get_oof(
            X_scaled, y, build_catboost, "CatBoost"
        )

        # 保存加權平均的權重（供 backward compat）
        total = xgb_score + lgb_score + cat_score
        self.weights = {
            "xgb": xgb_score / total,
            "lgb": lgb_score / total,
            "cat": cat_score / total,
        }
        print(f"\n  Base learner AUC-PR: XGB={xgb_score:.4f}, "
              f"LGB={lgb_score:.4f}, CAT={cat_score:.4f}")

        # ── Level-1: Meta-learner on OOF predictions ──
        print("\n[4/4] 訓練 Stacking meta-learner ...")
        meta_X = self._build_meta_features(xgb_oof, lgb_oof, cat_oof)
        meta_X_scaled = self.meta_scaler.fit_transform(meta_X)

        # Logistic Regression with class_weight='balanced' for imbalanced data
        self.meta_model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
        self.meta_model.fit(meta_X_scaled, y)

        # OOF prediction from meta-learner (using leave-one-out would be ideal,
        # but since meta features are already OOF, this is valid)
        oof_meta = self.meta_model.predict_proba(meta_X_scaled)[:, 1]

        self.use_gnn   = gnn_proba is not None
        self.is_fitted = True

        oof_ap = average_precision_score(y, oof_meta)
        print(f"\n  Stacking OOF AUC-PR = {oof_ap:.4f}")

        # 也顯示 simple averaging 的 OOF 作為對照
        oof_simple = (
            self.weights["xgb"] * xgb_oof +
            self.weights["lgb"] * lgb_oof +
            self.weights["cat"] * cat_oof
        )
        oof_simple_ap = average_precision_score(y, oof_simple)
        print(f"  (Simple Avg OOF AUC-PR = {oof_simple_ap:.4f})")

        # Meta-learner 的閾值（用 meta OOF 找）
        self.oof_threshold = _find_best_threshold_pr(y, oof_meta)

        # 顯示 meta-learner 學到的權重
        coefs = self.meta_model.coef_[0]
        names = ["XGB_p", "LGB_p", "CAT_p", "max_p", "min_p", "std_p", "mean_p"]
        print(f"  Meta-learner 權重:")
        for n, c in zip(names, coefs):
            print(f"    {n:<8}: {c:+.4f}")

        return self

    def predict_proba(
        self,
        X: np.ndarray,
        gnn_proba: np.ndarray = None,
    ) -> np.ndarray:
        X_scaled = self.scaler.transform(X)

        xgb_p = self._predict_avg(self.xgb_models, X_scaled)
        lgb_p = self._predict_avg(self.lgb_models, X_scaled, is_focal=self.use_focal_loss)
        cat_p = self._predict_avg(self.cat_models, X_scaled)

        meta_X = self._build_meta_features(xgb_p, lgb_p, cat_p)
        meta_X_scaled = self.meta_scaler.transform(meta_X)
        final_p = self.meta_model.predict_proba(meta_X_scaled)[:, 1]

        if self.use_gnn and gnn_proba is not None:
            final_p = 0.9 * final_p + 0.1 * gnn_proba

        return final_p

    def predict(self, X: np.ndarray, threshold: float = None, **kwargs) -> np.ndarray:
        if threshold is None:
            threshold = getattr(self, "oof_threshold", 0.5)
        return (self.predict_proba(X, **kwargs) >= threshold).astype(int)


# ─────────────────────────────────────────────
# 閾值搜索（PR-curve + F-beta）
# ─────────────────────────────────────────────

def _find_best_threshold_pr(y_true: np.ndarray, y_proba: np.ndarray, beta: float = 1.0) -> float:
    """用 Precision-Recall 曲線直接找最大 F-beta 的閾值"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    precision = precision[:-1]
    recall = recall[:-1]

    fbeta = ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall + 1e-9)
    best_idx = np.argmax(fbeta)
    best_t = float(thresholds[best_idx])
    print(f"  OOF 最佳閾值 = {best_t:.4f} (F{beta}={fbeta[best_idx]:.4f}, "
          f"P={precision[best_idx]:.4f}, R={recall[best_idx]:.4f})")
    return best_t


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
    """F1 最大化的最佳閾值搜索（PR-curve 法）"""
    return _find_best_threshold_pr(y_true, y_proba, beta=1.0)
