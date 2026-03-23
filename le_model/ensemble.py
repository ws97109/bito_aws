"""
Ensemble Model v4 — 基於 2025-2026 最新文獻，目標 F1(黑名單) ≥ 0.70
================================================================
核心改進（對比目前 F1=0.28 / Precision=0.22 / Recall=0.37）：

問題診斷：
  Precision=0.22 → 過採樣太激進，模型誤抓大量正常用戶
  Recall=0.37    → 模型仍學不好黑名單特徵
  F1=0.28        → 兩邊都差，根本是超參沒調

文獻依據：
  [1] Optuna 超參數調優 (arxiv 2505.10050, 2025)
      → XGBoost+CatBoost+LightGBM + Optuna 20 trials = F1=0.99
  [2] SMOTE 只用 10-15%（Scientific Reports RABEM 2025）
      → 過度過採樣反而製造噪點降低 Precision
  [3] PR 曲線找最優閾值（同上）
      → 最優閾值 ~0.44，比 0.5 更好但非盲目調低
  [4] Ethereum 黑名單：RF + XGBoost + NN + SMOTE-ENN + Bayesian (PeerJ 2025)
  [5] Soft Voting (RF+XGB+MLP) ADASYN: F1=0.8764 (PMC 2024)

策略：
  1. Optuna 先針對每個模型獨立調超參數（F1 為目標）
  2. SMOTE 比例保守（10%），避免 Precision 崩潰
  3. Base Learner: CatBoost + XGBoost + LightGBM + RandomForest
  4. Meta Learner: LR (簡單，避免過擬合)
  5. PR 曲線找最大 F1 閾值
================================================================
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    precision_recall_curve,
)
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[警告] optuna 未安裝，使用預設超參數。建議: pip install optuna")


# ═══════════════════════════════════════════════════════════
# 重採樣：保守 SMOTE（10-15%，避免 Precision 崩潰）
# ═══════════════════════════════════════════════════════════

def resample_conservative(
    X: np.ndarray,
    y: np.ndarray,
    target_ratio: float = 0.12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    保守過採樣策略：
    - 只用 SMOTE，不加激進的 ENN 清洗（ENN 會移除太多邊界樣本降低 Precision）
    - target_ratio 設 0.12（12%）而非 0.5（50%），符合 2025 RABEM 論文建議
    """
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    target_n = max(int(n_neg * target_ratio / (1 - target_ratio)), n_pos + 1)

    print(f"    採樣前：正常 {n_neg:,}  黑名單 {n_pos:,}  比率 {n_pos/(n_pos+n_neg)*100:.1f}%")

    try:
        smote = SMOTE(
            sampling_strategy={1: target_n},
            k_neighbors=min(5, n_pos - 1),
            random_state=42,
        )
        X_res, y_res = smote.fit_resample(X, y)
    except Exception as e:
        print(f"    SMOTE 失敗({e})，使用原始資料")
        return X, y

    n_pr = int((y_res == 1).sum())
    n_nr = int((y_res == 0).sum())
    print(f"    採樣後：正常 {n_nr:,}  黑名單 {n_pr:,}  比率 {n_pr/(n_pr+n_nr)*100:.1f}%")
    return X_res, y_res


# ═══════════════════════════════════════════════════════════
# Optuna 超參數調優（目標：最大化 F1 黑名單）
# ═══════════════════════════════════════════════════════════

def optuna_tune_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 30,
) -> dict:
    if not OPTUNA_AVAILABLE:
        return _default_xgb_params(y)

    spw = (y == 0).sum() / max((y == 1).sum(), 1)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 5.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", spw * 0.5, spw * 2.0),
            "eval_metric":      "aucpr",
            "tree_method":      "hist",
            "device":           "cpu",
            "random_state":     42,
        }
        model = xgb.XGBClassifier(**params)
        scores = cross_val_score(
            model, X, y, cv=3,
            scoring="f1",
            fit_params={"verbose": False},
        )
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"    XGBoost 最佳 F1={study.best_value:.4f} 超參數={study.best_params}")
    p = study.best_params
    p.update({"eval_metric": "aucpr", "tree_method": "hist",
               "device": "cpu", "random_state": 42,
               "early_stopping_rounds": 50})
    return p


def optuna_tune_catboost(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 20,
) -> dict:
    if not OPTUNA_AVAILABLE:
        return _default_cat_params()

    def objective(trial):
        params = {
            "iterations":         trial.suggest_int("iterations", 200, 800),
            "depth":              trial.suggest_int("depth", 4, 8),
            "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg":        trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "border_count":       trial.suggest_categorical("border_count", [64, 128, 254]),
            "auto_class_weights": "Balanced",
            "eval_metric":        "AUC",
            "random_seed":        42,
            "verbose":            False,
            "task_type":          "CPU",
        }
        model = CatBoostClassifier(**params)
        scores = cross_val_score(model, X, y, cv=3, scoring="f1")
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"    CatBoost 最佳 F1={study.best_value:.4f}")
    p = study.best_params
    p.update({"auto_class_weights": "Balanced", "eval_metric": "AUC",
               "random_seed": 42, "verbose": False, "task_type": "CPU",
               "early_stopping_rounds": 50})
    return p


def optuna_tune_lgbm(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 20,
) -> dict:
    if not OPTUNA_AVAILABLE:
        return _default_lgb_params()

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 800),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 100),
            "max_depth":         trial.suggest_int("max_depth", 4, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "is_unbalance":      True,
            "n_jobs":            -1,
            "random_state":      42,
            "verbose":           -1,
        }
        model = lgb.LGBMClassifier(**params)
        scores = cross_val_score(model, X, y, cv=3, scoring="f1")
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"    LightGBM 最佳 F1={study.best_value:.4f}")
    p = study.best_params
    p.update({"is_unbalance": True, "n_jobs": -1, "random_state": 42, "verbose": -1})
    return p


# 預設超參數（無 Optuna 時使用）
def _default_xgb_params(y) -> dict:
    spw = max((y == 0).sum() / max((y == 1).sum(), 1), 1)
    return {
        "n_estimators": 600, "max_depth": 6, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 3,
        "reg_alpha": 0.1, "reg_lambda": 1.0, "scale_pos_weight": spw,
        "eval_metric": "aucpr", "tree_method": "hist", "device": "cpu",
        "random_state": 42, "early_stopping_rounds": 50,
    }

def _default_cat_params() -> dict:
    return {
        "iterations": 500, "depth": 6, "learning_rate": 0.05,
        "l2_leaf_reg": 3.0, "auto_class_weights": "Balanced",
        "eval_metric": "AUC", "random_seed": 42, "verbose": False,
        "task_type": "CPU", "early_stopping_rounds": 50,
    }

def _default_lgb_params() -> dict:
    return {
        "n_estimators": 500, "num_leaves": 50, "max_depth": 6,
        "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.7,
        "is_unbalance": True, "n_jobs": -1, "random_state": 42, "verbose": -1,
    }


# ═══════════════════════════════════════════════════════════
# PR 曲線找最優閾值（文獻建議 ~0.44，非盲目調低）
# ═══════════════════════════════════════════════════════════

def find_best_threshold_pr(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> Tuple[float, float]:
    """
    用 sklearn 的 precision_recall_curve 直接計算每個閾值的 F1，
    找到最大 F1 對應的閾值（2025 文獻最佳實務）
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # 計算每個閾值的 F1（避免除零）
    f1s = np.where(
        (precisions + recalls) > 0,
        2 * precisions * recalls / (precisions + recalls),
        0.0
    )
    best_idx = np.argmax(f1s[:-1])   # 最後一個 threshold 對應 recall=0
    best_t   = float(thresholds[best_idx])
    best_f1  = float(f1s[best_idx])
    best_pr  = float(precisions[best_idx])
    best_rc  = float(recalls[best_idx])
    print(f"\n  PR 曲線最優閾值: {best_t:.3f}  "
          f"F1={best_f1:.4f}  Precision={best_pr:.4f}  Recall={best_rc:.4f}")
    return best_t, best_f1


# ═══════════════════════════════════════════════════════════
# 主集成
# ═══════════════════════════════════════════════════════════

class StackingEnsemble:
    """
    四基學習器 Stacking（基於 2025-2026 最優論文架構）：

    Base Learners（每個均有 Optuna 調超參）：
      - CatBoost（auto_class_weights=Balanced）
      - XGBoost（scale_pos_weight=IR）
      - LightGBM（is_unbalance=True）
      - RandomForest（class_weight=balanced）

    重採樣：保守 SMOTE 10-12%（只在訓練 fold 內做，防 leakage）
    Meta Learner：LR（calibrated，避免過擬合）
    閾值：PR 曲線找最大 F1 閾值
    """

    def __init__(
        self,
        n_splits: int = 5,
        smote_ratio: float = 0.12,
        optuna_trials_xgb: int = 30,
        optuna_trials_cat: int = 20,
        optuna_trials_lgb: int = 20,
        use_optuna: bool = True,
    ):
        self.n_splits           = n_splits
        self.smote_ratio        = smote_ratio
        self.optuna_trials_xgb  = optuna_trials_xgb
        self.optuna_trials_cat  = optuna_trials_cat
        self.optuna_trials_lgb  = optuna_trials_lgb
        self.use_optuna         = use_optuna and OPTUNA_AVAILABLE

        # 模型容器
        self.cat_models: List[CatBoostClassifier]  = []
        self.xgb_models: List[xgb.XGBClassifier]   = []
        self.lgb_models: List[lgb.LGBMClassifier]  = []
        self.rf_models:  List[RandomForestClassifier] = []
        self.meta_model: Optional[CalibratedClassifierCV] = None
        self.iso_model:  Optional[IsolationForest]         = None

        self.scaler            = RobustScaler()
        self.optimal_threshold = 0.44   # PR 曲線初始值
        self.use_gnn           = False
        self.is_fitted         = False

        # 調好的超參數
        self._xgb_params: dict = {}
        self._cat_params: dict = {}
        self._lgb_params: dict = {}

    # ── 內部：單模型 OOF ─────────────────────────────────

    def _oof(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str,
    ) -> np.ndarray:
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        oof = np.zeros(len(y))

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            # 每 fold 獨立 SMOTE（防 data leakage）
            X_tr_r, y_tr_r = resample_conservative(
                X_tr, y_tr, target_ratio=self.smote_ratio
            )
            spw = max((y_tr_r == 0).sum() / max((y_tr_r == 1).sum(), 1), 1.0)

            if model_type == "catboost":
                p = dict(self._cat_params)
                m = CatBoostClassifier(**p)
                m.fit(X_tr_r, y_tr_r, eval_set=(X_val, y_val))
                prob = m.predict_proba(X_val)[:, 1]
                self.cat_models.append(m)

            elif model_type == "xgboost":
                p = dict(self._xgb_params)
                p["scale_pos_weight"] = spw
                m = xgb.XGBClassifier(**p)
                m.fit(X_tr_r, y_tr_r, eval_set=[(X_val, y_val)], verbose=False)
                prob = m.predict_proba(X_val)[:, 1]
                self.xgb_models.append(m)

            elif model_type == "lightgbm":
                p = dict(self._lgb_params)
                m = lgb.LGBMClassifier(**p)
                m.fit(X_tr_r, y_tr_r, eval_set=[(X_val, y_val)])
                prob = m.predict_proba(X_val)[:, 1]
                self.lgb_models.append(m)

            elif model_type == "randomforest":
                m = RandomForestClassifier(
                    n_estimators=300, max_depth=10,
                    class_weight="balanced", n_jobs=-1, random_state=42,
                )
                m.fit(X_tr_r, y_tr_r)
                prob = m.predict_proba(X_val)[:, 1]
                self.rf_models.append(m)

            oof[val_idx] = prob
            pr = average_precision_score(y_val, prob)
            f1 = f1_score(y_val, (prob >= 0.44).astype(int), zero_division=0)
            print(f"    Fold {fold+1}: AUC-PR={pr:.4f}  F1@0.44={f1:.4f}")

        return oof

    # ── 公開：訓練 ───────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        gnn_proba: Optional[np.ndarray] = None,
    ) -> "StackingEnsemble":

        print("\n[1/6] RobustScaler 標準化")
        X_s = self.scaler.fit_transform(X)

        # 用整個訓練集做 Optuna 調超參（只做一次）
        print("\n[2/6] Optuna 超參數調優（目標：最大化 F1 黑名單）")
        X_res_tune, y_res_tune = resample_conservative(X_s, y, self.smote_ratio)

        if self.use_optuna:
            print("  調優 XGBoost ...")
            self._xgb_params = optuna_tune_xgboost(
                X_res_tune, y_res_tune, self.optuna_trials_xgb
            )
            print("  調優 CatBoost ...")
            self._cat_params = optuna_tune_catboost(
                X_res_tune, y_res_tune, self.optuna_trials_cat
            )
            print("  調優 LightGBM ...")
            self._lgb_params = optuna_tune_lgbm(
                X_res_tune, y_res_tune, self.optuna_trials_lgb
            )
        else:
            self._xgb_params = _default_xgb_params(y)
            self._cat_params = _default_cat_params()
            self._lgb_params = _default_lgb_params()

        print("\n[3/6] CatBoost OOF ...")
        cat_oof = self._oof(X_s, y, "catboost")

        print("\n[4/6] XGBoost OOF ...")
        xgb_oof = self._oof(X_s, y, "xgboost")

        print("\n[5/6] LightGBM OOF ...")
        lgb_oof = self._oof(X_s, y, "lightgbm")

        print("\n[6/6] RandomForest OOF + Isolation Forest + Meta Learner ...")
        rf_oof  = self._oof(X_s, y, "randomforest")

        self.iso_model = IsolationForest(
            n_estimators=300, contamination=0.02,
            random_state=42, n_jobs=-1,
        )
        self.iso_model.fit(X_s)
        iso_s = self._iso_score(X_s)

        # Meta 特徵：4 模型 OOF + Isolation Forest 分數
        meta_X = np.column_stack([cat_oof, xgb_oof, lgb_oof, rf_oof, iso_s])
        if gnn_proba is not None:
            meta_X = np.column_stack([meta_X, gnn_proba])
            self.use_gnn = True

        # Meta LR（Calibrated，isotonic 校準，比 Platt 更精確）
        self.meta_model = CalibratedClassifierCV(
            LogisticRegression(
                C=1.0, class_weight="balanced",
                max_iter=1000, solver="lbfgs",
            ),
            method="isotonic", cv=3,
        )
        self.meta_model.fit(meta_X, y)

        # PR 曲線找最優閾值
        oof_proba = self.meta_model.predict_proba(meta_X)[:, 1]
        self.optimal_threshold, _ = find_best_threshold_pr(y, oof_proba)
        self.is_fitted = True
        return self

    def _iso_score(self, X_s: np.ndarray) -> np.ndarray:
        raw = -self.iso_model.decision_function(X_s)
        rng = raw.max() - raw.min()
        return (raw - raw.min()) / (rng + 1e-9)

    # ── 公開：推論 ───────────────────────────────────────

    def predict_proba(
        self,
        X: np.ndarray,
        gnn_proba: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        X_s = self.scaler.transform(X)
        cat_p = np.mean([m.predict_proba(X_s)[:, 1] for m in self.cat_models], axis=0)
        xgb_p = np.mean([m.predict_proba(X_s)[:, 1] for m in self.xgb_models], axis=0)
        lgb_p = np.mean([m.predict_proba(X_s)[:, 1] for m in self.lgb_models], axis=0)
        rf_p  = np.mean([m.predict_proba(X_s)[:, 1] for m in self.rf_models],  axis=0)
        iso_s = self._iso_score(X_s)

        meta_X = np.column_stack([cat_p, xgb_p, lgb_p, rf_p, iso_s])
        if self.use_gnn and gnn_proba is not None:
            meta_X = np.column_stack([meta_X, gnn_proba])

        return self.meta_model.predict_proba(meta_X)[:, 1]

    def predict(
        self,
        X: np.ndarray,
        threshold: Optional[float] = None,
        **kwargs,
    ) -> np.ndarray:
        t = threshold if threshold is not None else self.optimal_threshold
        return (self.predict_proba(X, **kwargs) >= t).astype(int)


# ═══════════════════════════════════════════════════════════
# 評估
# ═══════════════════════════════════════════════════════════

def evaluate(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    label: str = "Model",
) -> Dict:
    y_pred = (y_proba >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    metrics = {
        "AUC-ROC":   roc_auc_score(y_true, y_proba),
        "AUC-PR":    average_precision_score(y_true, y_proba),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "FPR":       fp / max(tn + fp, 1),
    }

    print(f"\n{'='*52}")
    print(f"  {label}  (threshold={threshold:.3f})")
    print(f"{'='*52}")
    for k, v in metrics.items():
        bar  = "█" * int(v * 25)
        flag = ""
        if k == "F1"        and v < 0.5:  flag = " ⚠ 目標≥0.70"
        if k == "Precision" and v < 0.5:  flag = " ⚠ 過採樣過強"
        if k == "Recall"    and v < 0.5:  flag = " ⚠ 漏抓黑名單"
        print(f"  {k:<12}: {v:.4f}  {bar}{flag}")

    print(f"\n  混淆矩陣：TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  黑名單：抓到 {tp}/{tp+fn}  漏掉 {fn}/{tp+fn}")
    print(f"  正常用戶誤判：{fp}/{tn+fp} ({fp/max(tn+fp,1)*100:.1f}%)\n")
    print(classification_report(
        y_true, y_pred,
        target_names=["正常", "黑名單"],
        zero_division=0,
    ))
    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_recall: float = 0.0,
) -> float:
    """PR 曲線找最大 F1 閾值，同時列出完整掃描表"""
    best_t, best_f1 = find_best_threshold_pr(y_true, y_proba)

    print("\n  閾值掃描（0.10 ~ 0.70）：")
    print("  閾值    Recall  Precision   F1     FPR")
    for t in np.arange(0.10, 0.71, 0.05):
        y_pred = (y_proba >= t).astype(int)
        rc  = recall_score(y_true, y_pred, zero_division=0)
        pr  = precision_score(y_true, y_pred, zero_division=0)
        f1  = f1_score(y_true, y_pred, zero_division=0)
        fpr = (y_pred[y_true == 0]).mean()
        mark = " ◄ 最優" if abs(t - best_t) < 0.03 else ""
        print(f"  {t:.2f}    {rc:.3f}   {pr:.3f}      {f1:.3f}  {fpr:.3f}{mark}")

    return best_t