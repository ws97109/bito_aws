"""
小規模端到端測試：特徵工程 → 篩選 → 集成訓練 → SHAP → SSR
跳過 GNN，只用前 5000 筆用戶快速驗證
"""
import sys, json, os
sys.path.insert(0, "model")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from Feature_rngineering import build_all_features
from feature_selection import select_features
from ensemble import StackingEnsemble, evaluate, find_optimal_threshold
from shap_explainer import SHAPExplainer, CounterfactualExplainer, SSREvaluator

os.makedirs("output/test_run", exist_ok=True)

# ── 載入資料（取前 5000 筆用戶加速）──
print("=" * 55)
print("[1] 載入資料（取樣 5000 筆）")
print("=" * 55)
user_info = pd.read_csv("adjust_data/train/user_info_train.csv", low_memory=False)
# 確保包含足夠的黑名單用戶
black_ids = user_info[user_info["status"] == 1]["user_id"].values
normal_ids = user_info[user_info["status"] == 0]["user_id"].head(4000).values
sample_ids = set(np.concatenate([black_ids[:1000], normal_ids]))

user_info = user_info[user_info["user_id"].isin(sample_ids)]
twd     = pd.read_csv("adjust_data/train/twd_transfer_train.csv", low_memory=False)
crypto  = pd.read_csv("adjust_data/train/crypto_transfer_train.csv", low_memory=False)
trading = pd.read_csv("adjust_data/train/usdt_twd_trading_train.csv", low_memory=False)
swap    = pd.read_csv("adjust_data/train/usdt_swap_train.csv", low_memory=False)

twd     = twd[twd["user_id"].isin(sample_ids)]
crypto  = crypto[crypto["user_id"].isin(sample_ids)]
trading = trading[trading["user_id"].isin(sample_ids)]
swap    = swap[swap["user_id"].isin(sample_ids)]

print(f"  用戶數: {len(user_info)}, 黑名單: {(user_info['status']==1).sum()}")

# ── 特徵工程 ──
print("\n" + "=" * 55)
print("[2] 特徵工程")
print("=" * 55)
feat = build_all_features(user_info, twd, crypto, trading, swap)
labels = user_info.set_index("user_id")["status"]
y = labels.reindex(feat.index).fillna(0).astype(int).values
print(f"  特徵: {feat.shape[1]}, 用戶: {feat.shape[0]}")

# ── 特徵篩選 ──
print("\n" + "=" * 55)
print("[3] 特徵篩選")
print("=" * 55)
X_selected, report = select_features(feat, y)
feature_names = X_selected.columns.tolist()
X = X_selected.values.astype(np.float32)
print(f"  篩選後: {X.shape[1]} 個特徵")

# ── Train/Test 分割 ──
print("\n" + "=" * 55)
print("[4] Train/Test 分割")
print("=" * 55)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"  訓練: {len(X_tr)} (黑名單 {y_tr.sum()})")
print(f"  測試: {len(X_te)} (黑名單 {y_te.sum()})")

# ── 集成訓練 ──
print("\n" + "=" * 55)
print("[5] 集成訓練（XGBoost + LightGBM + CatBoost + IsoForest）")
print("=" * 55)
ensemble = StackingEnsemble(n_splits=3)  # 少量資料用 3-fold
ensemble.fit(X_tr, y_tr)

# ── 評估 ──
print("\n" + "=" * 55)
print("[6] 評估")
print("=" * 55)
y_proba = ensemble.predict_proba(X_te)
metrics = evaluate(y_te, y_proba, label="Stacking Ensemble")
optimal_t = find_optimal_threshold(y_te, y_proba)
# 用最佳閾值重新評估
metrics_opt = evaluate(y_te, y_proba, threshold=optimal_t, label=f"Optimal Threshold={optimal_t:.2f}")

# ── SHAP ──
print("\n" + "=" * 55)
print("[7] SHAP 分析")
print("=" * 55)
explainer = SHAPExplainer(ensemble.xgb_model, feature_names)
X_te_scaled = ensemble.scaler.transform(X_te)
bg_n = min(100, len(X_te_scaled))
explainer.fit(X_te_scaled[:bg_n], X_te_scaled)
explainer.plot_global_importance(
    top_n=15,
    save_path="output/test_run/shap_global.png",
)

# ── SSR（少量樣本快速測試）──
print("\n" + "=" * 55)
print("[8] SSR 穩定性（少量測試）")
print("=" * 55)
ssr_eval = SSREvaluator(explainer, feature_names)
ssr_results = ssr_eval.evaluate(
    X_te_scaled, y_te,
    epsilons=[0.05, 0.10],
    top_k_list=[1, 3],
    n_samples=50,
    n_perturbations=5,
)
ssr_eval.plot_ssr_curves(ssr_results, save_path="output/test_run/ssr_curves.png")

print("\n" + "=" * 55)
print("測試完成！所有模組正常運作")
print("=" * 55)
