"""
端到端測試 + 消融實驗：比較 Baseline vs Phase 1 優化
1. Baseline: 原始 scale_pos_weight（v3 行為）
2. + Anomaly Scores: 加入 IF/HBOS/LOF 異常分數
3. + Focal Loss: LightGBM 改用 Focal Loss
4. + SMOTE: Borderline-SMOTE
5. Full Phase 1: 全部啟用
"""
import sys, json, os, time

# Wei_model/ → 專案根目錄
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_SCRIPT_DIR)

sys.path.insert(0, os.path.join(_SCRIPT_DIR, "model"))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from Feature_rngineering import build_all_features
from feature_selection import select_features
from anomaly_detection import AnomalyFeatureExtractor, add_anomaly_scores_to_splits
from ensemble import StackingEnsemble, evaluate, find_optimal_threshold
from shap_explainer import SHAPExplainer, CounterfactualExplainer, SSREvaluator

DATA_DIR = os.path.join(ROOT, "adjust_data", "train")
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "output", "test_run")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════
# 資料載入 + 特徵工程（共用）
# ══════════════════════════════════════════════════
print("=" * 60)
print("[1] 載入資料（取樣 5000 筆）")
print("=" * 60)
user_info = pd.read_csv(os.path.join(DATA_DIR, "user_info_train.csv"), low_memory=False)
black_ids = user_info[user_info["status"] == 1]["user_id"].values
normal_ids = user_info[user_info["status"] == 0]["user_id"].head(4000).values
sample_ids = set(np.concatenate([black_ids[:1000], normal_ids]))

user_info = user_info[user_info["user_id"].isin(sample_ids)]
twd     = pd.read_csv(os.path.join(DATA_DIR, "twd_transfer_train.csv"), low_memory=False)
crypto  = pd.read_csv(os.path.join(DATA_DIR, "crypto_transfer_train.csv"), low_memory=False)
trading = pd.read_csv(os.path.join(DATA_DIR, "usdt_twd_trading_train.csv"), low_memory=False)
swap    = pd.read_csv(os.path.join(DATA_DIR, "usdt_swap_train.csv"), low_memory=False)

twd     = twd[twd["user_id"].isin(sample_ids)]
crypto  = crypto[crypto["user_id"].isin(sample_ids)]
trading = trading[trading["user_id"].isin(sample_ids)]
swap    = swap[swap["user_id"].isin(sample_ids)]

n_black = (user_info["status"] == 1).sum()
n_total = len(user_info)
print(f"  用戶數: {n_total}, 黑名單: {n_black} ({n_black/n_total*100:.1f}%)")

print("\n" + "=" * 60)
print("[2] 特徵工程")
print("=" * 60)
feat = build_all_features(user_info, twd, crypto, trading, swap)
labels = user_info.set_index("user_id")["status"]
y = labels.reindex(feat.index).fillna(0).astype(int).values
print(f"  原始特徵: {feat.shape[1]}")

print("\n" + "=" * 60)
print("[3] 特徵篩選")
print("=" * 60)
X_selected, report = select_features(feat, y)
feature_names_base = X_selected.columns.tolist()
X_base = X_selected.values.astype(np.float32)
print(f"  篩選後: {X_base.shape[1]} 個特徵")

print("\n" + "=" * 60)
print("[4] Train/Test 分割（共用，確保公平比較）")
print("=" * 60)
X_tr_base, X_te_base, y_tr, y_te = train_test_split(
    X_base, y, test_size=0.2, stratify=y, random_state=42
)
print(f"  訓練: {len(X_tr_base)} (黑名單 {y_tr.sum()})")
print(f"  測試: {len(X_te_base)} (黑名單 {y_te.sum()})")


# ══════════════════════════════════════════════════
# 消融實驗
# ══════════════════════════════════════════════════

all_results = {}

def run_experiment(name, X_tr, X_te, use_focal_loss=False, use_smote=False):
    """執行單一實驗配置"""
    print(f"\n{'='*60}")
    print(f"  實驗: {name}")
    print(f"  特徵維度: {X_tr.shape[1]}, Focal Loss: {use_focal_loss}, SMOTE: {use_smote}")
    print(f"{'='*60}")

    t0 = time.time()
    ens = StackingEnsemble(
        n_splits=3,
        use_focal_loss=use_focal_loss,
        use_smote=use_smote,
        smote_strategy=0.3,
    )
    ens.fit(X_tr, y_tr)
    proba = ens.predict_proba(X_te)

    # 用 OOF 閾值
    metrics = evaluate(y_te, proba, threshold=ens.oof_threshold, label=name)
    elapsed = time.time() - t0
    metrics["threshold"] = float(ens.oof_threshold)
    metrics["time_sec"] = round(elapsed, 1)

    all_results[name] = metrics
    return ens, proba


# ── Exp 1: Baseline (v3 行為) ──
ens_baseline, _ = run_experiment(
    "Baseline (scale_pos_weight only)",
    X_tr_base, X_te_base,
    use_focal_loss=False, use_smote=False,
)

# ── Exp 2: + Anomaly Scores ──
print(f"\n{'='*60}")
print("[非監督異常偵測] 加入 IF/HBOS/LOF 異常分數")
print(f"{'='*60}")
X_tr_anom, X_te_anom, anomaly_ext = add_anomaly_scores_to_splits(X_tr_base, X_te_base)
feature_names_anom = feature_names_base + AnomalyFeatureExtractor.get_feature_names()

ens_anom, _ = run_experiment(
    "+ Anomaly Scores",
    X_tr_anom, X_te_anom,
    use_focal_loss=False, use_smote=False,
)

# ── Exp 3: + Anomaly + Focal Loss ──
ens_focal, _ = run_experiment(
    "+ Anomaly + Focal Loss",
    X_tr_anom, X_te_anom,
    use_focal_loss=True, use_smote=False,
)

# ── Exp 4: + Anomaly + Focal Loss + SMOTE ──
ens_full, proba_full = run_experiment(
    "Full Phase 1 (Anomaly + Focal + SMOTE)",
    X_tr_anom, X_te_anom,
    use_focal_loss=True, use_smote=True,
)


# ══════════════════════════════════════════════════
# 結果彙整
# ══════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  消融實驗結果比較")
print("=" * 60)

header = f"{'配置':<42} {'F1':>6} {'AUC-PR':>7} {'Prec':>6} {'Rec':>6} {'AUC-ROC':>8} {'閾值':>6} {'時間':>6}"
print(header)
print("-" * len(header))

for name, m in all_results.items():
    print(f"{name:<42} {m['F1']:>6.4f} {m['AUC-PR']:>7.4f} {m['Precision']:>6.4f} "
          f"{m['Recall']:>6.4f} {m['AUC-ROC']:>8.4f} {m['threshold']:>6.3f} {m['time_sec']:>5.1f}s")

# 計算提升
baseline_f1 = all_results["Baseline (scale_pos_weight only)"]["F1"]
baseline_ap = all_results["Baseline (scale_pos_weight only)"]["AUC-PR"]

print(f"\n--- 相對 Baseline 提升 ---")
for name, m in all_results.items():
    if name == "Baseline (scale_pos_weight only)":
        continue
    f1_delta = m["F1"] - baseline_f1
    ap_delta = m["AUC-PR"] - baseline_ap
    print(f"  {name}: F1 {f1_delta:+.4f}, AUC-PR {ap_delta:+.4f}")


# ── 儲存結果 ──
with open(os.path.join(OUTPUT_DIR, "ablation_results.json"), "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)


# ── SHAP（用最佳配置）──
print(f"\n{'='*60}")
print("[SHAP] 可解釋性分析（使用最佳配置）")
print(f"{'='*60}")

# 選 F1 最高的配置
best_name = max(all_results, key=lambda k: all_results[k]["F1"])
print(f"  最佳配置: {best_name}")

# 用最後一個 ensemble 做 SHAP
best_ens = ens_full
best_features = feature_names_anom

explainer = SHAPExplainer(best_ens.xgb_model, best_features)
X_te_scaled = best_ens.scaler.transform(X_te_anom)
bg_n = min(100, len(X_te_scaled))
explainer.fit(X_te_scaled[:bg_n], X_te_scaled)
explainer.plot_global_importance(
    top_n=20,
    save_path=os.path.join(OUTPUT_DIR, "shap_global.png"),
)

# ── SSR ──
print(f"\n{'='*60}")
print("[SSR] 穩定性評測（快速版）")
print(f"{'='*60}")
ssr_eval = SSREvaluator(explainer, best_features)
ssr_results = ssr_eval.evaluate(
    X_te_scaled, y_te,
    epsilons=[0.05, 0.10],
    top_k_list=[1, 3],
    n_samples=50,
    n_perturbations=5,
)
ssr_eval.plot_ssr_curves(ssr_results, save_path=os.path.join(OUTPUT_DIR, "ssr_curves.png"))


print(f"\n{'='*60}")
print("  測試完成！")
print(f"{'='*60}")
print(f"  消融實驗結果: {OUTPUT_DIR}/ablation_results.json")
print(f"  SHAP 圖: {OUTPUT_DIR}/shap_global.png")
print(f"  SSR 曲線: {OUTPUT_DIR}/ssr_curves.png")
