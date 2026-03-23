"""
測試混合模型 - 結合 Ensemble 和黑名單學習器
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import os

from ensemble import StackingEnsemble
from blacklist_learner import BlacklistLearner
from hybrid_model import HybridBlacklistDetector, test_all_strategies


def main():
    print("\n" + "="*80)
    print("  混合模型系統 - 完整測試")
    print("="*80)

    # 1. 載入數據
    print("\n[1] 載入特徵數據...")
    feat_df = pd.read_csv("output/features.csv", index_col=0)

    if 'status' not in feat_df.columns:
        print("  [錯誤] 找不到 'status' 欄位")
        return

    y = feat_df['status'].values
    X = feat_df.drop(columns=['status']).values
    feature_names = feat_df.drop(columns=['status']).columns.tolist()

    print(f"  總用戶數：{len(X):,}")
    print(f"  黑名單數：{y.sum():,} ({y.mean()*100:.2f}%)")

    # 2. 分割訓練/測試集
    print("\n[2] 分割訓練/測試集（8:2）...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  訓練集：{len(X_train):,}（黑名單 {y_train.sum()}）")
    print(f"  測試集：{len(X_test):,}（黑名單 {y_test.sum()}）")

    # 3. 訓練或載入 Ensemble 模型
    print("\n[3] 載入/訓練 Ensemble 模型...")

    ensemble_path = "output/ensemble_model.pkl"

    if os.path.exists(ensemble_path):
        print("  → 載入已訓練的 Ensemble 模型...")
        try:
            with open(ensemble_path, 'rb') as f:
                ensemble = pickle.load(f)
            print("  ✓ Ensemble 模型載入成功")
        except Exception as e:
            print(f"  ✗ 載入失敗：{e}")
            print("  → 重新訓練 Ensemble 模型...")
            ensemble = train_ensemble(X_train, y_train)
    else:
        print("  → 訓練新的 Ensemble 模型...")
        ensemble = train_ensemble(X_train, y_train)

    # 4. 訓練黑名單學習器
    print("\n[4] 訓練黑名單學習器...")
    bl_learner = BlacklistLearner(n_clusters=5, contamination=0.05, n_neighbors=10)
    bl_learner.fit(X_train, y_train, feature_names)

    # 5. 測試所有融合策略
    print("\n[5] 測試所有融合策略...")
    results, best_strategy = test_all_strategies(
        ensemble, bl_learner,
        X_test, y_test,
        gnn_proba=None
    )

    # 6. 使用最佳策略進行詳細分析
    print("\n" + "="*80)
    print(f"  使用最佳策略 ({best_strategy.upper()}) 進行詳細分析")
    print("="*80)

    hybrid = HybridBlacklistDetector(
        ensemble, bl_learner,
        fusion_strategy=best_strategy,
        ensemble_weight=0.7,
        bl_weight=0.3,
    )
    hybrid.optimal_threshold = results[best_strategy]['optimal_threshold']

    # 7. 高風險用戶解釋
    print("\n[6] 生成高風險用戶解釋報告...")

    result = hybrid.predict_proba(X_test)
    top_risk_indices = result['hybrid_score'].argsort()[-10:][::-1]

    print(f"\nTop 10 高風險用戶：")
    print("="*80)

    for rank, idx in enumerate(top_risk_indices, 1):
        explanation = hybrid.explain_prediction(idx, X_test)

        true_label = "黑名單" if y_test[idx] == 1 else "正常"
        pred_label = "黑名單" if explanation['prediction'] == 1 else "正常"
        correct = "✓" if (y_test[idx] == explanation['prediction']) else "✗"

        print(f"\n{rank}. 用戶索引 {idx}")
        print(f"   真實: {true_label}  |  預測: {pred_label}  {correct}")
        print(f"   {explanation['explanation']}")

        if len(explanation['similar_blacklists']) > 0:
            print(f"   最相似的黑名單: {explanation['similar_blacklists'][:3]}")

    # 8. 儲存結果
    print("\n" + "="*80)
    print("  [7] 儲存結果")
    print("="*80)

    # 儲存混合模型
    with open("output/hybrid_model.pkl", 'wb') as f:
        pickle.dump(hybrid, f)
    print("  ✓ 混合模型已儲存: output/hybrid_model.pkl")

    # 儲存預測結果
    result_df = pd.DataFrame({
        'user_idx': np.arange(len(y_test)),
        'true_label': y_test,
        'ensemble_score': result['ensemble_score'],
        'bl_score': result['bl_score'],
        'hybrid_score': result['hybrid_score'],
        'confidence': result['confidence'],
        'prediction': (result['hybrid_score'] >= hybrid.optimal_threshold).astype(int),
        'closest_cluster': result['closest_cluster'],
    })
    result_df = result_df.sort_values('hybrid_score', ascending=False)
    result_df.to_csv("output/hybrid_predictions.csv", index=False)
    print("  ✓ 預測結果已儲存: output/hybrid_predictions.csv")

    # 9. 總結報告
    print("\n" + "="*80)
    print("  總結報告")
    print("="*80)

    best_metrics = results[best_strategy]

    print(f"\n最佳融合策略: {best_strategy.upper()}")
    print(f"  最佳閾值: {best_metrics['optimal_threshold']:.2f}")
    print(f"\n效能指標:")
    print(f"  Precision : {best_metrics['precision']:.4f} ({best_metrics['precision']*100:.1f}%)")
    print(f"  Recall    : {best_metrics['recall']:.4f} ({best_metrics['recall']*100:.1f}%)")
    print(f"  F1 Score  : {best_metrics['f1']:.4f}")
    print(f"  AUC-ROC   : {best_metrics['auc_roc']:.4f}")

    print(f"\n混淆矩陣:")
    print(f"  TP={best_metrics['tp']}  FP={best_metrics['fp']}")
    print(f"  FN={best_metrics['fn']}  TN={best_metrics['tn']}")

    # 與單一模型對比
    print(f"\n與單一模型對比:")

    # Ensemble 單獨
    ensemble_pred = (result['ensemble_score'] >= 0.5).astype(int)
    from sklearn.metrics import f1_score, precision_score, recall_score
    ensemble_f1 = f1_score(y_test, ensemble_pred, zero_division=0)
    ensemble_pr = precision_score(y_test, ensemble_pred, zero_division=0)
    ensemble_rc = recall_score(y_test, ensemble_pred, zero_division=0)

    # 黑名單學習器單獨
    bl_pred = (result['bl_score'] >= 0.4).astype(int)
    bl_f1 = f1_score(y_test, bl_pred, zero_division=0)
    bl_pr = precision_score(y_test, bl_pred, zero_division=0)
    bl_rc = recall_score(y_test, bl_pred, zero_division=0)

    print(f"\n  {'模型':<20} {'Precision':<12} {'Recall':<10} {'F1'}")
    print("  " + "-"*50)
    print(f"  {'Ensemble':<20} {ensemble_pr:<12.4f} {ensemble_rc:<10.4f} {ensemble_f1:.4f}")
    print(f"  {'黑名單學習器':<20} {bl_pr:<12.4f} {bl_rc:<10.4f} {bl_f1:.4f}")
    print(f"  {'混合模型 ★':<20} {best_metrics['precision']:<12.4f} {best_metrics['recall']:<10.4f} {best_metrics['f1']:.4f}")

    improvement = (best_metrics['f1'] - max(ensemble_f1, bl_f1)) / max(ensemble_f1, bl_f1) * 100
    if improvement > 0:
        print(f"\n  → 混合模型改善: +{improvement:.1f}% ✓")
    else:
        print(f"\n  → 混合模型改善: {improvement:.1f}%")

    print(f"\n輸出檔案:")
    print(f"  - output/hybrid_model.pkl                混合模型（可載入使用）")
    print(f"  - output/hybrid_predictions.csv          詳細預測結果")
    print(f"  - output/hybrid_weighted.png             加權策略圖表")
    print(f"  - output/hybrid_cascade.png              串聯策略圖表")
    print(f"  - output/hybrid_voting.png               投票策略圖表")
    print(f"  - output/hybrid_tiered.png               分級策略圖表")

    print("\n" + "="*80)
    print("  完成！")
    print("="*80 + "\n")

    return hybrid, results


def train_ensemble(X_train, y_train):
    """訓練 Ensemble 模型"""
    print("  訓練 Ensemble 模型中...")

    ensemble = StackingEnsemble(
        n_splits=5,
        smote_ratio=0.12,
        use_optuna=False,  # 使用預設參數（更快）
    )
    ensemble.fit(X_train, y_train, gnn_proba=None)

    # 儲存模型
    import pickle
    with open("output/ensemble_model.pkl", 'wb') as f:
        pickle.dump(ensemble, f)
    print("  ✓ Ensemble 模型訓練完成並已儲存")

    return ensemble


if __name__ == "__main__":
    try:
        hybrid, results = main()
    except FileNotFoundError as e:
        print(f"\n[錯誤] {e}")
        print("\n請先運行以下命令生成特徵數據：")
        print("  python main.py --data_dir ../adjust_data/train --output output")
    except Exception as e:
        print(f"\n[錯誤] {e}")
        import traceback
        traceback.print_exc()
