"""
測試黑名單學習器
使用現有的特徵和標籤來訓練並評估
"""
import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 導入黑名單學習器
from blacklist_learner import BlacklistLearner, demo_blacklist_learner


def main():
    print("\n" + "="*70)
    print("  黑名單特徵學習器 - 完整測試")
    print("="*70)

    # 1. 載入已有的特徵數據
    print("\n[1] 載入特徵數據...")
    feat_df = pd.read_csv("output/features.csv", index_col=0)

    # 分離標籤和特徵
    if 'status' in feat_df.columns:
        y = feat_df['status'].values
        X = feat_df.drop(columns=['status']).values
        feature_names = feat_df.drop(columns=['status']).columns.tolist()
    else:
        print("  [錯誤] 找不到 'status' 欄位，請先運行 main.py 生成特徵")
        return

    print(f"  總用戶數：{len(X):,}")
    print(f"  特徵維度：{X.shape[1]}")
    print(f"  黑名單數：{y.sum():,} ({y.mean()*100:.2f}%)")
    print(f"  正常用戶：{(y==0).sum():,} ({(y==0).mean()*100:.2f}%)")

    # 2. 分割訓練/測試集
    print("\n[2] 分割訓練/測試集（8:2）...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  訓練集：{len(X_train):,}（黑名單 {y_train.sum()}）")
    print(f"  測試集：{len(X_test):,}（黑名單 {y_test.sum()}）")

    # 3. 運行黑名單學習器
    print("\n[3] 訓練與評估...")
    learner, result = demo_blacklist_learner(
        X_train, y_train,
        X_test, y_test,
        feature_names=feature_names,
        output_dir="output",
    )

    # 4. 額外分析：不同閾值的效果
    print("\n" + "="*70)
    print("  閾值敏感度分析")
    print("="*70)

    from sklearn.metrics import precision_score, recall_score, f1_score

    print("\n  閾值    Precision  Recall    F1      檢出率")
    print("  " + "-"*50)

    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred = (result['combined_score'] >= threshold).astype(int)
        pr = precision_score(y_test, y_pred, zero_division=0)
        rc = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        detect_rate = y_pred.sum() / len(y_pred)

        print(f"  {threshold:.1f}     {pr:.4f}     {rc:.4f}    {f1:.4f}  {detect_rate*100:.1f}%")

    # 5. 群組分析
    print("\n" + "="*70)
    print("  黑名單子群組特徵分析")
    print("="*70)

    # 取得訓練集中的黑名單
    blacklist_mask = (y_train == 1)
    X_bl = X_train[blacklist_mask]

    # 為每個群組計算特徵均值
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_bl_scaled = scaler.fit_transform(X_bl)

    print(f"\n  共發現 {learner.n_clusters} 個黑名單子群組：")

    for i in range(learner.n_clusters):
        cluster_mask = (learner.cluster_labels == i)
        cluster_size = cluster_mask.sum()
        cluster_features = X_bl_scaled[cluster_mask].mean(axis=0)

        # 找出該群組最顯著的特徵（絕對值最大）
        top_feature_idx = np.abs(cluster_features).argsort()[-5:][::-1]

        print(f"\n  群組 {i}：{cluster_size} 個黑名單 ({cluster_size/len(X_bl)*100:.1f}%)")
        print(f"    最顯著特徵：")
        for idx in top_feature_idx:
            feat_name = feature_names[idx] if idx < len(feature_names) else f"F{idx}"
            feat_val = cluster_features[idx]
            print(f"      {feat_name:<30}: {feat_val:>7.3f}")

    # 6. 儲存結果
    print("\n" + "="*70)
    print("  儲存結果")
    print("="*70)

    result_df = pd.DataFrame({
        'user_idx': np.arange(len(y_test)),
        'true_label': y_test,
        'similarity_score': result['combined_score'],
        'closest_cluster': result['closest_cluster'],
        'svm_score': result['svm_score'],
        'iso_score': result['iso_score'],
        'knn_similarity': result['knn_similarity'],
        'predicted_label': (result['combined_score'] >= 0.5).astype(int),
    })
    result_df = result_df.sort_values('similarity_score', ascending=False)
    result_df.to_csv("output/blacklist_learner_results.csv", index=False)
    print(f"  ✓ 詳細結果：output/blacklist_learner_results.csv")

    # 7. 總結
    print("\n" + "="*70)
    print("  總結")
    print("="*70)

    from sklearn.metrics import roc_auc_score, average_precision_score

    auc_roc = roc_auc_score(y_test, result['combined_score'])
    auc_pr = average_precision_score(y_test, result['combined_score'])

    y_pred_05 = (result['combined_score'] >= 0.5).astype(int)
    pr_05 = precision_score(y_test, y_pred_05, zero_division=0)
    rc_05 = recall_score(y_test, y_pred_05, zero_division=0)
    f1_05 = f1_score(y_test, y_pred_05, zero_division=0)

    print(f"\n  效能指標（閾值=0.5）：")
    print(f"    AUC-ROC       : {auc_roc:.4f}")
    print(f"    AUC-PR        : {auc_pr:.4f}")
    print(f"    Precision     : {pr_05:.4f}")
    print(f"    Recall        : {rc_05:.4f}")
    print(f"    F1            : {f1_05:.4f}")

    print(f"\n  優勢：")
    print(f"    ✓ 只需要學習黑名單樣本")
    print(f"    ✓ 可解釋性強（顯示與哪個黑名單群組相似）")
    print(f"    ✓ 可發現新型態黑名單（與已知黑名單相似）")
    print(f"    ✓ 提供多種視角的相似度評分")

    print(f"\n  輸出檔案：")
    print(f"    output/blacklist_space.png              - 黑名單特徵空間可視化")
    print(f"    output/blacklist_learner_results.csv    - 詳細預測結果")

    print("\n" + "="*70)
    print("  完成！")
    print("="*70 + "\n")

    return learner, result


if __name__ == "__main__":
    try:
        learner, result = main()
    except FileNotFoundError as e:
        print(f"\n[錯誤] {e}")
        print("\n請先運行以下命令生成特徵數據：")
        print("  python main.py --data_dir ../adjust_data/train --output output")
    except Exception as e:
        print(f"\n[錯誤] {e}")
        import traceback
        traceback.print_exc()
