"""
詳細評估黑名單學習器在測試集上的表現
展示：
1. 不同閾值下的混淆矩陣
2. 被正確識別的黑名單案例
3. 被漏掉的黑名單案例
4. 被誤判的正常用戶案例
5. ROC 曲線和 PR 曲線
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    precision_score, recall_score, f1_score,
)
import seaborn as sns

from blacklist_learner import BlacklistLearner


def detailed_evaluation(
    learner: BlacklistLearner,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_user_ids: np.ndarray,
    feature_names: list,
    threshold: float = 0.3,
):
    """詳細評估黑名單學習器"""

    print("\n" + "="*80)
    print("  黑名單學習器 - 詳細評估報告")
    print("="*80)

    # 1. 預測
    result = learner.predict_similarity(X_test)
    y_pred = (result['combined_score'] >= threshold).astype(int)

    # 2. 混淆矩陣
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n【測試集概況】")
    print(f"  總用戶數：{len(y_test):,}")
    print(f"  黑名單數：{y_test.sum():,} ({y_test.mean()*100:.2f}%)")
    print(f"  正常用戶：{(y_test==0).sum():,} ({(y_test==0).mean()*100:.2f}%)")

    print(f"\n【閾值設定】{threshold:.2f}")

    print(f"\n【混淆矩陣】")
    print(f"                預測為正常    預測為黑名單")
    print(f"  實際正常      {tn:>6}       {fp:>6}       (FPR={fp/(tn+fp)*100:.1f}%)")
    print(f"  實際黑名單    {fn:>6}       {tp:>6}       (Recall={tp/(tp+fn)*100:.1f}%)")

    print(f"\n【關鍵指標】")
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"  ✓ 真陽性 (TP)      : {tp:>6} - 正確抓到的黑名單")
    print(f"  ✗ 假陽性 (FP)      : {fp:>6} - 誤判為黑名單的正常用戶")
    print(f"  ✗ 假陰性 (FN)      : {fn:>6} - 漏掉的黑名單")
    print(f"  ✓ 真陰性 (TN)      : {tn:>6} - 正確識別的正常用戶")
    print(f"\n  Precision (精確率)  : {precision:.4f} ({precision*100:.1f}%)")
    print(f"  Recall (召回率)     : {recall:.4f} ({recall*100:.1f}%)")
    print(f"  F1 Score           : {f1:.4f}")
    print(f"  Specificity (特異度): {specificity:.4f} ({specificity*100:.1f}%)")

    # 3. 分析被正確識別的黑名單（TP）
    print(f"\n" + "="*80)
    print(f"  【成功案例】正確識別的黑名單 (TP={tp})")
    print("="*80)

    if tp > 0:
        tp_mask = (y_test == 1) & (y_pred == 1)
        tp_indices = np.where(tp_mask)[0]
        tp_scores = result['combined_score'][tp_mask]
        tp_sorted = tp_indices[np.argsort(tp_scores)[::-1]]  # 按分數排序

        print(f"\n  顯示前 5 個最高分的成功案例：")
        for i, idx in enumerate(tp_sorted[:5], 1):
            score = result['combined_score'][idx]
            cluster = result['closest_cluster'][idx]
            svm_s = result['svm_score'][idx]
            iso_s = result['iso_score'][idx]
            knn_s = result['knn_similarity'][idx]

            print(f"\n  案例 {i}: 用戶 ID={test_user_ids[idx]}")
            print(f"    ✓ 綜合分數：{score:.4f}")
            print(f"      - SVM 分數：{svm_s:.4f}")
            print(f"      - 異常分數：{iso_s:.4f}")
            print(f"      - KNN 相似度：{knn_s:.4f}")
            print(f"      - 最相似群組：群組 {cluster}")

    # 4. 分析被漏掉的黑名單（FN）
    print(f"\n" + "="*80)
    print(f"  【漏報案例】被漏掉的黑名單 (FN={fn})")
    print("="*80)

    if fn > 0:
        fn_mask = (y_test == 1) & (y_pred == 0)
        fn_indices = np.where(fn_mask)[0]
        fn_scores = result['combined_score'][fn_mask]
        fn_sorted = fn_indices[np.argsort(fn_scores)[::-1]]  # 按分數排序

        print(f"\n  顯示前 5 個最接近閾值的漏報案例（最可惜的）：")
        for i, idx in enumerate(fn_sorted[:5], 1):
            score = result['combined_score'][idx]
            cluster = result['closest_cluster'][idx]
            svm_s = result['svm_score'][idx]
            iso_s = result['iso_score'][idx]
            knn_s = result['knn_similarity'][idx]

            print(f"\n  案例 {i}: 用戶 ID={test_user_ids[idx]}")
            print(f"    ✗ 綜合分數：{score:.4f} (低於閾值 {threshold:.2f})")
            print(f"      - SVM 分數：{svm_s:.4f}")
            print(f"      - 異常分數：{iso_s:.4f}")
            print(f"      - KNN 相似度：{knn_s:.4f}")
            print(f"      - 最相似群組：群組 {cluster}")
            print(f"      原因：可能與已知黑名單模式差異較大")

    # 5. 分析被誤判的正常用戶（FP）
    print(f"\n" + "="*80)
    print(f"  【誤報案例】被誤判為黑名單的正常用戶 (FP={fp})")
    print("="*80)

    if fp > 0:
        fp_mask = (y_test == 0) & (y_pred == 1)
        fp_indices = np.where(fp_mask)[0]
        fp_scores = result['combined_score'][fp_mask]
        fp_sorted = fp_indices[np.argsort(fp_scores)[::-1]]  # 按分數排序

        print(f"\n  顯示前 5 個分數最高的誤報案例（最容易被誤判）：")
        for i, idx in enumerate(fp_sorted[:5], 1):
            score = result['combined_score'][idx]
            cluster = result['closest_cluster'][idx]
            svm_s = result['svm_score'][idx]
            iso_s = result['iso_score'][idx]
            knn_s = result['knn_similarity'][idx]

            print(f"\n  案例 {i}: 用戶 ID={test_user_ids[idx]}")
            print(f"    ✗ 綜合分數：{score:.4f} (超過閾值 {threshold:.2f})")
            print(f"      - SVM 分數：{svm_s:.4f}")
            print(f"      - 異常分數：{iso_s:.4f}")
            print(f"      - KNN 相似度：{knn_s:.4f}")
            print(f"      - 最相似群組：群組 {cluster}")
            print(f"      原因：行為模式與黑名單群組 {cluster} 相似")

    # 6. 閾值敏感度分析
    print(f"\n" + "="*80)
    print(f"  【閾值敏感度分析】不同閾值的效果對比")
    print("="*80)

    print(f"\n  閾值   TP    FP    FN    TN    Precision  Recall    F1      FPR")
    print("  " + "-"*75)

    for t in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        y_pred_t = (result['combined_score'] >= t).astype(int)
        cm_t = confusion_matrix(y_test, y_pred_t)
        tn_t, fp_t, fn_t, tp_t = cm_t.ravel()

        pr_t = precision_score(y_test, y_pred_t, zero_division=0)
        rc_t = recall_score(y_test, y_pred_t, zero_division=0)
        f1_t = f1_score(y_test, y_pred_t, zero_division=0)
        fpr_t = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0

        marker = " ◄ 當前" if abs(t - threshold) < 0.01 else ""
        print(f"  {t:.2f}  {tp_t:>4} {fp_t:>5} {fn_t:>5} {tn_t:>5}  {pr_t:>8.4f}  {rc_t:>8.4f}  {f1_t:>6.4f}  {fpr_t:>6.4f}{marker}")

    # 7. 繪製評估圖表
    plot_evaluation(y_test, result['combined_score'], threshold)

    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'specificity': specificity,
    }


def plot_evaluation(y_true, y_scores, threshold):
    """繪製評估圖表"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('黑名單學習器 - 測試集評估', fontsize=16, fontweight='bold')

    # 1. 分數分布
    ax = axes[0, 0]
    blacklist_scores = y_scores[y_true == 1]
    normal_scores = y_scores[y_true == 0]

    ax.hist(normal_scores, bins=50, alpha=0.6, label='正常用戶', color='blue', density=True)
    ax.hist(blacklist_scores, bins=50, alpha=0.6, label='黑名單', color='red', density=True)
    ax.axvline(threshold, color='green', linestyle='--', linewidth=2, label=f'閾值={threshold:.2f}')
    ax.set_xlabel('相似度分數')
    ax.set_ylabel('密度')
    ax.set_title('分數分布對比')
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. ROC 曲線
    ax = axes[0, 1]
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {roc_auc:.4f}')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='隨機猜測')
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR/Recall)')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. PR 曲線
    ax = axes[0, 2]
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    ax.plot(recall, precision, 'b-', linewidth=2, label=f'AUC-PR = {pr_auc:.4f}')
    baseline = y_true.sum() / len(y_true)
    ax.axhline(baseline, color='r', linestyle='--', linewidth=1, label=f'隨機猜測 = {baseline:.4f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. 混淆矩陣
    ax = axes[1, 0]
    y_pred = (y_scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['預測正常', '預測黑名單'],
                yticklabels=['實際正常', '實際黑名單'])
    ax.set_title(f'混淆矩陣 (閾值={threshold:.2f})')

    # 5. 閾值 vs 指標
    ax = axes[1, 1]
    thresholds = np.arange(0.05, 0.95, 0.01)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        y_pred_t = (y_scores >= t).astype(int)
        pr = precision_score(y_true, y_pred_t, zero_division=0)
        rc = recall_score(y_true, y_pred_t, zero_division=0)
        f1 = f1_score(y_true, y_pred_t, zero_division=0)
        precisions.append(pr)
        recalls.append(rc)
        f1s.append(f1)

    ax.plot(thresholds, precisions, 'g-', label='Precision', linewidth=2)
    ax.plot(thresholds, recalls, 'b-', label='Recall', linewidth=2)
    ax.plot(thresholds, f1s, 'r-', label='F1', linewidth=2)
    ax.axvline(threshold, color='black', linestyle='--', linewidth=1, label=f'當前閾值={threshold:.2f}')
    ax.set_xlabel('閾值')
    ax.set_ylabel('分數')
    ax.set_title('閾值 vs 指標')
    ax.legend()
    ax.grid(alpha=0.3)

    # 6. 累積分布
    ax = axes[1, 2]
    blacklist_sorted = np.sort(blacklist_scores)
    normal_sorted = np.sort(normal_scores)

    ax.plot(blacklist_sorted, np.arange(len(blacklist_sorted))/len(blacklist_sorted),
            'r-', linewidth=2, label='黑名單')
    ax.plot(normal_sorted, np.arange(len(normal_sorted))/len(normal_sorted),
            'b-', linewidth=2, label='正常用戶')
    ax.axvline(threshold, color='green', linestyle='--', linewidth=2, label=f'閾值={threshold:.2f}')
    ax.set_xlabel('相似度分數')
    ax.set_ylabel('累積比例')
    ax.set_title('累積分布函數 (CDF)')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/blacklist_learner_evaluation.png', dpi=150, bbox_inches='tight')
    print(f"\n  ✓ 評估圖表已儲存：output/blacklist_learner_evaluation.png")
    plt.close()


def main():
    print("\n" + "="*80)
    print("  使用原始測試集評估黑名單學習器")
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
    user_ids = feat_df.index.values

    print(f"  總用戶數：{len(X):,}")
    print(f"  黑名單數：{y.sum():,} ({y.mean()*100:.2f}%)")

    # 2. 分割訓練/測試集（使用相同的 random_state）
    print("\n[2] 分割訓練/測試集（8:2）...")
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)), test_size=0.2, stratify=y, random_state=42
    )
    test_user_ids = user_ids[idx_test]

    print(f"  訓練集：{len(X_train):,}（黑名單 {y_train.sum()}）")
    print(f"  測試集：{len(X_test):,}（黑名單 {y_test.sum()}）")

    # 3. 訓練黑名單學習器
    print("\n[3] 訓練黑名單學習器...")
    learner = BlacklistLearner(n_clusters=5, contamination=0.05, n_neighbors=10)
    learner.fit(X_train, y_train, feature_names)

    # 4. 測試不同閾值
    print("\n[4] 測試不同閾值...")

    best_threshold = 0.3
    best_f1 = 0.0

    result = learner.predict_similarity(X_test)

    for t in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        y_pred = (result['combined_score'] >= t).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

        print(f"  閾值 {t:.2f}: F1={f1:.4f}, Recall={recall:.4f}")

    print(f"\n  → 選擇最佳閾值：{best_threshold:.2f} (F1={best_f1:.4f})")

    # 5. 詳細評估
    metrics = detailed_evaluation(
        learner, X_test, y_test, test_user_ids,
        feature_names, threshold=best_threshold
    )

    # 6. 儲存詳細結果
    print(f"\n[5] 儲存詳細結果...")
    result_df = pd.DataFrame({
        'user_id': test_user_ids,
        'true_label': y_test,
        'similarity_score': result['combined_score'],
        'predicted_label': (result['combined_score'] >= best_threshold).astype(int),
        'closest_cluster': result['closest_cluster'],
        'svm_score': result['svm_score'],
        'iso_score': result['iso_score'],
        'knn_similarity': result['knn_similarity'],
        'result': ['TP' if (y_test[i]==1 and result['combined_score'][i]>=best_threshold)
                   else 'FP' if (y_test[i]==0 and result['combined_score'][i]>=best_threshold)
                   else 'FN' if (y_test[i]==1 and result['combined_score'][i]<best_threshold)
                   else 'TN' for i in range(len(y_test))]
    })
    result_df = result_df.sort_values('similarity_score', ascending=False)
    result_df.to_csv("output/blacklist_learner_detailed_results.csv", index=False)
    print(f"  ✓ 詳細結果已儲存：output/blacklist_learner_detailed_results.csv")

    # 7. 總結
    print(f"\n" + "="*80)
    print(f"  【最終總結】")
    print("="*80)
    print(f"\n  測試集規模：{len(y_test):,} 用戶（黑名單 {y_test.sum()}, 正常 {(y_test==0).sum()}）")
    print(f"  最佳閾值：{best_threshold:.2f}")
    print(f"\n  效能指標：")
    print(f"    ✓ 真陽性 (TP)  : {metrics['tp']:>4} - 成功抓到的黑名單")
    print(f"    ✗ 假陽性 (FP)  : {metrics['fp']:>4} - 誤判的正常用戶")
    print(f"    ✗ 假陰性 (FN)  : {metrics['fn']:>4} - 漏掉的黑名單")
    print(f"    ✓ 真陰性 (TN)  : {metrics['tn']:>4} - 正確識別的正常用戶")
    print(f"\n    Precision      : {metrics['precision']:.4f} ({metrics['precision']*100:.1f}%)")
    print(f"    Recall         : {metrics['recall']:.4f} ({metrics['recall']*100:.1f}%)")
    print(f"    F1 Score       : {metrics['f1']:.4f}")
    print(f"    Specificity    : {metrics['specificity']:.4f} ({metrics['specificity']*100:.1f}%)")

    print(f"\n  解讀：")
    if metrics['recall'] > 0.7:
        print(f"    ✓ 召回率優秀 ({metrics['recall']*100:.1f}%) - 能抓到大多數黑名單")
    elif metrics['recall'] > 0.5:
        print(f"    ⚠ 召回率中等 ({metrics['recall']*100:.1f}%) - 漏掉了一些黑名單")
    else:
        print(f"    ✗ 召回率偏低 ({metrics['recall']*100:.1f}%) - 漏掉太多黑名單")

    if metrics['precision'] > 0.5:
        print(f"    ✓ 精確率優秀 ({metrics['precision']*100:.1f}%) - 誤報率低")
    elif metrics['precision'] > 0.3:
        print(f"    ⚠ 精確率中等 ({metrics['precision']*100:.1f}%) - 有一定誤報")
    else:
        print(f"    ✗ 精確率偏低 ({metrics['precision']*100:.1f}%) - 誤報較多")

    print(f"\n  輸出檔案：")
    print(f"    - output/blacklist_learner_evaluation.png")
    print(f"    - output/blacklist_learner_detailed_results.csv")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
