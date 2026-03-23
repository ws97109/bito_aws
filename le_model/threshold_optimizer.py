"""
閾值優化與分析工具
用途：分析不同閾值對模型表現的影響，找出最佳業務平衡點
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    f1_score, precision_score, recall_score,
    confusion_matrix
)
import json
import os


def threshold_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fn: float = 10.0,    # 漏抓黑名單的業務成本
    cost_fp: float = 1.0,     # 誤抓正常用戶的業務成本
    save_dir: str = "output"
):
    """
    全面的閾值分析：
    1. Precision-Recall 曲線
    2. ROC 曲線
    3. 不同閾值下的指標表格
    4. 成本-效益分析
    5. 業務建議
    """
    os.makedirs(save_dir, exist_ok=True)

    # ═══ 1. 計算各閾值下的指標 ═══
    thresholds = np.arange(0.05, 0.95, 0.01)
    results = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        # 業務成本
        total_cost = cost_fn * fn + cost_fp * fp
        cost_per_user = total_cost / len(y_true)

        results.append({
            'threshold': t,
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'fpr': fpr,
            'fnr': fnr,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'total_cost': total_cost,
            'cost_per_user': cost_per_user,
        })

    df_results = pd.DataFrame(results)

    # ═══ 2. 找出關鍵閾值 ═══
    best_f1_idx = df_results['f1'].idxmax()
    best_cost_idx = df_results['cost_per_user'].idxmin()
    recall_70_df = df_results[df_results['recall'] >= 0.70]
    best_pr_at_70_idx = recall_70_df['precision'].idxmax() if len(recall_70_df) > 0 else best_f1_idx

    # 平衡點（Precision ≈ Recall）
    df_results['pr_diff'] = abs(df_results['precision'] - df_results['recall'])
    balanced_idx = df_results['pr_diff'].idxmin()

    recommendations = {
        'F1最優': {
            'threshold': df_results.loc[best_f1_idx, 'threshold'],
            'f1': df_results.loc[best_f1_idx, 'f1'],
            'recall': df_results.loc[best_f1_idx, 'recall'],
            'precision': df_results.loc[best_f1_idx, 'precision'],
            'description': '最大化 F1 分數，適合均衡場景'
        },
        '成本最優': {
            'threshold': df_results.loc[best_cost_idx, 'threshold'],
            'cost': df_results.loc[best_cost_idx, 'total_cost'],
            'recall': df_results.loc[best_cost_idx, 'recall'],
            'precision': df_results.loc[best_cost_idx, 'precision'],
            'description': f'最小化業務成本（漏抓成本={cost_fn}×誤抓成本）'
        },
        'Recall@70最佳精度': {
            'threshold': df_results.loc[best_pr_at_70_idx, 'threshold'],
            'recall': df_results.loc[best_pr_at_70_idx, 'recall'],
            'precision': df_results.loc[best_pr_at_70_idx, 'precision'],
            'description': '確保至少抓到 70% 黑名單的前提下最大化精度'
        },
        '平衡點': {
            'threshold': df_results.loc[balanced_idx, 'threshold'],
            'recall': df_results.loc[balanced_idx, 'recall'],
            'precision': df_results.loc[balanced_idx, 'precision'],
            'description': 'Precision ≈ Recall 的平衡點'
        }
    }

    # ═══ 3. 繪圖 ═══
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('閾值優化分析', fontsize=16, fontweight='bold')

    # 3.1 Recall vs Threshold
    ax = axes[0, 0]
    ax.plot(df_results['threshold'], df_results['recall'], 'b-', linewidth=2, label='Recall')
    ax.axhline(y=0.70, color='r', linestyle='--', alpha=0.5, label='Target=0.70')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Recall')
    ax.set_title('Recall vs Threshold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3.2 Precision vs Threshold
    ax = axes[0, 1]
    ax.plot(df_results['threshold'], df_results['precision'], 'g-', linewidth=2, label='Precision')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Precision')
    ax.set_title('Precision vs Threshold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3.3 F1 vs Threshold
    ax = axes[0, 2]
    ax.plot(df_results['threshold'], df_results['f1'], 'purple', linewidth=2, label='F1 Score')
    best_f1_t = df_results.loc[best_f1_idx, 'threshold']
    best_f1 = df_results.loc[best_f1_idx, 'f1']
    ax.axvline(x=best_f1_t, color='r', linestyle='--', alpha=0.5)
    ax.scatter([best_f1_t], [best_f1], color='red', s=100, zorder=5, label=f'Best={best_f1_t:.2f}')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score vs Threshold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3.4 Precision-Recall 曲線
    ax = axes[1, 0]
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
    ax.plot(recall_curve, precision_curve, 'b-', linewidth=2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve (AUC-PR={auc(recall_curve, precision_curve):.3f})')
    ax.grid(alpha=0.3)

    # 3.5 ROC 曲線
    ax = axes[1, 1]
    fpr_curve, tpr_curve, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr_curve, tpr_curve)
    ax.plot(fpr_curve, tpr_curve, 'b-', linewidth=2, label=f'AUC={roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate (Recall)')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3.6 成本分析
    ax = axes[1, 2]
    ax.plot(df_results['threshold'], df_results['cost_per_user'], 'r-', linewidth=2)
    best_cost_t = df_results.loc[best_cost_idx, 'threshold']
    best_cost = df_results.loc[best_cost_idx, 'cost_per_user']
    ax.axvline(x=best_cost_t, color='g', linestyle='--', alpha=0.5)
    ax.scatter([best_cost_t], [best_cost], color='green', s=100, zorder=5, label=f'Min={best_cost_t:.2f}')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Cost per User')
    ax.set_title(f'Business Cost Analysis (FN={cost_fn}×FP)')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'threshold_analysis_comprehensive.png'), dpi=150, bbox_inches='tight')
    print(f"\n  ✓ 圖表已儲存：{save_dir}/threshold_analysis_comprehensive.png")

    # ═══ 4. 輸出建議 ═══
    print("\n" + "="*70)
    print("  閾值優化建議")
    print("="*70)

    for strategy, info in recommendations.items():
        print(f"\n  【{strategy}】")
        for k, v in info.items():
            if k != 'description':
                print(f"    {k:<12}: {v:.4f}" if isinstance(v, float) else f"    {k:<12}: {v}")
        print(f"    說明: {info['description']}")

    # ═══ 5. 儲存詳細表格 ═══
    df_export = df_results[df_results['threshold'] % 0.05 < 0.01].copy()  # 每 0.05 取樣
    df_export.to_csv(os.path.join(save_dir, 'threshold_analysis_table.csv'), index=False)
    print(f"\n  ✓ 詳細表格已儲存：{save_dir}/threshold_analysis_table.csv")

    # ═══ 6. 儲存建議 JSON ═══
    with open(os.path.join(save_dir, 'threshold_recommendations.json'), 'w', encoding='utf-8') as f:
        json.dump(recommendations, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 建議已儲存：{save_dir}/threshold_recommendations.json")

    return df_results, recommendations


if __name__ == "__main__":
    # 測試用法
    print("載入測試集預測結果...")
    df = pd.read_csv("output/user_risk_scores.csv")

    # 提取測試集（假設前 20% 是測試集）
    n_test = int(len(df) * 0.2)
    df_test = df.head(n_test)

    y_true = df_test['true_label'].values
    y_proba = df_test['risk_score'].values

    # 執行分析
    results, recommendations = threshold_analysis(
        y_true, y_proba,
        cost_fn=10.0,   # 漏掉一個黑名單的代價是誤抓正常用戶的 10 倍
        cost_fp=1.0,
        save_dir="output"
    )

    print("\n" + "="*70)
    print("  分析完成！")
    print("="*70)
