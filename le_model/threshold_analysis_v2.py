"""
阈值分析脚本v2 - 针对低分数分布优化
"""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False

# 读取风险评分
scores_df = pd.read_csv('output/user_risk_scores.csv')

y_true = scores_df['true_label'].values
y_score = scores_df['risk_score'].values

print("=" * 80)
print("阈值分析 - 寻找最佳 Precision-Recall 平衡")
print("=" * 80)
print(f"\n数据概况:")
print(f"  总样本数: {len(y_true):,}")
print(f"  黑名单数: {y_true.sum():,} ({y_true.mean()*100:.2f}%)")
print(f"  风险分数中位数: {np.median(y_score):.6f}")
print(f"  风险分数75%分位: {np.percentile(y_score, 75):.6f}")
print(f"  风险分数90%分位: {np.percentile(y_score, 90):.6f}")
print(f"  风险分数95%分位: {np.percentile(y_score, 95):.6f}")
print(f"  风险分数99%分位: {np.percentile(y_score, 99):.6f}")

# 使用更合理的阈值范围
thresholds_to_test = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
                      0.12, 0.15, 0.20, 0.25, 0.30]
results = []

print(f"\n{'阈值':<8} {'Precision':<12} {'Recall':<10} {'F1':<10} {'预测黑名单':<12} {'TP':<6} {'FP':<6} {'FN':<6}")
print("-" * 80)

for threshold in thresholds_to_test:
    y_pred = (y_score >= threshold).astype(int)

    if y_pred.sum() == 0:
        precision = 0
        recall = 0
        f1 = 0
        tp = fp = fn = 0
    else:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # 混淆矩阵
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    n_predicted = y_pred.sum()

    results.append({
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_predicted': n_predicted,
        'tp': tp if 'tp' in locals() else 0,
        'fp': fp if 'fp' in locals() else 0,
        'fn': fn if 'fn' in locals() else 0
    })

    print(f"{threshold:<8.2f} {precision:<12.4f} {recall:<10.4f} {f1:<10.4f} {n_predicted:<12} {tp:<6} {fp:<6} {fn:<6}")

# 找出不同目标下的最佳阈值
print("\n" + "=" * 80)
print("推荐阈值方案:")
print("=" * 80)

# 1. 最佳F1
best_f1_idx = max(range(len(results)), key=lambda i: results[i]['f1'])
best_f1 = results[best_f1_idx]
print(f"\n【方案1: 最佳 F1-Score】")
print(f"  阈值: {best_f1['threshold']:.3f}")
print(f"  Precision: {best_f1['precision']:.4f} | Recall: {best_f1['recall']:.4f} | F1: {best_f1['f1']:.4f}")
print(f"  预测黑名单: {best_f1['n_predicted']} | TP: {best_f1['tp']} | FP: {best_f1['fp']} | FN: {best_f1['fn']}")

# 2. 高召回率（>50%）+ 可接受精确率（>20%）
print(f"\n【方案2: 平衡方案（Recall≥50%, Precision≥20%）】")
balanced = [r for r in results if r['recall'] >= 0.50 and r['precision'] >= 0.20]
if balanced:
    best_balanced = max(balanced, key=lambda x: x['f1'])
    print(f"  阈值: {best_balanced['threshold']:.3f}")
    print(f"  Precision: {best_balanced['precision']:.4f} | Recall: {best_balanced['recall']:.4f} | F1: {best_balanced['f1']:.4f}")
    print(f"  预测黑名单: {best_balanced['n_predicted']} | TP: {best_balanced['tp']} | FP: {best_balanced['fp']} | FN: {best_balanced['fn']}")
else:
    print("  未找到满足条件的阈值")

# 3. 高召回率优先（>70%）
print(f"\n【方案3: 高召回率优先（Recall≥70%）】")
high_recall = [r for r in results if r['recall'] >= 0.70]
if high_recall:
    best_recall = max(high_recall, key=lambda x: x['precision'])
    print(f"  阈值: {best_recall['threshold']:.3f}")
    print(f"  Precision: {best_recall['precision']:.4f} | Recall: {best_recall['recall']:.4f} | F1: {best_recall['f1']:.4f}")
    print(f"  预测黑名单: {best_recall['n_predicted']} | TP: {best_recall['tp']} | FP: {best_recall['fp']} | FN: {best_recall['fn']}")
else:
    print("  未找到满足条件的阈值")

# 绘制分析图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 子图1: Precision-Recall曲线
precision_curve, recall_curve, thresholds_curve = precision_recall_curve(y_true, y_score)
axes[0, 0].plot(recall_curve, precision_curve, linewidth=2, color='blue')
axes[0, 0].set_xlabel('Recall', fontsize=12)
axes[0, 0].set_ylabel('Precision', fontsize=12)
axes[0, 0].set_title('Precision-Recall Curve', fontsize=14)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(y=0.2, color='r', linestyle='--', alpha=0.5, label='20% Precision')
axes[0, 0].axvline(x=0.5, color='g', linestyle='--', alpha=0.5, label='50% Recall')
axes[0, 0].legend()

# 子图2: 不同阈值下的指标
thresholds_plot = [r['threshold'] for r in results]
precision_plot = [r['precision'] for r in results]
recall_plot = [r['recall'] for r in results]
f1_plot = [r['f1'] for r in results]

axes[0, 1].plot(thresholds_plot, precision_plot, 'o-', label='Precision', linewidth=2)
axes[0, 1].plot(thresholds_plot, recall_plot, 's-', label='Recall', linewidth=2)
axes[0, 1].plot(thresholds_plot, f1_plot, '^-', label='F1-Score', linewidth=2)
axes[0, 1].axvline(x=best_f1['threshold'], color='red', linestyle='--', alpha=0.5, label=f"Best F1 ({best_f1['threshold']:.3f})")
axes[0, 1].set_xlabel('Threshold', fontsize=12)
axes[0, 1].set_ylabel('Score', fontsize=12)
axes[0, 1].set_title('Metrics vs Threshold', fontsize=14)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 子图3: 预测黑名单数量
n_predicted_plot = [r['n_predicted'] for r in results]
axes[1, 0].plot(thresholds_plot, n_predicted_plot, 'o-', linewidth=2, color='purple')
axes[1, 0].axhline(y=y_true.sum(), color='r', linestyle='--', label=f'真实黑名单数 ({y_true.sum()})', alpha=0.7)
axes[1, 0].set_xlabel('Threshold', fontsize=12)
axes[1, 0].set_ylabel('预测黑名单数', fontsize=12)
axes[1, 0].set_title('Predicted Blacklist Count vs Threshold', fontsize=14)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 子图4: TP, FP, FN分布
tp_plot = [r['tp'] for r in results]
fp_plot = [r['fp'] for r in results]
fn_plot = [r['fn'] for r in results]

axes[1, 1].plot(thresholds_plot, tp_plot, 'o-', label='True Positive (TP)', linewidth=2, color='green')
axes[1, 1].plot(thresholds_plot, fp_plot, 's-', label='False Positive (FP)', linewidth=2, color='red')
axes[1, 1].plot(thresholds_plot, fn_plot, '^-', label='False Negative (FN)', linewidth=2, color='orange')
axes[1, 1].set_xlabel('Threshold', fontsize=12)
axes[1, 1].set_ylabel('Count', fontsize=12)
axes[1, 1].set_title('TP/FP/FN vs Threshold', fontsize=14)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/threshold_analysis_detailed.png', dpi=150, bbox_inches='tight')
print("\n图表已保存: output/threshold_analysis_detailed.png")
print("=" * 80)
