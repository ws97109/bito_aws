"""
阈值分析脚本 - 找出最佳的 Precision-Recall 平衡点
"""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import json

# 读取风险评分
scores_df = pd.read_csv('output/user_risk_scores.csv')

# 使用测试集（假设后20%）
n_test = int(len(scores_df) * 0.2)
test_df = scores_df.iloc[-n_test:]

y_true = test_df['true_label'].values
y_score = test_df['risk_score'].values

print("=" * 70)
print("阈值分析 - 寻找最佳 Precision-Recall 平衡")
print("=" * 70)

# 计算不同阈值下的指标
thresholds_to_test = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
results = []

print(f"\n{'阈值':<8} {'Precision':<12} {'Recall':<10} {'F1-Score':<10} {'预测黑名单数':<12}")
print("-" * 70)

for threshold in thresholds_to_test:
    y_pred = (y_score >= threshold).astype(int)

    if y_pred.sum() == 0:  # 没有预测为黑名单
        precision = 0
        recall = 0
        f1 = 0
    else:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

    n_predicted_blacklist = y_pred.sum()

    results.append({
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_predicted': n_predicted_blacklist
    })

    print(f"{threshold:<8.2f} {precision:<12.4f} {recall:<10.4f} {f1:<10.4f} {n_predicted_blacklist:<12}")

# 找出最佳F1阈值
best_f1_idx = max(range(len(results)), key=lambda i: results[i]['f1'])
best_f1_threshold = results[best_f1_idx]

print("\n" + "=" * 70)
print("最佳 F1-Score 阈值:")
print(f"  阈值: {best_f1_threshold['threshold']:.2f}")
print(f"  Precision: {best_f1_threshold['precision']:.4f}")
print(f"  Recall: {best_f1_threshold['recall']:.4f}")
print(f"  F1-Score: {best_f1_threshold['f1']:.4f}")
print(f"  预测黑名单数: {best_f1_threshold['n_predicted']}")

# 找出最佳 Recall（至少20%）且保持合理 Precision（>30%）的阈值
print("\n推荐阈值（平衡 Recall 和 Precision）:")
for r in results:
    if r['recall'] >= 0.20 and r['precision'] >= 0.30:
        print(f"  阈值: {r['threshold']:.2f} | Precision: {r['precision']:.4f} | Recall: {r['recall']:.4f} | F1: {r['f1']:.4f}")

# 绘制 Precision-Recall 曲线
precision_curve, recall_curve, thresholds_curve = precision_recall_curve(y_true, y_score)

plt.figure(figsize=(12, 5))

# 子图1: Precision-Recall曲线
plt.subplot(1, 2, 1)
plt.plot(recall_curve, precision_curve, linewidth=2)
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14)
plt.grid(True, alpha=0.3)

# 标记当前使用的阈值
current_threshold = 0.26
current_pred = (y_score >= current_threshold).astype(int)
current_precision = precision_score(y_true, current_pred, zero_division=0)
current_recall = recall_score(y_true, current_pred, zero_division=0)
plt.plot(current_recall, current_precision, 'ro', markersize=10, label=f'Current (t={current_threshold})')
plt.legend()

# 子图2: 不同阈值下的指标
plt.subplot(1, 2, 2)
thresholds_plot = [r['threshold'] for r in results]
precision_plot = [r['precision'] for r in results]
recall_plot = [r['recall'] for r in results]
f1_plot = [r['f1'] for r in results]

plt.plot(thresholds_plot, precision_plot, 'o-', label='Precision', linewidth=2)
plt.plot(thresholds_plot, recall_plot, 's-', label='Recall', linewidth=2)
plt.plot(thresholds_plot, f1_plot, '^-', label='F1-Score', linewidth=2)
plt.xlabel('Threshold', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Metrics vs Threshold', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/threshold_analysis.png', dpi=150, bbox_inches='tight')
print("\n图表已保存: output/threshold_analysis.png")
print("=" * 70)
