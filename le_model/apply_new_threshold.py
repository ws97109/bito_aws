"""
应用新阈值重新评分用户
"""
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# 读取风险评分
scores_df = pd.read_csv('output/user_risk_scores.csv')

# 选择阈值（可修改）
THRESHOLD = 0.15  # 方案2：平衡方案

# 重新预测
y_true = scores_df['true_label'].values
y_score = scores_df['risk_score'].values
y_pred_new = (y_score >= THRESHOLD).astype(int)

# 更新预测标签和风险等级
scores_df['predicted_label_new'] = y_pred_new

def assign_risk_level(score):
    if score >= 0.40:
        return '极高风险'
    elif score >= 0.25:
        return '高风险'
    elif score >= 0.15:
        return '中风险'
    elif score >= 0.08:
        return '低风险'
    else:
        return '正常'

scores_df['risk_level_new'] = scores_df['risk_score'].apply(assign_risk_level)

# 保存更新后的结果
scores_df.to_csv('output/user_risk_scores_optimized.csv', index=False)

print("=" * 70)
print(f"应用新阈值: {THRESHOLD}")
print("=" * 70)

print("\n分类报告:")
print(classification_report(y_true, y_pred_new,
                          target_names=['正常', '黑名单'],
                          digits=4))

print("\n混淆矩阵:")
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_new).ravel()
print(f"  真阴性(TN): {tn:6,}  |  假阳性(FP): {fp:6,}")
print(f"  假阴性(FN): {fn:6,}  |  真阳性(TP): {tp:6,}")

print("\n风险等级分布:")
print(scores_df['risk_level_new'].value_counts().sort_index())

print("\n已保存优化后的评分: output/user_risk_scores_optimized.csv")
print("=" * 70)

# 列出新识别出的高风险用户（Top 10）
high_risk_users = scores_df[scores_df['predicted_label_new'] == 1].sort_values('risk_score', ascending=False).head(10)
print("\nTop 10 高风险用户:")
print(high_risk_users[['user_id', 'risk_score', 'true_label', 'risk_level_new']].to_string(index=False))
