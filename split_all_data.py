import pandas as pd
import os

# 创建输出目录
os.makedirs('adjust_data', exist_ok=True)

print("=" * 60)
print("开始处理所有CSV文件")
print("=" * 60)

# 读取标签文件获取user_id
print("\n[1/5] 读取标签文件...")
train_label = pd.read_csv('RawData/train_label.csv')
train_user_ids = set(train_label['user_id'].unique())
print(f"  训练集 user_id 数量: {len(train_user_ids)}")

predict_label = pd.read_csv('RawData/predict_label.csv')
predict_user_ids = set(predict_label['user_id'].unique())
print(f"  预测集 user_id 数量: {len(predict_user_ids)}")

# 需要处理的文件列表（不包括已处理的usdt_twd_trading和标签文件）
files_to_process = [
    'crypto_transfer.csv',
    'twd_transfer.csv',
    'usdt_swap.csv',
    'user_info.csv'
]

# 处理每个文件
for idx, filename in enumerate(files_to_process, start=2):
    print(f"\n[{idx}/5] 处理 {filename}...")

    file_path = f'RawData/{filename}'
    data = pd.read_csv(file_path)

    print(f"  总记录数: {len(data):,}")
    print(f"  总用户数: {data['user_id'].nunique():,}")

    # 特殊处理 user_info - 需要merge status
    if filename == 'user_info.csv':
        print("  → 合并 train_label 的 status 列...")
        # 先分割
        data_train = data[data['user_id'].isin(train_user_ids)].copy()
        data_predict = data[data['user_id'].isin(predict_user_ids)].copy()

        # 为训练集merge status
        data_train = data_train.merge(
            train_label[['user_id', 'status']],
            on='user_id',
            how='left'
        )

        print(f"  训练集记录: {len(data_train):,} (包含status列)")
        print(f"  预测集记录: {len(data_predict):,}")
    else:
        # 其他文件直接分割
        data_train = data[data['user_id'].isin(train_user_ids)]
        data_predict = data[data['user_id'].isin(predict_user_ids)]

        print(f"  训练集记录: {len(data_train):,}")
        print(f"  预测集记录: {len(data_predict):,}")

    # 保存文件
    base_name = filename.replace('.csv', '')
    train_file = f'adjust_data/{base_name}_train.csv'
    predict_file = f'adjust_data/{base_name}_predict.csv'

    data_train.to_csv(train_file, index=False)
    data_predict.to_csv(predict_file, index=False)

    print(f"  ✓ 已保存 {base_name}_train.csv")
    print(f"  ✓ 已保存 {base_name}_predict.csv")

print("\n" + "=" * 60)
print("所有文件处理完成！")
print("=" * 60)

# 显示最终文件列表
print("\n生成的文件:")
import glob
adjust_files = sorted(glob.glob('adjust_data/*.csv'))
for f in adjust_files:
    size = os.path.getsize(f) / (1024 * 1024)  # MB
    print(f"  {f} ({size:.1f} MB)")
