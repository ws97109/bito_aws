import pandas as pd
import numpy as np
import os
from pathlib import Path

print("=" * 70)
print("开始处理CSV文件中的空白值")
print("=" * 70)

# 查找所有CSV文件
csv_files = list(Path('adjust_data').rglob('*.csv'))
print(f"\n找到 {len(csv_files)} 个CSV文件\n")

# 统计信息
total_null_filled = 0
processed_files = 0

for csv_file in csv_files:
    print(f"处理: {csv_file}")

    try:
        # 读取CSV，将空字符串视为NaN
        df = pd.read_csv(csv_file, keep_default_na=True, na_values=['', ' ', 'NA', 'N/A', 'null', 'NULL'])

        # 统计空值数量
        null_count_before = df.isnull().sum().sum()

        # 将所有只包含空格的字符串也替换为NaN
        for col in df.columns:
            if df[col].dtype == 'object':  # 只处理字符串列
                # 替换只包含空格的值
                df[col] = df[col].apply(lambda x: np.nan if isinstance(x, str) and x.strip() == '' else x)

        null_count_after = df.isnull().sum().sum()
        null_filled = null_count_after - null_count_before

        if null_filled > 0:
            print(f"  → 发现并处理了 {null_filled} 个空白值")
            total_null_filled += null_filled
        else:
            print(f"  ✓ 未发现空白值")

        # 保存回CSV，空值将被保存为空字符串（CSV标准格式）
        df.to_csv(csv_file, index=False)
        processed_files += 1

    except Exception as e:
        print(f"  ✗ 错误: {e}")
        continue

print("\n" + "=" * 70)
print(f"处理完成!")
print(f"  处理文件数: {processed_files}")
print(f"  总共填充空白值: {total_null_filled}")
print("=" * 70)

# 显示每个文件的空值统计
print("\n各文件空值统计:")
for csv_file in sorted(csv_files):
    try:
        df = pd.read_csv(csv_file)
        null_counts = df.isnull().sum()
        total_nulls = null_counts.sum()
        if total_nulls > 0:
            print(f"\n{csv_file.name}:")
            for col, count in null_counts.items():
                if count > 0:
                    print(f"  {col}: {count} 个空值")
    except:
        pass
