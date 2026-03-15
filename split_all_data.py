import pandas as pd
import os
import glob

# 建立輸出目錄
os.makedirs('adjust_data/train', exist_ok=True)
os.makedirs('adjust_data/predict', exist_ok=True)

print("=" * 60)
print("開始處理所有 CSV 檔案")
print("=" * 60)

# ── 1. 讀取標籤檔案 ──────────────────────────────
print("\n[1/6] 讀取標籤檔案...")
train_label = pd.read_csv('RawData/train_label.csv')
train_user_ids = set(train_label['user_id'].unique())
print(f"  訓練集 user_id 數量: {len(train_user_ids)}")

predict_label = pd.read_csv('RawData/predict_label.csv')
predict_user_ids = set(predict_label['user_id'].unique())
print(f"  預測集 user_id 數量: {len(predict_user_ids)}")

# 檢查 train / predict 是否有重疊
overlap = train_user_ids & predict_user_ids
if overlap:
    print(f"  ⚠ 警告: train 與 predict 有 {len(overlap)} 個重複 user_id")
else:
    print("  ✓ train 與 predict 的 user_id 無重疊")

# ── 2. 需要處理的檔案（含 trading） ───────────────
files_to_process = [
    'crypto_transfer.csv',
    'twd_transfer.csv',
    'usdt_swap.csv',
    'usdt_twd_trading.csv',
    'user_info.csv',
]

# ── 3. 逐檔處理 ──────────────────────────────────
for idx, filename in enumerate(files_to_process, start=2):
    print(f"\n[{idx}/6] 處理 {filename}...")

    file_path = f'RawData/{filename}'
    data = pd.read_csv(file_path)

    print(f"  總記錄數: {len(data):,}")
    print(f"  總用戶數: {data['user_id'].nunique():,}")

    # 特殊處理 user_info — merge status 標籤
    if filename == 'user_info.csv':
        print("  → 合併 train_label 的 status 欄...")
        data_train = data[data['user_id'].isin(train_user_ids)].copy()
        data_predict = data[data['user_id'].isin(predict_user_ids)].copy()

        data_train = data_train.merge(
            train_label[['user_id', 'status']],
            on='user_id',
            how='left',
        )
        print(f"  訓練集記錄: {len(data_train):,} (包含 status 欄)")
        print(f"  預測集記錄: {len(data_predict):,}")
    else:
        data_train = data[data['user_id'].isin(train_user_ids)]
        data_predict = data[data['user_id'].isin(predict_user_ids)]

        print(f"  訓練集記錄: {len(data_train):,}")
        print(f"  預測集記錄: {len(data_predict):,}")

    # 檢查遺漏的 user_id
    data_user_ids = set(data['user_id'].unique())
    missing_train = train_user_ids - data_user_ids
    missing_predict = predict_user_ids - data_user_ids
    if missing_train:
        print(f"  ℹ {len(missing_train)} 個訓練集 user_id 在 {filename} 中無記錄")
    if missing_predict:
        print(f"  ℹ {len(missing_predict)} 個預測集 user_id 在 {filename} 中無記錄")

    # 儲存到子目錄
    base_name = filename.replace('.csv', '')
    train_file = f'adjust_data/train/{base_name}_train.csv'
    predict_file = f'adjust_data/predict/{base_name}_predict.csv'

    data_train.to_csv(train_file, index=False)
    data_predict.to_csv(predict_file, index=False)

    print(f"  ✓ 已儲存 train/{base_name}_train.csv")
    print(f"  ✓ 已儲存 predict/{base_name}_predict.csv")

# ── 4. 摘要 ──────────────────────────────────────
print("\n" + "=" * 60)
print("所有檔案處理完成！")
print("=" * 60)

print("\n生成的檔案:")
for subdir in ['train', 'predict']:
    print(f"\n  adjust_data/{subdir}/")
    for f in sorted(glob.glob(f'adjust_data/{subdir}/*.csv')):
        size = os.path.getsize(f) / (1024 * 1024)
        print(f"    {os.path.basename(f)} ({size:.1f} MB)")
