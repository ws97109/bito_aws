"""
單獨測試特徵工程，確認產出是否正確
"""
import sys
import os
import pandas as pd
import numpy as np

# Wei_model/ → 專案根目錄
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_SCRIPT_DIR)

sys.path.insert(0, os.path.join(_SCRIPT_DIR, "model"))
from Feature_rngineering import build_all_features

DATA_DIR = os.path.join(ROOT, "adjust_data", "train")
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "output")

print("=" * 60)
print("特徵工程測試")
print("=" * 60)

# ── 載入資料 ──────────────────────────────────
print("\n[1] 載入資料...")
user_info = pd.read_csv(os.path.join(DATA_DIR, "user_info_train.csv"), low_memory=False)
twd       = pd.read_csv(os.path.join(DATA_DIR, "twd_transfer_train.csv"), low_memory=False)
crypto    = pd.read_csv(os.path.join(DATA_DIR, "crypto_transfer_train.csv"), low_memory=False)
trading   = pd.read_csv(os.path.join(DATA_DIR, "usdt_twd_trading_train.csv"), low_memory=False)
swap      = pd.read_csv(os.path.join(DATA_DIR, "usdt_swap_train.csv"), low_memory=False)

print(f"  user_info : {len(user_info):,} 筆")
print(f"  twd       : {len(twd):,} 筆")
print(f"  crypto    : {len(crypto):,} 筆")
print(f"  trading   : {len(trading):,} 筆")
print(f"  swap      : {len(swap):,} 筆")

# ── 執行特徵工程 ──────────────────────────────
print("\n[2] 執行特徵工程...")
feat_df = build_all_features(user_info, twd, crypto, trading, swap)

# ── 輸出結果 ──────────────────────────────────
print("\n[3] 特徵矩陣概覽")
print(f"  shape: {feat_df.shape}")
print(f"  用戶數: {len(feat_df):,}")
print(f"  特徵數: {feat_df.shape[1]}")

print(f"\n[4] 所有特徵欄位:")
for i, col in enumerate(feat_df.columns, 1):
    print(f"  {i:3d}. {col}")

print(f"\n[5] 基本統計:")
print(feat_df.describe().T.to_string())

print(f"\n[6] 缺值檢查:")
null_cols = feat_df.isnull().sum()
null_cols = null_cols[null_cols > 0]
if len(null_cols) == 0:
    print("  ✓ 無缺值")
else:
    for col, cnt in null_cols.items():
        print(f"  {col}: {cnt} ({cnt/len(feat_df)*100:.1f}%)")

print(f"\n[7] inf 值檢查:")
inf_cols = feat_df.isin([np.inf, -np.inf]).sum()
inf_cols = inf_cols[inf_cols > 0]
if len(inf_cols) == 0:
    print("  ✓ 無 inf 值")
else:
    for col, cnt in inf_cols.items():
        print(f"  {col}: {cnt}")

# 儲存
os.makedirs(OUTPUT_DIR, exist_ok=True)
feat_df.to_csv(os.path.join(OUTPUT_DIR, "features_test.csv"))
print(f"\n[8] 已儲存至 {OUTPUT_DIR}/features_test.csv")
print("=" * 60)
