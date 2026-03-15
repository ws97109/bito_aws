"""
獨立測試腳本：跑特徵篩選並輸出結果
"""
import sys
import json
sys.path.insert(0, "model")

import pandas as pd
import numpy as np
from Feature_rngineering import build_all_features
from feature_selection import select_features

# 載入資料
print("載入資料...")
user_info = pd.read_csv("adjust_data/train/user_info_train.csv", low_memory=False)
twd       = pd.read_csv("adjust_data/train/twd_transfer_train.csv", low_memory=False)
crypto    = pd.read_csv("adjust_data/train/crypto_transfer_train.csv", low_memory=False)
trading   = pd.read_csv("adjust_data/train/usdt_twd_trading_train.csv", low_memory=False)
swap      = pd.read_csv("adjust_data/train/usdt_swap_train.csv", low_memory=False)

# 特徵工程
print("特徵工程...")
feat = build_all_features(user_info, twd, crypto, trading, swap)
labels = user_info.set_index("user_id")["status"]
y = labels.reindex(feat.index).fillna(0).astype(int).values

print(f"特徵工程完成: {feat.shape[0]} 用戶, {feat.shape[1]} 特徵")

# 特徵篩選
print("\n" + "=" * 55)
print("開始特徵篩選")
print("=" * 55)

X_selected, report = select_features(feat, y, corr_threshold=0.95)

# 儲存結果
X_selected.to_csv("output/features_selected.csv")

with open("output/feature_selection_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"\n已儲存:")
print(f"  output/features_selected.csv          篩選後特徵矩陣")
print(f"  output/feature_selection_report.json   篩選報告")
