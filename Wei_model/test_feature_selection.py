"""
獨立測試腳本：跑特徵篩選並輸出結果
"""
import sys
import os
import json

# Wei_model/ → 專案根目錄
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_SCRIPT_DIR)

sys.path.insert(0, os.path.join(_SCRIPT_DIR, "model"))

import pandas as pd
import numpy as np
from Feature_rngineering import build_all_features
from feature_selection import select_features

DATA_DIR = os.path.join(ROOT, "adjust_data", "train")
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "output")

# 載入資料
print("載入資料...")
user_info = pd.read_csv(os.path.join(DATA_DIR, "user_info_train.csv"), low_memory=False)
twd       = pd.read_csv(os.path.join(DATA_DIR, "twd_transfer_train.csv"), low_memory=False)
crypto    = pd.read_csv(os.path.join(DATA_DIR, "crypto_transfer_train.csv"), low_memory=False)
trading   = pd.read_csv(os.path.join(DATA_DIR, "usdt_twd_trading_train.csv"), low_memory=False)
swap      = pd.read_csv(os.path.join(DATA_DIR, "usdt_swap_train.csv"), low_memory=False)

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
os.makedirs(OUTPUT_DIR, exist_ok=True)
X_selected.to_csv(os.path.join(OUTPUT_DIR, "features_selected.csv"))

with open(os.path.join(OUTPUT_DIR, "feature_selection_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"\n已儲存:")
print(f"  {OUTPUT_DIR}/features_selected.csv          篩選後特徵矩陣")
print(f"  {OUTPUT_DIR}/feature_selection_report.json   篩選報告")
