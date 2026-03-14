# BitoGuard：智慧合規風險雷達

> BitoGroup × AWS 黑名單用戶偵測競賽專案

## 專案目標

針對加密貨幣交易所用戶交易行為，建立機器學習模型識別黑名單（人頭戶）用戶，並透過 SHAP 可解釋性分析提供風險成因說明。

## 資料集

| 表名 | 筆數 | 說明 |
|------|------|------|
| `user_info` | 63,770 | 用戶基本資料與 KYC 驗證資訊 |
| `twd_transfer` | 195,601 | 法幣（台幣）加值/提領交易 |
| `crypto_transfer` | 239,958 | 加密貨幣加值/提領/內轉 |
| `usdt_twd_trading` | 217,634 | 掛單簿 USDT/TWD 成交訂單 |
| `usdt_swap` | 53,841 | 一鍵買賣 USDT 成交訂單 |
| `train_label` | 51,017 | 訓練標籤（0: 正常 / 1: 黑名單） |
| `predict_label` | 12,753 | 預測目標（需繳交預測結果） |

## Pipeline

```
Step 1: 資料整理 — 5 張表以 user_id 聚合合併、金額換算、缺失值處理
Step 2: 特徵工程 — 用戶基本特徵 + 法幣/加密貨幣/USDT 交易行為 + 跨表衍生特徵
Step 3: 特徵篩選 — 零方差移除、相關性篩選、LightGBM 重要性篩選
Step 4: 不平衡處理 — class_weight balanced + 閾值調整（PR Curve 最佳 F1）
Step 5: 建模與訓練 — XGBoost / LightGBM / Random Forest / Logistic Regression
Step 6: SHAP 可解釋性 — Global + Local 解釋、自然語言風險報告
Step 7: SSR 穩定性評測 — 驗證 SHAP 解釋在資料擾動下的穩定性
Step 8: 繳交預測結果 + 簡報 + Live Demo
```

## 專案結構

```
Bio_AWS_Workshop/
├── RawData/                # 原始資料（7 張表）
├── Data/                   # 資料集說明文件
├── docs/                   # 專案企劃書
├── notebooks/              # Jupyter Notebooks（各步驟）
├── src/                    # 核心程式碼模組
├── models/                 # 訓練好的模型
├── results/                # 實驗結果與圖表
└── output/                 # 最終繳交檔案
```

## 評分標準

| 權重 | 項目 |
|------|------|
| 40% | 模型辨識效能（F1-score 為主） |
| 30% | 風險說明能力（SHAP 可解釋性） |
| 15% | 完整性與實務可用性 |
| 10% | 主題切合及創意度 |
| 5%  | 加分項（視覺化關聯圖譜、AWS Kiro） |

## 詳細企劃

完整 pipeline 規劃請參閱 [docs/project_plan.md](docs/project_plan.md)
