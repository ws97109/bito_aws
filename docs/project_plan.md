# BitoGuard：智慧合規風險雷達 — 專案企劃書

> **競賽名稱**：BitoGroup × AWS 黑名單用戶偵測
> **目標**：打造 BitoGuard 智慧合規風險雷達，識別加密貨幣交易所中的黑名單（人頭戶）用戶
> **預測任務**：二元分類（0: 正常使用 / 1: 黑名單）
> **繳交格式**：predict_label CSV（user_id + status）

---

## 目錄

1. [評分標準](#1-評分標準)
2. [資料集概覽](#2-資料集概覽)
3. [Step 1：資料整理](#3-step-1資料整理)
4. [Step 2：特徵工程](#4-step-2特徵工程)
5. [Step 3：特徵篩選](#5-step-3特徵篩選)
6. [Step 4：不平衡處理](#6-step-4不平衡處理)
7. [Step 5：建模與訓練](#7-step-5建模與訓練)
8. [Step 6：SHAP 可解釋性分析](#8-step-6shap-可解釋性分析)
9. [Step 7：SSR 穩定性評測](#9-step-7ssr-穩定性評測)
10. [Step 8：最終繳交與簡報](#10-step-8最終繳交與簡報)

---

## 1. 評分標準

| 權重 | 項目 | 說明 |
|------|------|------|
| **40%** | 模型辨識效能 | Precision、Recall、FPR、**F1-score（主要依據）** |
| **30%** | 風險說明能力 | SHAP/LIME 可解釋性，解釋為何判定為高風險，且符合實務邏輯 |
| **15%** | 完整性與實務可用性 | 系統完整度、體驗流暢、支援風控決策流程 |
| **10%** | 主題切合及創意度 | 原創性，展現前所未有的構想 |
| **5%** | 加分項 | 視覺化關聯圖譜（GNN）、採用 AWS Kiro AI 整合開發環境 |

### 核心能力要求

1. **動態風險評分引擎** — 模型輸出風險分數或機率，依自訂閾值判斷是否為高風險
2. **AI 可解釋性報告** — 利用 SHAP 或 LIME，以自然語言或圖表呈現風險成因
3. **低誤判率** — 在確保捕捉率的同時，極大化降低對正常用戶的干擾
4. **模型驗證** — 訓練結果包含 Precision/Recall/F1-score 以及不平衡處理成效
5. **視覺化關聯圖譜（加分）** — 視覺化呈現資金流向與共犯結構，可用 GNN 技術

---

## 2. 資料集概覽

### 2.1 資料表一覽

| 表名 | 筆數 | 說明 |
|------|------|------|
| `user_info` | 63,770 | 用戶基本資料與 KYC 驗證資訊 |
| `twd_transfer` | 195,601 | 法幣（台幣）加值/提領交易 |
| `crypto_transfer` | 239,958 | 加密貨幣加值/提領/內轉 |
| `usdt_twd_trading` | 217,634 | 掛單簿 USDT/TWD 成交訂單 |
| `usdt_swap` | 53,841 | 一鍵買賣 USDT 成交訂單 |
| `train_label` | 51,017 | 訓練標籤（有 status） |
| `predict_label` | 12,753 | 預測目標（僅 user_id，需繳交預測結果） |

### 2.2 標籤分布（train_label）

| status | 筆數 | 佔比 |
|--------|------|------|
| 0（正常） | 49,377 | 96.8% |
| 1（黑名單） | 1,640 | 3.2% |

- 正負比約 **30:1**，嚴重不平衡
- train_label + predict_label = 63,770 = user_info 總數（完整覆蓋）

### 2.3 金額換算規則

> ⚠️ 所有金額、數量、匯率欄位皆以整數儲存，使用時需換算。

| 欄位類型 | 換算方式 | 範例 |
|----------|----------|------|
| 台幣金額（`ori_samount`） | `× 1e-8` | `998500000000 × 1e-8 = 9,985 TWD` |
| 虛擬貨幣數量（`ori_samount`, `trade_samount` 等） | `× 1e-8` | `814764500 × 1e-8 = 8.147645` |
| 匯率（`twd_srate`） | `× 1e-8` | `3054000000 × 1e-8 = 30.54 TWD/USDT` |

---

## 3. Step 1：資料整理

### 3.1 合併策略

以 `user_id` 為核心關聯鍵，將 4 張交易表聚合至用戶層級後合併：

```
user_info (63,770 users)
    ├── LEFT JOIN train_label   (51,017 users → 有標籤，用於訓練)
    ├── LEFT JOIN predict_label (12,753 users → 無標籤，用於預測)
    │
    ├── GROUP BY user_id ← twd_transfer     (195,601 txns)
    ├── GROUP BY user_id ← crypto_transfer  (239,958 txns)
    ├── GROUP BY user_id ← usdt_twd_trading (217,634 txns)
    └── GROUP BY user_id ← usdt_swap        (53,841 txns)
```

### 3.2 金額換算

所有金額欄位在聚合前先完成 `× 1e-8` 轉換：
- `ori_samount`、`trade_samount`、`twd_samount`、`currency_samount`、`twd_srate`

### 3.3 時間處理

- `user_info` 的 `confirmed_at`、`level1_finished_at`、`level2_finished_at` → 計算帳戶年齡、KYC 完成速度
- 交易表的時間欄位 → 計算交易頻率、時段分佈、首末交易間隔

### 3.4 缺失值處理

| 情境 | 處理方式 |
|------|----------|
| `level2_finished_at` 為空（未完成 KYC Level 2） | `has_kyc2 = 0`，KYC2 速度填 -1 或 NaN |
| `source_ip_hash` 為空（外部匯入） | 不納入 IP 計數，另設 `has_null_ip` 旗標 |
| `relation_user_id` 為空（非內轉） | 僅在 sub_kind=1 時使用 |
| 用戶在某張交易表完全無紀錄 | 該表所有聚合特徵填 0 |

---

## 4. Step 2：特徵工程

### 4.1 用戶基本特徵（from user_info）

| 特徵 | 說明 | 類型 |
|------|------|------|
| `sex` | 性別 | categorical |
| `age` | 年齡 | continuous |
| `career` | 職業類別（31 類） | categorical |
| `income_source` | 收入來源（10 類） | categorical |
| `user_source` | 註冊來源 WEB/APP | binary |
| `account_age_days` | 帳戶存在天數（confirmed_at → 資料截止日） | continuous |
| `kyc1_speed_hours` | Email 驗證 → KYC1 完成時間（小時） | continuous |
| `kyc2_speed_days` | KYC1 → KYC2 完成時間（天） | continuous |
| `has_kyc2` | 是否完成 KYC Level 2 | binary |

### 4.2 法幣行為特徵（from twd_transfer）

| 特徵 | 說明 | 人頭戶意義 |
|------|------|-----------|
| `twd_in_count` / `twd_out_count` | 入金/出金次數 | 頻繁進出 |
| `twd_in_sum` / `twd_out_sum` | 入金/出金總額（TWD） | 大額資金流動 |
| `twd_in_mean` / `twd_out_mean` | 平均單筆金額 | 交易規模 |
| `twd_net_flow` | 淨流入金額（入金 - 出金） | 資金方向 |
| `twd_out_in_ratio` | 出金/入金比 | 快進快出特徵 |
| `twd_ip_nunique` | 使用的不同 IP 數 | 多 IP 操作 |
| `twd_active_days` | 有交易的天數 | 活躍度 |
| `twd_max_single` | 單筆最大金額 | 異常大額交易 |

### 4.3 加密貨幣行為特徵（from crypto_transfer）

| 特徵 | 說明 | 人頭戶意義 |
|------|------|-----------|
| `crypto_in_count` / `crypto_out_count` | 入幣/出幣次數 | |
| `crypto_in_sum_twd` / `crypto_out_sum_twd` | 入幣/出幣換算台幣總額 | |
| `crypto_internal_count` | 內轉次數（sub_kind=1） | 帳戶間洗幣 |
| `crypto_external_count` | 外部轉帳次數（sub_kind=0） | 鏈上出入金 |
| `crypto_relation_user_nunique` | 內轉對象數量 | 關聯帳戶網絡規模 |
| `crypto_wallet_nunique` | 不同錢包地址數 | 分散資金 |
| `crypto_protocol_nunique` | 使用的不同區塊鏈數 | 跨鏈操作 |
| `crypto_currency_nunique` | 交易幣種數量 | 幣種多樣性 |

### 4.4 USDT 交易行為特徵（from usdt_twd_trading + usdt_swap）

| 特徵 | 說明 |
|------|------|
| `trading_buy_count` / `trading_sell_count` | 掛單買/賣次數 |
| `trading_buy_sum` / `trading_sell_sum` | 買/賣成交總額 |
| `trading_market_ratio` | 市價單佔比（急於成交的傾向） |
| `trading_source_nunique` | 使用終端數（WEB/APP/API） |
| `swap_buy_count` / `swap_sell_count` | 閃兌買/賣次數 |
| `swap_buy_sum` / `swap_sell_sum` | 閃兌買/賣總額 |

### 4.5 跨表衍生特徵

| 特徵 | 說明 | 人頭戶意義 |
|------|------|-----------|
| `total_tx_count` | 所有交易總次數 | 整體活躍度 |
| `total_volume_twd` | 所有交易換算台幣總額 | 整體資金規模 |
| `fiat_crypto_ratio` | 法幣交易額 / 加密貨幣交易額 | 資金轉換模式 |
| `avg_tx_per_active_day` | 日均交易頻率 | 集中操作強度 |
| `first_to_last_tx_days` | 首筆到末筆交易的天數間隔 | 短期集中操作（人頭戶常短期使用） |
| `night_tx_ratio` | 深夜交易佔比（00:00-06:00） | 異常交易時段 |
| `weekend_tx_ratio` | 週末交易佔比 | 異常交易時段 |
| `deposit_to_withdraw_speed` | 入金到出金的平均間隔（小時） | 快進快出核心指標 |

---

## 5. Step 3：特徵篩選

### 5.1 初步清理

- 移除**零方差特徵**（所有值相同的欄位）
- 移除**高缺失率特徵**（> 80% 缺失）
- 無交易紀錄的用戶，交易類特徵填 0（代表無行為）

### 5.2 相關性篩選

- 計算特徵間 Pearson 相關係數
- 若兩特徵相關性 > 0.95，保留與 target 相關性較高的那個
- 例如 `twd_in_sum` 和 `twd_in_mean` 可能高度相關

### 5.3 重要性篩選

- 跑一輪初步 **LightGBM**，取 `feature_importances_`
- 移除 importance = 0 的特徵
- 可搭配 SHAP summary plot 做二次確認

> **注意**：不使用 PCA。因競賽 30% 分數為風險說明能力，PCA 會將特徵轉為 PC1/PC2 等抽象主成分，喪失業務可解釋性（如「twd_out_in_ratio 異常」→「PC3 異常」），直接影響 SHAP 解釋品質。

### 5.4 預期結果

- 原始提取約 **50-60 個特徵**
- 篩選後預估保留 **30-40 個**

---

## 6. Step 4：不平衡處理

### 6.1 處理策略

| 方法 | 說明 | 優先級 |
|------|------|--------|
| **class_weight='balanced'** | 模型內建加權，不改變資料量 | 首選 |
| **SMOTE** | 對少數類生成合成樣本 | 備選 |
| **閾值調整** | 不改資料，調整分類閾值 | 必做 |
| Undersampling | 減少多數類 | 不建議（會丟失資訊） |

### 6.2 閾值選擇策略

官方 Live Demo 要求說明「選了什麼 threshold、為什麼選這個值」：

- 基於 **Precision-Recall Curve** 找最佳 F1 對應的閾值
- 也可依業務考量調整（寧可誤判也不漏抓 → 降低閾值提高 Recall）
- 最終報告需呈現不同閾值下的 Precision / Recall / F1 trade-off 圖表

---

## 7. Step 5：建模與訓練

### 7.1 模型選擇

| 模型 | 選用原因 | SHAP Explainer |
|------|----------|----------------|
| **XGBoost** | 表格資料 SOTA，處理不平衡佳 | TreeExplainer |
| **LightGBM** | 速度快，大數據友好 | TreeExplainer |
| **Random Forest** | 穩定基線模型 | TreeExplainer |
| Logistic Regression | 線性基線，可解釋性強 | LinearExplainer |

> 建議以 **XGBoost 或 LightGBM 為主模型**，RF 和 LR 作為對照基線。

### 7.2 訓練/驗證切分

```
train_label (51,017 users)
  ├── 70% Training set   (35,712)  → 模型訓練
  ├── 15% Validation set  (7,652)  → 調參、選閾值
  └── 15% Test set        (7,653)  → 最終評估

predict_label (12,753 users) → 最終預測繳交
```

- 使用 **Stratified Split** 保持各組正負比一致（均為 96.8% : 3.2%）
- 搭配 **5-Fold Stratified Cross-Validation** 做交叉驗證

### 7.3 評估指標

| 指標 | 說明 | 重要性 |
|------|------|--------|
| **F1-score** | Precision 與 Recall 的調和平均 | 官方主要依據 |
| Precision | 預測為黑名單中，真正是黑名單的比例 | 低誤判率 |
| Recall | 實際黑名單被成功識別的比例 | 捕捉率 |
| FPR | 正常用戶被誤判為黑名單的比例 | 官方強調低誤判 |

---

## 8. Step 6：SHAP 可解釋性分析

> 此步佔競賽評分 **30%**，為核心重點。

### 8.1 SHAP 計算

| 層級 | 方法 | 產出 |
|------|------|------|
| **Global（全局）** | `shap.TreeExplainer(model)` 對全部測試集計算 | 整體哪些特徵對黑名單判定最重要 |
| **Local（單一用戶）** | 取個別用戶的 SHAP 值 | 該用戶為什麼被判為高風險 |

### 8.2 視覺化產出

| 圖表 | 用途 | 對應需求 |
|------|------|----------|
| **Summary Plot（蜂群圖）** | 全局特徵重要性 + 方向性 | 風險因子總覽 |
| **Bar Plot** | 全局特徵重要性排序 | 簡報用 |
| **Waterfall Plot** | 單一用戶的風險貢獻拆解 | Live Demo：「這個用戶為什麼是黑名單」 |
| **Force Plot** | 單一用戶推力圖 | 同上，另一種直觀呈現 |
| **Dependence Plot** | 單一特徵值與 SHAP 值的關係 | 解釋特徵閾值效果 |

### 8.3 自然語言風險報告

官方範例要求能產生類似以下說明：

> 「該用戶與已知黑名單地址有 2 層內的關聯，且在深夜時段進行異常大額提現」

**實作方式**：

1. 取單一用戶 SHAP 值 Top-3 貢獻特徵
2. 依據特徵名稱 + SHAP 正負方向，用模板生成中文風險描述
3. 範例對應：

| 特徵 | SHAP 方向 | 生成描述 |
|------|-----------|----------|
| `twd_out_in_ratio` | 正（推向黑名單） | 「出金/入金比異常偏高，資金快速流出」 |
| `night_tx_ratio` | 正 | 「深夜交易佔比顯著高於一般用戶」 |
| `crypto_relation_user_nunique` | 正 | 「內轉對象帳戶數量異常，疑似資金分散」 |
| `account_age_days` | 負（帳齡短推向黑名單） | 「帳戶註冊時間極短，疑似新開人頭戶」 |

---

## 9. Step 7：SSR 穩定性評測

> 驗證 SHAP 解釋的可靠性與穩定性，強化可解釋性報告的可信度。

### 9.1 為什麼需要 SSR

SHAP 給出的解釋如果在資料微小波動下就完全改變，則該解釋不可靠。SSR（Stable Sample Ratio）回答的問題是：

> 「對同一個用戶，稍微擾動他的資料後，SHAP 給出的 Top 風險因子是否一致？」

### 9.2 SSR 公式

$$
SSR_{Top\text{-}k}(\varepsilon) = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\left[ \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}\left[ Top\text{-}k(x_i) = Top\text{-}k(x_i + \delta_t) \right] \geq \tau \right]
$$

- **N**：測試樣本數
- **T**：每個樣本的擾動次數
- **τ**：穩定閾值（≥ τ 比例一致才算穩定）
- **ε**：擾動強度
- **δₜ**：第 t 次擾動（依特徵類型使用不同策略）

### 9.3 計算流程

```
對每個測試用戶 xᵢ：
  1. 計算原始 SHAP 值 → 取 |SHAP| 排序的 Top-k 特徵
  2. 重複 T=10 次：
     a. 對 xᵢ 加入微小擾動 δₜ（依特徵類型選擇擾動方式）
     b. 計算擾動後 SHAP 值 → 取 Top-k 排序
     c. 比對擾動前後 Top-k 排序是否一致
  3. 若 ≥ 8/10 次排序一致 → 該用戶「穩定」

SSR = 穩定用戶數 / 總測試用戶數
```

### 9.4 Type-aware 擾動策略

| 特徵類型 | 本專案對應特徵範例 | 擾動方式 |
|---------|-------------------|----------|
| Continuous | age, twd_in_sum, account_age_days | 高斯噪音 N(0, ε·σⱼ) |
| Binary | has_kyc2, user_source | 以機率 ε 翻轉 0↔1 |
| Categorical | career, income_source | 以機率 ε 隨機替換為其他合法類別 |

### 9.5 實驗參數

| 參數 | 值 | 說明 |
|------|-----|------|
| 擾動級別 ε | 5%, 10%, 15%, 20% | 4 個級別觀察衰減趨勢 |
| 測試樣本數 N | 500 | 從測試集抽樣 |
| 每樣本擾動次數 T | 10 | |
| 穩定閾值 τ | 0.8 | 10 次中 ≥ 8 次一致才算穩定 |
| Top-k 值 | k = 1, 3, 5 | 分別評估首要、前三、前五風險因子穩定性 |

### 9.6 SSR 產出

| 產出 | 說明 |
|------|------|
| SSR_top1 / SSR_top3 / SSR_top5 | 各 k 值下的穩定樣本比例 |
| 擾動衰減曲線 | SSR 隨 ε 增大的下降趨勢圖 |
| 模型間比較 | 各模型（XGBoost/LightGBM/RF/LR）的 SSR 對比 |

> **簡報論述**：「我們的 SHAP 解釋不僅有業務意義，且經過穩定性驗證 — SSR_top1 達到 X%，代表 X% 的用戶在資料微小波動下，首要風險因子保持不變，解釋具備可信度。」

---

## 10. Step 8：最終繳交與簡報

### 10.1 繳交項目

| 項目 | 格式 | 內容 |
|------|------|------|
| **預測結果** | CSV | `user_id, status`（12,753 筆） |
| **簡報** | PPT/PDF | 完整方法論與結果展示 |
| **Live Demo** | 互動展示 | BitoGuard 系統演示 |

### 10.2 簡報架構

```
1. 資料概述 — 數據清洗流程與處理策略
2. 模型詳細設計
   ├── 特徵工程邏輯（人頭戶行為特徵設計理念）
   ├── 模型演算法選型與比較
   └── 可解釋性設計（SHAP + SSR）
3. Live Demo — BitoGuard 展示
   ├── 輸入用戶 → 輸出風險機率值
   ├── SHAP 解釋：該用戶為何被判為高風險
   ├── 閾值判斷說明（選了什麼 threshold、為什麼）
   └── 模型輸出的機率值分佈圖
```

### 10.3 預測結果格式

| user_id | status |
|---------|--------|
| 967903 | 0 |
| 204939 | 1 |
| 873334 | 0 |
| ... | ... |

- `0`：正常使用
- `1`：黑名單

---

## 附錄：專案目錄結構（規劃）

```
Bio_AWS_Workshop/
├── RawData/                          # 原始資料（7 張表）
│   ├── user_info.csv
│   ├── twd_transfer.csv
│   ├── crypto_transfer.csv
│   ├── usdt_twd_trading.csv
│   ├── usdt_swap.csv
│   ├── train_label.csv
│   └── predict_label.csv
│
├── Data/                             # 資料文件
│   └── data.md                       # 官方資料集說明
│
├── docs/                             # 專案文件
│   └── project_plan.md               # 本企劃書
│
├── notebooks/                        # Jupyter Notebooks
│   ├── 01_data_cleaning.ipynb        # Step 1：資料整理
│   ├── 02_feature_engineering.ipynb   # Step 2：特徵工程
│   ├── 03_feature_selection.ipynb     # Step 3：特徵篩選
│   ├── 04_modeling.ipynb             # Step 4-5：不平衡處理 + 建模
│   ├── 05_shap_analysis.ipynb        # Step 6：SHAP 可解釋性
│   └── 06_ssr_evaluation.ipynb       # Step 7：SSR 穩定性評測
│
├── src/                              # 核心程式碼模組
│   ├── data_processing.py            # 資料處理函數
│   ├── feature_engineering.py        # 特徵工程函數
│   ├── model_training.py             # 模型訓練函數
│   ├── shap_explainer.py             # SHAP 分析函數
│   ├── ssr_evaluator.py              # SSR 穩定性評測函數
│   └── risk_report.py               # 自然語言風險報告生成
│
├── models/                           # 訓練好的模型檔案
├── results/                          # 實驗結果與圖表
├── output/                           # 最終繳交檔案
│   └── predict_label.csv             # 預測結果
│
├── plot/                             # 競賽官方截圖
└── requirements.txt                  # Python 依賴套件
```
