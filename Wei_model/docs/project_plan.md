# BitoGuard：智慧合規風險雷達 — 專案企劃書

> **競賽名稱**：BitoGroup × AWS 黑名單用戶偵測
> **目標**：打造 BitoGuard 智慧合規風險雷達，識別加密貨幣交易所中的黑名單（人頭戶）用戶
> **預測任務**：二元分類（0: 正常使用 / 1: 黑名單）
> **繳交格式**：predict_label CSV（user_id + status）

---

## 目錄

1. [評分標準](#1-評分標準)
2. [資料集概覽](#2-資料集概覽)
3. [系統架構總覽](#3-系統架構總覽)
4. [Step 1：資料整理](#4-step-1資料整理)
5. [Step 2：特徵工程（監督式特徵）](#5-step-2特徵工程監督式特徵)
6. [Step 2.5：非監督式異常偵測](#6-step-25非監督式異常偵測)
7. [Step 2.6：GNN 圖學習特徵提取](#7-step-26gnn-圖學習特徵提取)
8. [Step 3：特徵篩選與整合](#8-step-3特徵篩選與整合)
9. [Step 4：不平衡處理](#9-step-4不平衡處理)
10. [Step 5：監督式 Ensemble 建模](#10-step-5監督式-ensemble-建模)
11. [Step 5.5：半監督學習 — Pseudo-labeling](#11-step-55半監督學習--pseudo-labeling)
12. [Step 6：SHAP 可解釋性分析](#12-step-6shap-可解釋性分析)
13. [Step 7：SSR 穩定性評測](#13-step-7ssr-穩定性評測)
14. [Step 8：消融實驗與模型驗證](#14-step-8消融實驗與模型驗證)
15. [Step 9：最終繳交與簡報](#15-step-9最終繳交與簡報)

---

## 1. 評分標準

| 權重 | 項目 | 說明 |
|------|------|------|
| **40%** | 模型辨識效能 | 以混淆矩陣為基礎，綜合評估 Precision、Recall、FPR，**以 F1-score 作為主要比較依據** |
| **30%** | 風險說明能力 | SHAP/LIME 可解釋性，解釋為何判定為高風險，且符合實務邏輯 |
| **15%** | 完整性與實務可用性 | 系統完整度、體驗流暢、支援風控決策流程 |
| **10%** | 主題切合及創意度 | 原創性，展現前所未有的構想 |
| **5%** | 加分項 | 視覺化關聯圖譜（GNN）、採用 AWS Kiro AI 整合開發環境 |

### 官方評分機制

- **官方持有 predict_label（12,753 筆）的真實答案**，會直接用繳交的 CSV 對答案計算指標
- 官方未明確說明 F1 是 `F1(class=1)` 或 `macro-F1`，但根據官方用詞（「捕捉率」、「低誤判率」、「識別黑名單」）及 AML 業界慣例，**推定為 F1(class=1)**
- 繳交的是硬標籤（0/1），不是機率值 → **閾值選擇直接決定最終分數**
- 簡報中建議同時呈現 `F1(class=1)` 和 `macro-F1`，確保不論官方採用哪個都有數字可展示

### 核心能力要求

1. **動態風險評分引擎** — 模型輸出風險分數或機率，依自訂閾值判斷是否為高風險
2. **AI 可解釋性報告** — 利用 SHAP 或 LIME，以自然語言或圖表呈現風險成因
3. **低誤判率** — 在確保捕捉率的同時，極大化降低對正常用戶的干擾
4. **模型驗證** — 訓練結果包含 Precision/Recall/F1-score 以及不平衡處理成效
5. **監督式 + 非監督式結合** — 官方明確要求「結合監督式學習（已知模式）與非監督式學習（新型態/異常偵測）」
6. **視覺化關聯圖譜（加分）** — 視覺化呈現資金流向與共犯結構，可用 GNN 技術

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

> 所有金額、數量、匯率欄位皆以整數儲存，使用時需換算。

| 欄位類型 | 換算方式 | 範例 |
|----------|----------|------|
| 台幣金額（`ori_samount`） | `× 1e-8` | `998500000000 × 1e-8 = 9,985 TWD` |
| 虛擬貨幣數量（`ori_samount`, `trade_samount` 等） | `× 1e-8` | `814764500 × 1e-8 = 8.147645` |
| 匯率（`twd_srate`） | `× 1e-8` | `3054000000 × 1e-8 = 30.54 TWD/USDT` |

---

## 3. 系統架構總覽

### 3.1 三層偵測哲學

> **「人頭戶有三種被發現的方式，BitoGuard 全部覆蓋。」**

```
┌─────────────────────────────────────────────────────────────┐
│                   BitoGuard 三層偵測架構                      │
├─────────────────┬──────────────────┬────────────────────────┤
│   第一層          │    第二層          │      第三層            │
│   非監督式學習     │    圖學習 (GNN)    │     監督式學習         │
│  「誰的行為異常」  │ 「誰跟壞人有關聯」  │  「誰是黑名單」        │
├─────────────────┼──────────────────┼────────────────────────┤
│ 不看 label       │ 看關係網絡         │  完全依賴 label        │
│ 看資料分佈        │ 圖訊息傳播         │  看行為特徵            │
│ 捕捉新型態犯罪    │ 捕捉共犯結構       │  匹配已知黑名單模式     │
├─────────────────┼──────────────────┼────────────────────────┤
│ Isolation Forest │ GraphSAGE + GAT  │ XGBoost + LightGBM    │
│ HBOS、LOF       │ 異質圖神經網絡      │ + CatBoost Stacking   │
└────────┬────────┴────────┬─────────┴──────────┬─────────────┘
         │                 │                    │
    異常分數 (3)       圖嵌入向量 (8~16)    行為特徵 (47)
         │                 │                    │
         └─────────────────┼────────────────────┘
                           │
                    特徵整合 (~66 維)
                           │
                 監督式 Ensemble 最終判定
                           │
                  PR-curve 閾值最佳化
                           │
                      0 or 1 預測
```

### 3.2 三層偵測的互補性

| 層級 | 偵測邏輯 | 能抓到什麼 | 不能抓到什麼 |
|------|---------|-----------|------------|
| **非監督** | 個體行為在統計上的罕見程度 | 行為離群的用戶（新型態犯罪） | 行為正常但有共犯關係的人 |
| **GNN** | 交易網絡中的位置和角色 | 與可疑帳戶有資金往來的人 | 孤立節點（無交易紀錄的用戶） |
| **監督式** | 與已知黑名單的模式匹配 | 跟歷史黑名單行為相似的人 | 前所未見的新犯罪模式 |

> **三者聯集 > 任何單一方法**。非監督和 GNN 不直接做預測，而是作為「特徵提取器」，把不同維度的風險資訊轉化為數字，交由監督式 ensemble 做最終判定。

### 3.3 資訊流架構

```
原始資料 (7 張表)
    │
    ├─→ Step 1：資料整理（合併、換算、缺失值）
    │
    ├─→ Step 2：監督式特徵工程 ─────────→ 行為特徵 (47~50)
    │
    ├─→ Step 2.5：非監督異常偵測 ────────→ 異常分數 (3)
    │       Isolation Forest / HBOS / LOF
    │
    ├─→ Step 2.6：GNN 圖學習 ───────────→ 圖嵌入 (8~16)
    │       GraphSAGE + GAT on 交易圖
    │
    ├─→ Step 3：特徵篩選與整合 ──────────→ 最終特徵 (~66)
    │
    ├─→ Step 4：不平衡處理
    │       Focal Loss / Borderline-SMOTE / 閾值最佳化
    │
    ├─→ Step 5：監督式 Ensemble ─────────→ 機率預測
    │       XGBoost + LightGBM + CatBoost → Meta-Learner
    │
    ├─→ Step 5.5：半監督 Pseudo-labeling（可選）
    │       高信心 predict_label 回灌訓練
    │
    ├─→ Step 6：SHAP 可解釋性 ──────────→ 風險因子報告
    │
    ├─→ Step 7：SSR 穩定性評測 ─────────→ 解釋可信度驗證
    │
    ├─→ Step 8：消融實驗 ──────────────→ 各模組貢獻量化
    │
    └─→ Step 9：最終繳交 ──────────────→ CSV + 簡報 + Demo
```

---

## 4. Step 1：資料整理

### 4.1 合併策略

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

### 4.2 金額換算

所有金額欄位在聚合前先完成 `× 1e-8` 轉換：
- `ori_samount`、`trade_samount`、`twd_samount`、`currency_samount`、`twd_srate`

### 4.3 時間處理

- `user_info` 的 `confirmed_at`、`level1_finished_at`、`level2_finished_at` → 計算帳戶年齡、KYC 完成速度
- 交易表的時間欄位 → 計算交易頻率、時段分佈、首末交易間隔

### 4.4 缺失值處理

| 情境 | 處理方式 |
|------|----------|
| `level2_finished_at` 為空（未完成 KYC Level 2） | `has_kyc2 = 0`，KYC2 速度填 -1 或 NaN |
| `source_ip_hash` 為空（外部匯入） | 不納入 IP 計數，另設 `has_null_ip` 旗標 |
| `relation_user_id` 為空（非內轉） | 僅在 sub_kind=1 時使用 |
| 用戶在某張交易表完全無紀錄 | 該表所有聚合特徵填 0 |

---

## 5. Step 2：特徵工程（監督式特徵）

> 此步驟提取的是**描述用戶行為**的表格特徵，後續將與非監督異常分數和 GNN 圖嵌入合併。

### 5.1 用戶基本特徵（from user_info）

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

### 5.2 法幣行為特徵（from twd_transfer）

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

### 5.3 加密貨幣行為特徵（from crypto_transfer）

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

### 5.4 USDT 交易行為特徵（from usdt_twd_trading + usdt_swap）

| 特徵 | 說明 |
|------|------|
| `trading_buy_count` / `trading_sell_count` | 掛單買/賣次數 |
| `trading_buy_sum` / `trading_sell_sum` | 買/賣成交總額 |
| `trading_market_ratio` | 市價單佔比（急於成交的傾向） |
| `trading_source_nunique` | 使用終端數（WEB/APP/API） |
| `swap_buy_count` / `swap_sell_count` | 閃兌買/賣次數 |
| `swap_buy_sum` / `swap_sell_sum` | 閃兌買/賣總額 |

### 5.5 跨表衍生特徵

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

### 5.6 時序速度特徵

> 文獻支持（Google Cloud AML AI、Feedzai）：行為基線偏離比累計絕對值更能捕捉人頭戶模式。

| 特徵 | 說明 | 人頭戶意義 |
|------|------|-----------|
| `first_withdraw_after_deposit_hours` | 入金後第一筆出金的時間間隔 | 比平均間隔更精準的快進快出指標 |
| `velocity_ratio_7d_vs_30d` | 近 7 天交易頻率 / 近 30 天日均 | 行為加速偵測（突然活躍） |
| `max_single_vs_avg_ratio` | 最大單筆金額 / 平均單筆金額 | 異常大額交易的程度 |

### 5.7 紅旗特徵（高判別力）

> 基於 AML 實務經驗設計的高風險指標。

| 特徵 | 說明 | 人頭戶意義 |
|------|------|-----------|
| `dep_to_first_wit_hours` | 首筆入金到首筆出金時間間隔 | 快進快出最直接指標 |
| `twd_to_crypto_out_ratio` | 法幣入金 / 加密貨幣出金比 | 洗錢鏈：法幣→加密→轉出 |
| `tx_amount_cv` | 交易金額變異係數 | Smurfing 偵測（金額過度一致） |
| `rapid_kyc_then_trade` | KYC 完成後 48 小時內有交易 | 開戶即操作的人頭戶模式 |
| `crypto_out_in_ratio` | 加密貨幣出金/入金比 | 資金外流傾向 |
| `same_day_in_out_count` | 同日入金+出金次數 | 當日快速洗錢 |

### 5.8 類別特徵進階編碼

> 參考 Kaggle IEEE-CIS Fraud Detection 第一名策略。

| 編碼方式 | 適用特徵 | 說明 |
|---------|---------|------|
| **Frequency Encoding** | `career`、`income_source` | 類別出現頻率本身是風險信號（罕見職業 → 高風險） |
| **Target Encoding** | `career`、`income_source` | 類別內的黑名單比例（**必須在 CV fold 內執行，避免 leakage**） |

---

## 6. Step 2.5：非監督式異常偵測

> **官方明確要求**：「結合監督式學習（已知模式）與非監督式學習（新型態/異常偵測）」。
> 此步驟是 BitoGuard 三層偵測架構的**第一層**。

### 6.1 為什麼需要非監督

監督式學習只能學到「跟已知 1,640 個黑名單相似的人」。但：
- 犯罪模式會演變，新型態的人頭戶可能不在歷史標籤中
- 有些用戶的**行為組合**在統計上極為罕見，即使單一特徵看起來正常

非監督異常偵測問的是完全不同的問題：

> 「這個用戶的行為組合，在 63,770 個用戶中有多『罕見』？」

**它不看 label，純粹看資料分佈**，因此能捕捉到監督式模型的盲點。

### 6.2 三種演算法互補

| 演算法 | 偵測邏輯 | 擅長抓什麼 | 複雜度 |
|--------|---------|-----------|--------|
| **Isolation Forest** | 隨機切分空間，看誰最快被隔離 | 全局離群點（整體行為極端的用戶） | O(n log n) |
| **HBOS** | 每個特徵獨立建直方圖，看誰落在稀疏區 | 單一特徵極端值（如入金額特別高） | O(n)，最快 |
| **LOF** | 比較每個點跟「鄰居」的密度差 | 局部異常（在類似用戶中行為偏離的人） | O(n²)，用 k-d tree 加速 |

三種演算法從不同角度量化「異常程度」，互補性強。

### 6.3 整合方式：異常分數作為特徵

```python
# 必須在每個 CV fold 內部訓練，避免資料洩漏
def add_anomaly_features(X_train, X_val, contamination=0.033):
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    for name, model in [
        ('if', IForest(n_estimators=200, contamination=contamination)),
        ('hbos', HBOS(n_bins=20, contamination=contamination)),
        ('lof', LOF(n_neighbors=20, contamination=contamination))
    ]:
        model.fit(X_tr_sc)
        # 異常分數越高 = 越異常
        train_scores = model.decision_scores_
        val_scores = model.decision_function(X_val_sc)
```

### 6.4 產出

| 產出 | 維度 | 說明 |
|------|------|------|
| `if_score` | 1 | Isolation Forest 異常分數 |
| `hbos_score` | 1 | HBOS 異常分數 |
| `lof_score` | 1 | LOF 異常分數 |

> **關鍵**：異常分數**不直接做預測**，而是作為新特徵餵進 Step 5 的監督式 ensemble。等於告訴 XGBoost/LightGBM/CatBoost：「除了行為特徵之外，這個人在統計上有多奇怪。」

### 6.5 可選：Autoencoder 異常偵測（Phase 3）

- 只用 class=0 的正常用戶訓練 Autoencoder
- 重建誤差高的用戶 = 行為模式偏離正常群體
- 架構：`[64, 32, 16, 32, 64]` 對稱瓶頸結構
- **主要用途**：簡報展示（重建誤差分佈圖的視覺化效果佳）

---

## 7. Step 2.6：GNN 圖學習特徵提取

> BitoGuard 三層偵測架構的**第二層**。
> 文獻支持：Elliptic Bitcoin Dataset 論文、Google Cloud AML AI 白皮書均採用圖特徵 + 傳統 ML 結合的架構。

### 7.1 為什麼需要圖學習

表格特徵描述的是每個用戶**自己的行為**，但人頭戶最大的特徵之一是**跟其他可疑帳戶有資金往來**：

```
用戶 A：行為看起來正常（入金 5 萬、出金 4 萬）
  但 A 的錢轉給了 B，B 轉給了 C，C 是已知黑名單

  A → B → C (黑名單)

表格特徵看不出 A 有問題，但 GNN 能看到「A 離黑名單只有 2 跳」
```

### 7.2 異質圖建構

從 `crypto_transfer` 建立異質交易圖：

```
節點類型：
  ├── 用戶節點 (user_id)：特徵為表格特徵向量
  └── 錢包節點 (wallet_address)：特徵為零向量

邊類型：
  ├── user → wallet (sends)：鏈上出幣
  ├── wallet → user (funds)：鏈上入幣
  └── user → user (transfers)：站內內轉 (sub_kind=1)
```

### 7.3 模型架構

| 層 | 模型 | 說明 |
|----|------|------|
| Layer 1 | **HeteroSAGE** | 對每種邊類型分別用 SAGEConv 聚合鄰居資訊 |
| Layer 2 | **GATConv (4 heads)** | 在 user-user 邊上加 attention，學習不同鄰居的重要性 |
| Output | 64-dim embedding | 每個用戶在交易網絡中的「角色向量」 |

- 損失函數：`BCEWithLogitsLoss` with `pos_weight = n_neg/n_pos`
- 優化器：Adam (lr=1e-3, weight_decay=1e-4)
- 學習率排程：Cosine Annealing

### 7.4 GNN 產出的整合方式

| 方式 | 做法 | 採用 |
|------|------|------|
| ~~加權混合~~ | ~~0.9 × ensemble + 0.1 × gnn~~ | ❌ 寫死比例，ensemble 不知道 GNN 學了什麼 |
| **Embedding 作為特徵** | GNN 64-dim → PCA 降維至 8~16 維 → 作為特徵餵進 ensemble | ✅ **採用此方式** |

> **為什麼用 embedding 而非直接預測？**
> 1. Ensemble 的 meta-learner 能**自動學習 GNN 信號的最佳權重**
> 2. GNN 捕捉的網絡結構資訊直接參與 base learner 的決策樹分裂
> 3. 不是所有用戶都有交易圖連結（孤立節點），直接預測會有覆蓋問題
> 4. **可解釋性**：SHAP 對 tree-based model 有成熟的 TreeExplainer，對 GNN 的解釋尚不穩定

### 7.5 輕量圖統計特徵（補充）

> 除了 GNN embedding，額外用 NetworkX 提取可解釋的圖統計特徵：

| 特徵 | 說明 | 人頭戶意義 |
|------|------|-----------|
| `pagerank_score` | 用戶在內轉圖中的 PageRank 分數 | 資金流向中心性，中間人帳戶分數高 |
| `in_degree` / `out_degree` | 內轉圖的入度/出度 | Smurfing 模式（多入少出 or 少入多出） |
| `hop_to_blacklist` | BFS 到最近已知黑名單用戶的跳數 | 文獻中最具預測力的圖特徵之一 |
| `connected_component_size` | 所在連通分量大小 | 共犯團體規模 |

> 這些特徵保留業務可解釋性，可直接在 SHAP 中呈現（如「該用戶離最近黑名單僅 2 跳」）。

---

## 8. Step 3：特徵篩選與整合

### 8.1 特徵合併

```
最終特徵向量 = 行為特徵 (47~50) + 異常分數 (3) + GNN embedding (8~16) + 圖統計特徵 (4~5)
             ≈ 62~74 維
```

### 8.2 篩選流程

#### Stage 1：零方差移除
- 移除所有值相同的欄位
- 移除高缺失率特徵（> 80% 缺失）

#### Stage 2：高相關性去重
- 計算特徵間 Pearson 相關係數
- 若兩特徵相關性 > 0.95，保留與 target 相關性較高的那個

#### Stage 3：LightGBM 重要性篩選
- 跑一輪初步 LightGBM，取 `feature_importances_`
- 移除 importance = 0 的特徵
- 搭配 SHAP summary plot 做二次確認

> **注意**：不使用 PCA。因競賽 30% 分數為風險說明能力，PCA 會將特徵轉為抽象主成分，喪失業務可解釋性（如「twd_out_in_ratio 異常」→「PC3 異常」），直接影響 SHAP 解釋品質。**但 GNN embedding 例外 — 因其本身不可解釋，用 PCA 降維不影響整體可解釋性。**

### 8.3 預期結果

- 合併前約 **62~74 個特徵**
- 篩選後預估保留 **50~60 個**

---

## 9. Step 4：不平衡處理

> 30:1 的極端不平衡是本專案最大挑戰。採用**多層次策略**同時應對。

### 9.1 處理策略總覽

| 方法 | 說明 | 優先級 | 應用對象 |
|------|------|--------|---------|
| **Focal Loss** | 非線性降權容易樣本，專注邊界難分樣本 | **P0 核心** | LightGBM |
| **scale_pos_weight** | 線性放大正例梯度 | **P0 核心** | XGBoost、CatBoost |
| **Borderline-SMOTE** | 只在決策邊界附近生成合成正樣本 | **P1 建議** | 每個 CV fold 的訓練集 |
| **PR-curve 閾值最佳化** | 找最大化 F1(class=1) 的閾值 | **P0 必做** | 最終預測 |
| **class_weight='balanced'** | Meta-learner 加權 | P1 | Logistic Regression |

### 9.2 Focal Loss 詳解

> **為什麼比 scale_pos_weight 好？**

`scale_pos_weight=30` 對所有正樣本一視同仁加權。Focal Loss 則：
- 對模型**已經分對的容易樣本**大幅降低 loss（不浪費學習資源）
- 對模型**分不清的邊界樣本**保持高 loss（集中學習決策邊界）

```
黑名單 C（模型已 90% 確信）：loss × α × (1-0.9)^γ → 幾乎忽略
黑名單 D（模型只 40% 確信）：loss × α × (1-0.4)^γ → 重點學習
```

實作方式：自定義 LightGBM objective function，參數建議 α=0.75, γ=2.0。

> **重要**：Focal Loss 與 scale_pos_weight **不同時使用**。LightGBM 用 Focal Loss，XGBoost 和 CatBoost 保持 scale_pos_weight，確保 ensemble 多樣性。

### 9.3 Borderline-SMOTE 詳解

> **為什麼不用標準 SMOTE？**

標準 SMOTE 在所有少數類樣本間插值，包括遠離決策邊界的「典型黑名單」— 這些合成樣本對模型幫助不大。Borderline-SMOTE **只在決策邊界附近的黑名單之間生成合成樣本**，正好彌補模型最弱的地方。

```
原始：49,377 正常 : 1,640 黑名單 = 30:1
SMOTE 後（sampling_strategy=0.3）：49,377 正常 : ~14,813 黑名單 = 3.3:1
```

> **不做到 1:1**。過度合成會引入雜訊，3:1 左右是金融欺詐場景的經驗值。

> **必須在 CV fold 內執行**：只對當前 fold 的訓練集做 SMOTE，驗證集保持原始分佈。

### 9.4 閾值選擇策略

> **關鍵**：官方持有 predict_label 的真實答案，會直接用繳交的硬標籤（0/1）算 F1。閾值選擇直接決定最終分數。

官方 Live Demo 要求說明「選了什麼 threshold、為什麼選這個值」：

1. 基於 **Precision-Recall Curve** 找最大化 `F1(class=1)` 對應的閾值
2. 使用 **5-Fold CV 的 OOF 預測**搜尋最佳閾值，而非單一 fold，提高穩健性
3. 觀察 **predict_label 的機率分佈**是否與 train set 一致，若偏移則需微調閾值
4. 檢查最佳閾值 ±0.05 範圍內的 F1 變化幅度（若過於陡峭，代表閾值敏感，風險高）
5. 最終報告需呈現不同閾值下的 Precision / Recall / F1 trade-off 圖表
6. 簡報同時呈現 `F1(class=1)` 和 `macro-F1`，確保覆蓋官方可能的評分方式

---

## 10. Step 5：監督式 Ensemble 建模

> BitoGuard 三層偵測架構的**第三層**，也是最終做決策的主角。

### 10.1 Stacking Ensemble 架構

```
輸入特徵 (~60 維)
    │
    ├─→ XGBoost (scale_pos_weight)  ──→ OOF probability
    ├─→ LightGBM (Focal Loss)       ──→ OOF probability
    ├─→ CatBoost (auto_class_wt)    ──→ OOF probability
    │
    └─→ Meta-features: [xgb_p, lgb_p, cat_p, max_p, min_p, std_p, mean_p]
                │
                ▼
        Logistic Regression (class_weight='balanced')
                │
                ▼
          最終機率 → PR-curve 閾值 → 0/1 預測
```

### 10.2 Base Learner 配置

| 模型 | 損失函數 | 關鍵參數 | 選用原因 |
|------|---------|---------|---------|
| **XGBoost** | BCE + scale_pos_weight | max_depth=6, n_estimators=2500, lr=0.005 | 表格資料 SOTA，強正則化 |
| **LightGBM** | **Focal Loss** (α=0.75, γ=2.0) | max_depth=4, n_estimators=2500, lr=0.008 | 專注邊界樣本，與 XGB 互補 |
| **CatBoost** | Logloss + auto_class_weights | depth=7, iterations=2500, lr=0.01 | 原生類別特徵支持（career 31 類） |

> **Ensemble 多樣性設計**：三個模型故意使用**不同的損失函數和不同的超參數**（深度、學習率），確保它們從不同角度學習，集成時互補效果更好。

### 10.3 訓練/驗證切分

```
train_label (51,017 users)
  ├── 80% Training ──→ 5-Fold Stratified CV
  └── 20% Hold-out Test ──→ 最終評估

predict_label (12,753 users) → 最終預測繳交
```

- 使用 **Stratified Split** 保持各組正負比一致（均為 96.8% : 3.2%）
- 5-Fold CV 產出 OOF predictions，用於 meta-learner 訓練和閾值搜尋

### 10.4 評估指標

| 指標 | 說明 | 重要性 |
|------|------|--------|
| **F1-score (class=1)** | 黑名單類別的 Precision 與 Recall 調和平均 | **官方主要依據（推定）** |
| **AUC-PR** | Precision-Recall 曲線下面積 | 不平衡場景的核心指標 |
| **macro-F1** | 兩類 F1 的平均 | 附帶呈現，覆蓋官方可能的評分方式 |
| Precision | 預測為黑名單中，真正是黑名單的比例 | 低誤判率 |
| Recall | 實際黑名單被成功識別的比例 | 捕捉率 |
| AUC-ROC | ROC 曲線下面積 | 整體辨別力 |

---

## 11. Step 5.5：半監督學習 — Pseudo-labeling

> 利用 12,753 筆**無標籤的 predict_label 資料**擴充訓練集。

### 11.1 為什麼要做半監督

- train_label 只有 51,017 筆，其中黑名單僅 1,640 筆
- predict_label 的 12,753 筆用戶有完整的特徵，但完全沒用在訓練中
- 高信心的偽標籤可以增加訓練資料量，特別是**正常用戶的多樣性**

### 11.2 做法

1. 用 Step 5 訓練好的 ensemble，對 predict_label 12,753 筆預測機率
2. **只取高信心的**：
   - 機率 ≥ 0.85 → 標為黑名單（預估數十筆）
   - 機率 ≤ 0.05 → 標為正常（預估上萬筆）
   - 中間的 → 不用（不確定就不用）
3. 把偽標籤加回訓練集，重新訓練 ensemble
4. 最多迭代 **2~3 輪**，每輪都跟原始 CV 分數比較

### 11.3 風險控制

| 風險 | 應對 |
|------|------|
| 偽標籤錯誤引入雜訊 | 正例閾值極高（≥0.85），寧可少加也不加錯 |
| Error propagation（錯誤累積） | 最多迭代 3 輪，每輪比較 CV F1，下降就停止 |
| 過度自信 | 只有 F1 有提升才採用，否則回退到原始模型 |

### 11.4 此步驟定位

- **Phase 2 策略**：先完成 Phase 1（非監督 + Focal Loss + SMOTE）看效果，再決定是否啟用
- 主要價值在於增加正常用戶多樣性，讓模型學到更完整的「正常行為邊界」

---

## 12. Step 6：SHAP 可解釋性分析

> 此步佔競賽評分 **30%**，為核心重點。

### 12.1 SHAP 計算

| 層級 | 方法 | 產出 |
|------|------|------|
| **Global（全局）** | `shap.TreeExplainer(model)` 對全部測試集計算 | 整體哪些特徵對黑名單判定最重要 |
| **Local（單一用戶）** | 取個別用戶的 SHAP 值 | 該用戶為什麼被判為高風險 |

### 12.2 視覺化產出

| 圖表 | 用途 | 對應需求 |
|------|------|----------|
| **Summary Plot（蜂群圖）** | 全局特徵重要性 + 方向性 | 風險因子總覽 |
| **Bar Plot** | 全局特徵重要性排序 | 簡報用 |
| **Waterfall Plot** | 單一用戶的風險貢獻拆解 | Live Demo：「這個用戶為什麼是黑名單」 |
| **Force Plot** | 單一用戶推力圖 | 同上，另一種直觀呈現 |
| **Dependence Plot** | 單一特徵值與 SHAP 值的關係 | 解釋特徵閾值效果 |
| **Interaction Plot** | 兩特徵交互作用對 SHAP 值的影響 | 展示組合風險因子（如 `age × income_source`） |

### 12.3 SHAP 交互作用分析

> 補充單特徵層面以外的解釋深度，強化「風險說明能力」（佔 30%）。

利用 `shap.TreeExplainer` 的 `shap_interaction_values` 計算特徵間交互作用，重點關注：

| 交互組合 | 業務意義 |
|----------|----------|
| `age` × `income_source` | 年輕 + 無固定收入 = 高風險組合 |
| `night_tx_ratio` × `twd_out_in_ratio` | 深夜 + 快進快出 = 異常行為疊加 |
| `account_age_days` × `total_volume_twd` | 新帳戶 + 大額交易 = 人頭戶典型模式 |
| `kyc2_speed_days` × `deposit_to_withdraw_speed` | 快速完成 KYC 後立即大額交易 |
| `if_score` × `hop_to_blacklist` | 行為異常 + 接近已知黑名單 = 高度可疑 |

### 12.4 自然語言風險報告

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
| `if_score` | 正 | 「整體行為模式在統計上顯著偏離正常群體」 |
| `hop_to_blacklist` | 負（跳數少推向黑名單） | 「與已知黑名單帳戶的資金關聯距離僅 N 跳」 |
| `account_age_days` | 負（帳齡短推向黑名單） | 「帳戶註冊時間極短，疑似新開人頭戶」 |

---

## 13. Step 7：SSR 穩定性評測

> 驗證 SHAP 解釋的可靠性與穩定性，強化可解釋性報告的可信度。

### 13.1 為什麼需要 SSR

SHAP 給出的解釋如果在資料微小波動下就完全改變，則該解釋不可靠。SSR（Stable Sample Ratio）回答的問題是：

> 「對同一個用戶，稍微擾動他的資料後，SHAP 給出的 Top 風險因子是否一致？」

### 13.2 SSR 公式

$$
SSR_{Top\text{-}k}(\varepsilon) = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\left[ \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}\left[ Top\text{-}k(x_i) = Top\text{-}k(x_i + \delta_t) \right] \geq \tau \right]
$$

- **N**：測試樣本數
- **T**：每個樣本的擾動次數
- **τ**：穩定閾值（≥ τ 比例一致才算穩定）
- **ε**：擾動強度
- **δₜ**：第 t 次擾動（依特徵類型使用不同策略）

### 13.3 計算流程

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

### 13.4 Type-aware 擾動策略

| 特徵類型 | 本專案對應特徵範例 | 擾動方式 |
|---------|-------------------|----------|
| Continuous | age, twd_in_sum, account_age_days | 高斯噪音 N(0, ε·σⱼ) |
| Binary | has_kyc2, user_source | 以機率 ε 翻轉 0↔1 |
| Categorical | career, income_source | 以機率 ε 隨機替換為其他合法類別 |
| Anomaly Score | if_score, hbos_score, lof_score | 不擾動（由上游特徵擾動間接影響） |

### 13.5 實驗參數

| 參數 | 值 | 說明 |
|------|-----|------|
| 擾動級別 ε | 5%, 10%, 15%, 20% | 4 個級別觀察衰減趨勢 |
| 測試樣本數 N | 500 | 從測試集抽樣 |
| 每樣本擾動次數 T | 10 | |
| 穩定閾值 τ | 0.8 | 10 次中 ≥ 8 次一致才算穩定 |
| Top-k 值 | k = 1, 3, 5 | 分別評估首要、前三、前五風險因子穩定性 |

### 13.6 SSR 產出

| 產出 | 說明 |
|------|------|
| SSR_top1 / SSR_top3 / SSR_top5 | 各 k 值下的穩定樣本比例 |
| 擾動衰減曲線 | SSR 隨 ε 增大的下降趨勢圖 |
| 模型間比較 | 各模型（XGBoost/LightGBM/CatBoost）的 SSR 對比 |
| **分群 SSR 報告** | 分別報告黑名單用戶 vs 正常用戶的 SSR |

### 13.7 SSR 分群分析（原創貢獻）

> 文獻指出少數類（黑名單）在決策邊界附近比例更高，SHAP 值對特徵微小變化更敏感。分群報告填補現有文獻空白，具創新性。

| 分群 | 預期 SSR 表現 | 業務意義 |
|------|-------------|----------|
| 正常用戶（class=0） | 較高 | 正常用戶的風險因子解釋穩定，可信度高 |
| 黑名單用戶（class=1） | 較低 | 黑名單用戶在決策邊界附近，解釋較敏感 |

> **簡報論述**：「我們的 SHAP 解釋不僅有業務意義，且經過穩定性驗證 — SSR_top1 達到 X%，代表 X% 的用戶在資料微小波動下，首要風險因子保持不變，解釋具備可信度。此外，我們分別分析了黑名單與正常用戶的穩定性差異，發現 [具體結論]，這在現有文獻中尚屬首次。」

---

## 14. Step 8：消融實驗與模型驗證

> 量化每個模組的貢獻，回答「為什麼需要三層架構」。

### 14.1 消融實驗設計

| 配置 | 包含的模組 | 預期指標 |
|------|----------|---------|
| **Baseline** | 監督式 Ensemble（僅行為特徵） | F1 ≈ 0.34 |
| **+ 異常分數** | Baseline + IF/HBOS/LOF scores | F1 ↑ |
| **+ GNN embedding** | Baseline + GNN 8~16 維 | F1 ↑ |
| **+ 異常分數 + GNN** | Baseline + 兩者 | F1 ↑↑ |
| **+ Focal Loss** | 上述 + LightGBM Focal Loss | F1 ↑ |
| **+ Borderline-SMOTE** | 上述 + SMOTE | F1 ↑ |
| **Full Pipeline** | 所有模組 | F1 最高 |

### 14.2 目的

1. **證明每一層都有貢獻** — 非監督和 GNN 不是裝飾，有實際提升
2. **回應可能的質疑** — 「既然非監督和 GNN 都不直接預測，怎麼證明有用？」用消融實驗的數字說話
3. **簡報加分** — 展示嚴謹的實驗方法論

### 14.3 簡報展示

| 指標 | Baseline | + 異常分數 | + GNN | Full Pipeline |
|------|----------|-----------|-------|---------------|
| F1(class=1) | 0.34 | - | - | - |
| AUC-PR | 0.30 | - | - | - |
| Precision | 0.29 | - | - | - |
| Recall | 0.41 | - | - | - |

> （實際數字待實驗後填入）

---

## 15. Step 9：最終繳交與簡報

### 15.1 繳交項目

| 項目 | 格式 | 內容 |
|------|------|------|
| **預測結果** | CSV | `user_id, status`（12,753 筆） |
| **簡報** | PPT/PDF | 完整方法論與結果展示 |
| **Live Demo** | 互動展示 | BitoGuard 系統演示 |

### 15.2 簡報架構

```
1. 問題定義與挑戰
   ├── 人頭戶偵測的核心難題（30:1 不平衡、動態犯罪模式）
   └── BitoGuard 的設計哲學（三層偵測架構）

2. 系統架構
   ├── 三層偵測架構圖
   ├── 非監督式：異常偵測（為什麼需要、怎麼做、抓到什麼）
   ├── 圖學習：GNN 交易圖分析（網絡結構揭示的共犯關係）
   └── 監督式：Ensemble Stacking（多模型融合決策）

3. 資料處理與特徵工程
   ├── 數據清洗流程
   ├── 特徵設計理念（人頭戶行為剖析）
   └── 不平衡處理策略（Focal Loss + SMOTE + 閾值最佳化）

4. 模型驗證
   ├── 消融實驗（每層貢獻量化）
   ├── 評估指標（F1/Precision/Recall/AUC-PR）
   └── 閾值選擇說明（基於 PR-curve，為什麼選這個值）

5. 可解釋性
   ├── SHAP 全局分析（Top 風險因子）
   ├── SHAP 個體分析（單一用戶為何被判高風險）
   ├── SSR 穩定性驗證（解釋的可信度）
   └── 自然語言風險報告（合規人員可讀的判定理由）

6. Live Demo
   ├── 輸入用戶 → 輸出風險機率值與風險等級
   ├── SHAP Waterfall：該用戶的風險因子拆解
   ├── 交易關係圖譜：視覺化資金流向
   ├── 閾值判斷說明
   └── 模型輸出的機率值分佈圖
```

### 15.3 預測結果格式

| user_id | status |
|---------|--------|
| 967903 | 0 |
| 204939 | 1 |
| 873334 | 0 |
| ... | ... |

- `0`：正常使用
- `1`：黑名單

---

## 附錄：專案目錄結構

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
│   ├── project_plan.md               # 本企劃書
│   └── research/
│       ├── aml-model-literature-review.md      # 文獻調研報告
│       └── model-strategy-research-2026-03.md  # 模型策略調研
│
├── model/                            # 核心 ML Pipeline
│   ├── Feature_rngineering.py        # 特徵工程（60+ 特徵）
│   ├── feature_selection.py          # 特徵篩選（3 階段）
│   ├── anomaly_detection.py          # 非監督異常偵測模組
│   ├── Gnn_model.py                  # GNN 圖學習模組
│   ├── ensemble.py                   # Stacking Ensemble
│   ├── pseudo_labeling.py            # 半監督學習模組
│   ├── shap_explainer.py             # SHAP 可解釋性分析
│   ├── main.py                       # Pipeline 主程式
│   └── requirment.txt                # Python 依賴
│
├── output/                           # 實驗結果
│   └── predict_label.csv             # 最終預測結果
│
├── test_pipeline.py                  # 快速驗證 Pipeline
├── split_all_data.py                 # 資料前處理
├── plot/                             # 競賽官方截圖
└── README.md                         # 專案說明
```
