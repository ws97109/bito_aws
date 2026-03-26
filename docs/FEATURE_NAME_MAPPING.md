# 特徵英文 → 中文名稱對照表（Wei_model）

## 來源檔案

| 檔案路徑 | 變數名稱 | 特徵數 |
|----------|---------|--------|
| `Wei_model/model/shap_explainer.py` | `FEATURE_NAME_MAP` | 47 |
| `frontend/src/utils/graphDataStore.ts` | `FEATURE_NAME_ZH` | 80+ |

---

## 對照表

### 用戶基本特徵

| 英文 Key | 中文名稱 | 說明 |
|----------|---------|------|
| `kyc_speed_sec` | KYC 完成速度（秒） | KYC Level 1 到 Level 2 完成的時間差，過快可能是機器人 |
| `account_age_days` | 帳號年齡（天） | 註冊至今的天數 |
| `age` | 用戶年齡 | 用戶實際年齡（0–120） |
| `is_female` | 是否女性 | sex=2 為女性；黑名單中女性佔比遠高於正常用戶 |
| `is_high_risk_career` | 高風險職業 | 職業代碼屬於區塊鏈業、自由業、無業、珠寶銀樓（14, 22, 23, 29） |
| `is_high_risk_income` | 高風險收入來源 | 收入來源屬於理財投資、挖礦、買賣房地產（4, 8, 9） |
| `career_income_risk` | 職業×收入組合風險 | 同時命中高風險職業與高風險收入來源 |
| `career_freq` | 職業頻率 | 該職業代碼在全體用戶中出現的次數，低頻職業更可疑 |
| `is_app_user` | APP 用戶 | user_source=1 表示來自 APP |
| `reg_hour` | 註冊時間（時） | 註冊時的小時數（0–23） |
| `reg_is_night` | 深夜註冊 | 註冊時間在 0–5 點，可能是機器人或代辦 |
| `reg_is_weekend` | 週末註冊 | 註冊日為週六或週日 |
| `has_kyc_level2` | 已完成 KYC2 | 是否已完成 KYC Level 2 驗證 |
| `kyc_gap_days` | KYC 間隔（天） | KYC Level 1 到 Level 2 之間的天數 |
| `reg_to_kyc1_days` | 註冊到 KYC1（天） | 註冊到首次完成 KYC Level 1 的天數，快速完成較可疑 |

### 法幣行為

| 英文 Key | 中文名稱 | 說明 |
|----------|---------|------|
| `twd_dep_count` | 法幣入金次數 | 台幣入金交易筆數 |
| `twd_dep_sum` | 法幣入金總額 | 台幣入金金額加總 |
| `twd_dep_mean` | 法幣入金均值 | 台幣入金金額平均值 |
| `twd_dep_std` | 法幣入金標準差 | 台幣入金金額標準差 |
| `twd_dep_max` | 法幣入金最大值 | 單筆台幣入金最大金額 |
| `twd_wit_count` | 法幣提領次數 | 台幣提領交易筆數 |
| `twd_wit_sum` | 法幣提領總額 | 台幣提領金額加總 |
| `twd_wit_mean` | 法幣提領均值 | 台幣提領金額平均值 |
| `twd_wit_std` | 法幣提領標準差 | 台幣提領金額標準差 |
| `twd_wit_max` | 法幣提領最大值 | 單筆台幣提領最大金額 |
| `twd_net_flow` | 法幣淨流入 | 入金總額 − 提領總額，負值代表資金淨流出 |
| `twd_withdraw_ratio` | 提領/入金比 | 提領總額 / 入金總額，接近 1 表示幾乎全部提出（上限 10） |
| `twd_smurf_flag` | 結構化交易旗標 | 入金標準差 < 均值×5% 且次數 ≥ 5，偵測金額高度一致的 Smurfing |
| `twd_wit_ip_ratio` | 提領 IP 覆蓋率 | 提領交易中有 IP 記錄的比例，無 IP 表示外部觸發 |

### 加密貨幣行為

| 英文 Key | 中文名稱 | 說明 |
|----------|---------|------|
| `crypto_dep_count` | 加密入金次數 | 加密貨幣入金交易筆數 |
| `crypto_dep_sum` | 加密入金總額 | 加密貨幣入金金額加總（換算台幣） |
| `crypto_dep_mean` | 加密入金均值 | 加密貨幣入金金額平均值 |
| `crypto_dep_max` | 加密入金最大值 | 單筆加密貨幣入金最大金額 |
| `crypto_wit_count` | 加密提領次數 | 加密貨幣提領交易筆數 |
| `crypto_wit_sum` | 加密提領總額 | 加密貨幣提領金額加總（換算台幣） |
| `crypto_wit_mean` | 加密提領均值 | 加密貨幣提領金額平均值 |
| `crypto_wit_max` | 加密提領最大值 | 單筆加密貨幣提領最大金額 |
| `crypto_currency_diversity` | 使用幣種數 | 用戶使用過的不同幣種數量，越多可能在混淆追蹤 |
| `crypto_protocol_diversity` | 使用鏈協定數 | 用戶使用過的不同鏈協定數量（跨鏈行為） |
| `crypto_wallet_hash_nunique` | 錢包地址數 | 用戶涉及的不同錢包地址數，分散資金到多錢包是洗錢典型手法 |
| `crypto_internal_count` | 內轉次數 | 站內用戶間的加密貨幣內轉次數 |
| `crypto_internal_peer_count` | 內轉對象數 | 內轉涉及的不同對象用戶數 |
| `crypto_external_wit_count` | 鏈上提領次數 | sub_kind=0 的鏈上外部提領次數 |
| `crypto_wit_ip_ratio` | 加密提領 IP 覆蓋率 | 加密貨幣提領中有 IP 記錄的比例 |

### 交易行為

| 英文 Key | 中文名稱 | 說明 |
|----------|---------|------|
| `trading_count` | 掛單成交次數 | USDT/TWD 掛單交易成交筆數 |
| `trading_sum` | 掛單成交總額 | 掛單交易成交金額加總 |
| `trading_mean` | 掛單成交均值 | 掛單交易成交金額平均值 |
| `trading_max` | 掛單成交最大值 | 單筆掛單交易最大成交金額 |
| `trading_buy_ratio` | 買單比率 | 買入訂單佔全部掛單的比例 |
| `trading_market_order_ratio` | 市價單比率 | 市價單佔全部掛單的比例 |
| `swap_count` | 一鍵買賣次數 | 一鍵買賣（USDT Swap）交易筆數 |
| `swap_sum` | 一鍵買賣總額 | 一鍵買賣交易金額加總 |
| `total_trading_volume` | 總交易量 | 掛單成交總額 + 一鍵買賣總額 |

### IP 特徵

| 英文 Key | 中文名稱 | 說明 |
|----------|---------|------|
| `ip_unique_count` | 唯一 IP 數 | 用戶使用過的不同 IP 數量 |
| `ip_total_count` | 總 IP 使用次數 | 用戶所有帶 IP 記錄的操作總次數 |
| `ip_night_ratio` | 深夜操作 IP 比率 | 0–5 點操作次數佔總操作次數的比例 |
| `ip_max_shared` | IP 最大共用人數 | 該用戶使用的所有 IP 中，被最多不同用戶共用的那個 IP 的共用人數 |

### 資金停留

| 英文 Key | 中文名稱 | 說明 |
|----------|---------|------|
| `fund_stay_sec` | 資金停留時間（秒） | 最近一次入金到下一筆提領的最短時間差，越短風險越高 |

### 圖特徵

| 英文 Key | 中文名稱 | 說明 |
|----------|---------|------|
| `pagerank_score` | PageRank 分數 | 基於內轉關係圖的 PageRank 值，反映節點重要性 |
| `graph_in_degree` | 圖入度 | 有多少用戶向此用戶轉入（內轉關係圖） |
| `graph_out_degree` | 圖出度 | 此用戶向多少用戶轉出（內轉關係圖） |
| `connected_component_size` | 連通分量大小 | 所在連通分量的節點數，人頭戶常在同一群組 |
| `betweenness_centrality` | 介數中心性 | 節點作為最短路徑中繼站的頻率，抓資金中繼站（近似計算，取樣 500 節點） |

### 跨表整合 / 衍生特徵

| 英文 Key | 中文名稱 | 說明 |
|----------|---------|------|
| `total_tx_count` | 總交易筆數 | 法幣 + 加密 + 掛單 + 一鍵買賣的全部交易筆數 |
| `first_to_last_tx_days` | 首末交易間隔（天） | 第一筆交易到最後一筆交易的天數 |
| `weekend_tx_ratio` | 週末交易比率 | 週六日的交易佔全部交易的比例 |
| `velocity_ratio_7d_vs_30d` | 7天/30天交易加速比 | 近 7 天日均交易次數 / 近 30 天日均交易次數，偵測行為突然加速 |

### 行為紅旗特徵

| 英文 Key | 中文名稱 | 說明 |
|----------|---------|------|
| `dep_to_first_wit_hours` | 入金到首次提領（時） | 第一筆入金到第一筆提領的間隔（小時），過短為紅旗 |
| `twd_to_crypto_out_ratio` | 法幣入/幣出比 | 法幣入金總額 / 加密貨幣提領總額，抓「法幣→幣→鏈上」洗錢鏈（上限 100） |
| `tx_amount_cv` | 交易金額變異係數 | 所有交易金額的標準差/均值，Smurfing 的 CV 極低（需 ≥ 3 筆） |
| `rapid_kyc_then_trade` | KYC 後 48h 交易 | KYC Level 2 完成後 48 小時內即有交易的旗標 |
| `crypto_out_in_ratio` | 加密出/入比 | 加密貨幣提領總額 / 入金總額，出遠大於入為紅旗（上限 10） |
| `same_day_in_out_count` | 同天入出金天數 | 同一天既有入金又有出金的天數（法幣+加密合計） |

### 時序特徵

| 英文 Key | 中文名稱 | 說明 |
|----------|---------|------|
| `tx_interval_mean` | 交易間隔均值（秒） | 連續交易之間的平均間隔時間 |
| `tx_interval_std` | 交易間隔標準差 | 連續交易間隔的標準差 |
| `tx_interval_min` | 交易最短間隔（秒） | 兩筆連續交易之間的最短間隔，極短為快速連續交易紅旗 |
| `tx_interval_median` | 交易間隔中位數 | 連續交易間隔的中位數 |
| `tx_burst_count` | 交易爆發次數 | 1 小時滾動窗口內 ≥ 5 筆交易的次數 |
| `amount_p90_p10_ratio` | 金額 P90/P10 比 | 交易金額第 90 百分位 / 第 10 百分位，比值低表示金額過於均勻（Smurfing）（上限 1000） |
| `active_days` | 活躍天數 | 有交易紀錄的不同天數 |
| `active_day_ratio` | 活躍天數比 | 活躍天數 / 帳號年齡天數 |

### 異常偵測分數

| 英文 Key | 中文名稱 | 說明 |
|----------|---------|------|
| `if_score` | 孤立森林分數 | Isolation Forest 異常分數，在 CV fold 內部訓練產出 |
| `hbos_score` | HBOS 分數 | Histogram-Based Outlier Score，基於直方圖的異常分數 |
| `lof_score` | LOF 分數 | Local Outlier Factor，基於局部密度的異常分數 |
| `composite_risk_score` | 複合風險分數 | 手工加權：提領比×0.25 + 深夜IP×0.15 + 幣種多樣×0.10 + 職業收入風險×0.20 + 未完成KYC2×0.30 |

### GNN 嵌入向量

> **SHAP 解讀注意**：當 SHAP 顯示某個 `gnn_emb_X` 為重要特徵時，代表該用戶在交易網絡中的**拓撲結構位置與已知詐騙用戶相似**，具體可能涉及鄰域連接模式、資金流路徑等圖結構因素。但由於 GNN 嵌入的各維度是模型自動學習產生，單一維度無法對應到具體的圖結構含義（如「連接多個黑名單用戶」或「處在密集轉帳群組」），需搭配顯式圖特徵（如 `pagerank_score`、`graph_in_degree`）輔助解釋。

| 英文 Key | 中文名稱 | 說明 |
|----------|---------|------|
| `gnn_emb_0` | GNN 嵌入 0 | GraphSAGE 16 維嵌入的第 0 維，單一維度無獨立語義，需 16 維組合才能表達圖鄰域結構 |
| `gnn_emb_1` | GNN 嵌入 1 | GraphSAGE 16 維嵌入的第 1 維，單一維度無獨立語義，需 16 維組合才能表達圖鄰域結構 |
| `gnn_emb_2` | GNN 嵌入 2 | GraphSAGE 16 維嵌入的第 2 維，單一維度無獨立語義，需 16 維組合才能表達圖鄰域結構 |
| `gnn_emb_3` | GNN 嵌入 3 | GraphSAGE 16 維嵌入的第 3 維，單一維度無獨立語義，需 16 維組合才能表達圖鄰域結構 |
| `gnn_emb_4` | GNN 嵌入 4 | GraphSAGE 16 維嵌入的第 4 維，單一維度無獨立語義，需 16 維組合才能表達圖鄰域結構 |
| `gnn_emb_5` | GNN 嵌入 5 | GraphSAGE 16 維嵌入的第 5 維，單一維度無獨立語義，需 16 維組合才能表達圖鄰域結構 |
| `gnn_emb_6` | GNN 嵌入 6 | GraphSAGE 16 維嵌入的第 6 維，單一維度無獨立語義，需 16 維組合才能表達圖鄰域結構 |
| `gnn_emb_7` | GNN 嵌入 7 | GraphSAGE 16 維嵌入的第 7 維，單一維度無獨立語義，需 16 維組合才能表達圖鄰域結構 |
| `gnn_emb_8` | GNN 嵌入 8 | GraphSAGE 16 維嵌入的第 8 維，單一維度無獨立語義，需 16 維組合才能表達圖鄰域結構 |
| `gnn_emb_9` | GNN 嵌入 9 | GraphSAGE 16 維嵌入的第 9 維，單一維度無獨立語義，需 16 維組合才能表達圖鄰域結構 |
| `gnn_emb_10` | GNN 嵌入 10 | GraphSAGE 16 維嵌入的第 10 維，單一維度無獨立語義，需 16 維組合才能表達圖鄰域結構 |
| `gnn_emb_11` | GNN 嵌入 11 | GraphSAGE 16 維嵌入的第 11 維，單一維度無獨立語義，需 16 維組合才能表達圖鄰域結構 |
| `gnn_emb_12` | GNN 嵌入 12 | GraphSAGE 16 維嵌入的第 12 維，單一維度無獨立語義，需 16 維組合才能表達圖鄰域結構 |
| `gnn_emb_13` | GNN 嵌入 13 | GraphSAGE 16 維嵌入的第 13 維，單一維度無獨立語義，需 16 維組合才能表達圖鄰域結構 |
| `gnn_emb_14` | GNN 嵌入 14 | GraphSAGE 16 維嵌入的第 14 維，單一維度無獨立語義，需 16 維組合才能表達圖鄰域結構 |
| `gnn_emb_15` | GNN 嵌入 15 | GraphSAGE 16 維嵌入的第 15 維，單一維度無獨立語義，需 16 維組合才能表達圖鄰域結構 |

---

## 運作機制

```
Wei_model 後端 Python                 前端 TypeScript
┌─────────────────────┐              ┌──────────────────────────┐
│ shap_explainer.py   │              │ graphDataStore.ts        │
│                     │              │                          │
│ FEATURE_NAME_MAP    │  API / CSV   │ FEATURE_NAME_ZH          │
│ .get(key, key)      │ ──────────►  │ zhFeatureName(key)       │
│                     │              │                          │
│ → display_names     │              │ → feature_name (中文)    │
└─────────────────────┘              └──────────────────────────┘
                                              │
                                              ▼
                                     ShapPanel.tsx 直接顯示
                                     row.feature.feature_name
```

- **後端**：`SHAPExplainer.__init__` 用 `FEATURE_NAME_MAP.get(f, f)` 轉換，找不到就保留英文原名
- **前端**：`zhFeatureName()` 用 `FEATURE_NAME_ZH[eng] ?? eng` 轉換，同樣 fallback 回英文原名
