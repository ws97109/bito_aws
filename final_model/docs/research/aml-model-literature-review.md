# 技術評估報告：加密貨幣交易所反洗錢與人頭戶偵測

## 報告資訊

- **調研日期**：2026-03-14
- **調研目的**：為 BitoGuard 專案提供學術文獻、模型架構、特徵工程、不平衡處理、可解釋性與業界實踐的全面調研，支援開發決策
- **調研範圍**：2020–2025 年學術論文、公開資料集實驗、業界白皮書、競賽解法
- **關聯專案**：BitoGuard 智慧合規風險雷達（加密貨幣交易所人頭戶二元分類，正負比 30:1）

---

## 執行摘要

### 核心調研問題

BitoGuard 已規劃以 XGBoost/LightGBM + SHAP + SSR 作為核心方案。本報告旨在回答：
1. 學術界對同類問題（加密貨幣 AML、人頭戶偵測）有哪些已驗證的方法？
2. 是否有值得加入的模型架構（尤其圖神經網路 GNN）？
3. 特徵工程可否從文獻中補充新思路？
4. 不平衡比 30:1 的場景下，有哪些比 SMOTE + class_weight 更強的方法？
5. SHAP 在不平衡場景下是否可靠？有沒有替代或補充方法？

### 主要發現摘要

- 在 Elliptic 等標準資料集上，**XGBoost/RandomForest 仍然優於大多數 GNN 方案**，差距在 5–15% F1（illicit class）。GNN 的優勢在於捕捉「共犯網絡結構」，而非個別節點特徵。
- **GraphSAGE 是目前 GNN 架構中在 AML 任務上最穩定的選擇**，其次是 GAT。競賽加分項目（視覺化關聯圖譜）適合以 GraphSAGE + XGBoost 兩段式架構呈現。
- 文獻中最有效的不平衡處理方法為：**Focal Loss（整合進 XGBoost）+ 閾值優化 + class_weight**，三者組合優於單純 SMOTE。
- SHAP 在不平衡場景下有已知可靠性問題，**配合 SSR 穩定性評測是具原創性的補強方案**，與企劃書規劃完全吻合。
- Google Cloud AML AI（2023 年發布）的正式評估（HSBC 案例）顯示：結合交易網絡行為 + KYC 資料的客戶風險評分可達到**減少 60% 誤報、偵測率提升 2–4 倍**的效果。
- IBM NeurIPS 2023 基準研究指出：**GBT（Gradient Boosted Trees）需要特徵工程才能達到競爭力，GNN 則即使不做特徵工程也能得到不錯結果**，但二者在 AML 任務上效果相近。

### 對 BitoGuard 的建議方案

**主路線（核心得分）**：XGBoost/LightGBM + Focal Loss 改良 + 閾值優化 + SHAP (TreeExplainer) + SSR 穩定性評測

**加分路線（視覺化圖譜加分項）**：以 `crypto_relation_user_id` 建構轉帳圖，GraphSAGE 提取節點嵌入後輸入 XGBoost，形成 Tabular + Graph 兩段式 ensemble

---

## 一、學術文獻調研

### 1.1 核心公開資料集

#### Elliptic Dataset（2019 年發布，持續被引用至 2024+）

Weber et al.（2019）由 Elliptic 公司釋出的比特幣交易圖資料集，是 AML 領域最廣泛使用的公開基準：

- **規模**：203,769 個交易節點、234,355 條邊
- **標籤**：4,545 個 illicit（2%）、42,019 個 licit（21%）、餘為 unknown
- **特徵**：每節點 166 個特徵（94 個本地交易特徵 + 72 個聚合鄰居特徵）
- **不平衡比**：約 9:1（有標籤部分），與 BitoGuard 的 30:1 不同但可參考方法

**主要模型比較結果（illicit class F1）**：

| 模型 | Illicit F1（約） | 備註 |
|------|----------------|------|
| Logistic Regression | 0.39 | 基線 |
| Random Forest | 0.80 | 傳統方法最強 |
| XGBoost | 0.76 | 接近 RF |
| GCN | 0.55–0.65 | 依訓練設定差異大 |
| GAT | 0.60–0.66 | 較 GCN 穩定 |
| GraphSAGE | 0.63–0.68 | AUC: 0.8826, AUPRC: 0.6678 |
| EvolveGCN（時序）| 0.72 | 加入時間資訊後顯著提升 |
| GAT-ResNet（改良）| 接近 RF/XGB | 3 層殘差 GAT |

**關鍵結論**：在 Elliptic 資料集上，Random Forest 和 XGBoost 持續優於標準 GNN，但 EvolveGCN 等加入時序資訊的 GNN 可接近傳統方法水準。GAT-ResNet 則在圖結構資料上縮小了差距。

**與 BitoGuard 的關聯**：Elliptic 資料集主要是交易節點分類，而 BitoGuard 是用戶節點分類，但方法論高度可轉移。BitoGuard 的 `crypto_relation_user_id`（內轉對象）可直接建構用戶級交易圖。

#### Elliptic2 Dataset（2024 年，MIT × IBM 合作）

Bellei et al.（2024）在 KDD MLF '24 發布的進化版資料集：

- **規模**：背景圖含 49.3M 節點、196.2M 邊；有標籤子圖 121,810 個
- **標籤**：2.3% 可疑（2,763 個），98.3% 合法，不平衡比高達 43:1，比 BitoGuard 更極端
- **創新**：以「子圖」而非「節點」為分類單位，捕捉洗錢的「形狀（shape）」

**主要模型比較**：

| 方法 | Test F1 | Test PR-AUC | Test ROC-AUC |
|------|---------|------------|-------------|
| GNN-Seg（傳統節點分類 GNN）| 0.398 | 0.026 | 0.537 |
| Sub2Vec | 0.944 | 0.022 | 0.496 |
| GLASS（子圖分類）| 0.933 | 0.208 | 0.889 |

**核心洞察**：主流 GNN 在「節點層級」操作，但洗錢行為本質是「子圖層級」問題（一組帳戶共同操作）。GLASS 方法大幅提升 PR-AUC（從 0.026 → 0.208），代表**考慮帳戶群組關係**比單一帳戶特徵更有效。

**與 BitoGuard 的關聯**：`crypto_relation_user_nunique`（內轉對象帳戶數）和相互交易關係，實際上就是在建構這種子圖特徵。可以考慮將「有互相內轉的用戶群組」作為一個整體特徵。

#### IBM AML Synthetic Dataset（NeurIPS 2023）

Altman et al.（2023）提出的合成金融交易資料集基準：

- **背景**：真實 AML 訓練資料極度稀缺（洗錢交易不被完整標記），這份合成資料的標籤是完整的
- **關鍵發現**：GBT（Gradient Boosted Trees）需要特徵工程才有競爭力；GNN 即使不做特徵工程也能得到不錯結果，兩者最終效果相近
- **實務意義**：這說明 GBT 的強大主要來自「人工特徵工程」，而 GNN 能「自動從圖結構中學習特徵」

**與 BitoGuard 的關聯**：企劃書中已規劃了豐富的特徵工程（50–60 個特徵），這正是讓 XGBoost/LightGBM 優於 GNN 的關鍵投資。

---

### 1.2 重要學術論文

#### 論文一：Bitcoin Money Laundering Detection via Subgraph Contrastive Learning（2024）

**發表**：Entropy (MDPI), Vol.26, No.3, 2024
**作者**：Ouyang et al.

**方法（Bit-CHetG）**：
- 以異質圖建模比特幣交易（交易節點 + 地址節點）
- Transaction Subgraph Embedding（TSE）：用 GCN 提取交易子圖拓撲特徵
- Address Feature Aggregation（AFA）：聚合錢包地址特徵
- 監督對比學習損失（Supervised Contrastive Loss）：拉近同類、推離異類

**性能**：
- Elliptic 資料集 Micro-F1：0.919（較基線 GCN 的 0.82 高出 10%+）
- BlockSec 資料集 Micro-F1：0.815（較 HAN 的 0.718 高出 10%）

**對 BitoGuard 的可借鑑之處**：
- 對比學習可以作為處理不平衡的補充手段（少數類樣本相互靠攏）
- 地址特徵聚合的邏輯可映射到 BitoGuard 的「關聯用戶特徵聚合」

---

#### 論文二：Anti-Money Laundering in Cryptocurrency via Multi-Relational Graph Neural Network（2023）

**發表**：Springer LNCS, PAKDD 2023
**核心創新**：提出多關係 GNN（BitcoNN），將不同類型的交易邊（入金、出金、內轉）以不同關係建模

**關鍵發現**：
- 在 Elliptic 資料集上優於標準 GCN 和 GAT
- 多關係建模能捕捉「透過不同管道流動」的資金洗錢模式

**與 BitoGuard 的關聯**：BitoGuard 資料包含法幣轉帳、加密貨幣轉帳、USDT 交易三種不同類型的「邊」，正好適合多關係 GNN 建模。若實作 GNN 加分項，建議考慮此架構。

---

#### 論文三：Graph Contrastive Pre-training for Anti-money Laundering（GCPAL）（2024）

**發表**：International Journal of Computational Intelligence Systems, Springer, 2024

**方法**：
- 自監督對比預訓練：從無標籤交易網路中挖掘監督訊號
- 大幅降低對標籤資料的依賴
- GCPAL with GIN encoder 達到最佳整體性能
- 三種 GNN 架構比較：GraphSAGE 最穩定，其次 GAT，再次 GCN

**與 BitoGuard 的關聯**：
- BitoGuard 的正負比是 30:1（有效正樣本僅 1,640 個），對比學習預訓練可以幫助模型在少量標籤下學到更好的表示
- 如果想嘗試 GNN，以 GraphSAGE 作為首選架構是有文獻支持的

---

#### 論文四：Deep Learning Approaches for Anti-Money Laundering on Mobile Transactions（Review, 2025）

**發表**：arXiv:2503.10058, 2025（評述論文）

**整理的主要模型比較（在 Elliptic 資料集上）**：

| 模型類別 | 代表方法 | 少數類 F1 |
|---------|---------|---------|
| 傳統 ML | Random Forest, XGBoost | 0.76–0.83 |
| CNN 系列 | 1D-CNN | ~0.71 |
| RNN/LSTM | GRU | ~0.68 |
| GNN 系列 | EvolveGCN | 0.72 |
| GNN+LSTM | GEGCN-BiLSTM | >0.72 |
| 混合框架 | CRP-AML (Conv1D+SHAP) | 0.8251 |

**重要觀察**：
- 模型訓練在含錯誤標籤的資料上常「學習錯誤模式」
- WGAN 生成合成詐欺交易可改善不平衡
- Recall-First Threshold (RFT) 是專為最大化 Recall 設計的閾值策略
- 可解釋性在 AML 系統中是「監管要求的缺口」，被視為研究機會

---

#### 論文五：Detecting Anomalous Cryptocurrency Transactions: An AML/CFT Application（Electronic Markets, 2023）

**發表**：Electronic Markets, Springer, 2023

**方法**：
- 使用 GCN 和 GAT 建模比特幣交易網絡
- GAT 被首次應用於比特幣 AML 異常偵測
- 提出行為特徵提取框架

**績效**：GCN 在多數指標上優於傳統分類器；GAT 在精準率上表現更佳

---

#### 論文六：Graph-Based LSTM for Anti-money Laundering（Neural Processing Letters, 2022）

**發表**：Neural Processing Letters, Springer, 2022

**方法**：結合 GCN 與 LSTM（Temporal Graph Convolutional Network，T-GCN）

**核心發現**：加入時序資訊後，GNN 在偵測「分期洗錢」（peel-chain）模式時優於靜態 GNN

**與 BitoGuard 的關聯**：BitoGuard 的 `first_to_last_tx_days`、`deposit_to_withdraw_speed` 等時序特徵，部分對應了 T-GCN 所捕捉的時序洗錢模式。

---

### 1.3 文獻研究空缺

目前文獻對加密貨幣交易所內的「用戶層級」分類研究相對稀少，大多集中在：
1. 鏈上交易分類（比特幣 UTXO 模型）
2. 以太坊地址分類

**BitoGuard 場景的特殊性**：
- 資料包含「鏈外」的法幣交易（twd_transfer）
- 有完整 KYC 資訊（age、career、income_source）
- USDT 場內交易（掛單、閃兌）在文獻中幾乎無先例
- 人頭戶偵測（mule account）比一般洗錢偵測更強調「身份行為一致性」

這意味著 BitoGuard 在特徵設計上具有**原創性**，尤其是將 KYC 資訊與交易行為特徵的結合。

---

## 二、可參考模型架構

### 2.1 表格資料模型比較

在加密貨幣 AML 和金融詐欺偵測任務中，各模型的實際表現如下：

| 模型 | 在 AML 任務表現 | 可解釋性 | 適用情境 |
|------|--------------|---------|---------|
| **XGBoost** | 最強（F1 0.76–0.83） | SHAP TreeExplainer | 所有場景，主力選擇 |
| **LightGBM** | 接近 XGBoost | SHAP TreeExplainer | 大數據、速度優先 |
| Random Forest | 有時比 XGBoost 高 | SHAP TreeExplainer | 穩定基線 |
| CatBoost | 與 XGBoost 接近 | SHAP TreeExplainer | 類別特徵多時 |
| TabNet | 通常低於 GBDT 5–10% | 內建注意力機制 | 需要深度學習可解釋性 |
| FT-Transformer | 7/11 資料集優於 GBDT | 較複雜 | 特徵互動複雜的情境 |
| Logistic Regression | 基線（F1 ~0.39） | 線性可解釋 | 基線對照 |

**結論**：對 BitoGuard 而言，XGBoost 和 LightGBM 是有充分文獻支持的最佳選擇，符合企劃書規劃。

#### 進階選項：CatBoost

CatBoost 在類別特徵（`career` 31 類、`income_source` 10 類）較多的情況下，無需手動 Label Encoding 即可有效處理，且同樣支援 SHAP TreeExplainer。可考慮作為 ensemble 的第三個成員。

---

### 2.2 圖神經網路架構比較

#### GNN 架構選型指南

| 架構 | 核心機制 | 優點 | 缺點 | AML 推薦度 |
|------|---------|------|------|-----------|
| **GraphSAGE** | 鄰居採樣聚合 | 可擴展至大規模圖；在 Elliptic 上最穩定 | 不含注意力機制 | ★★★★★ |
| **GAT** | 注意力加權聚合 | 自動學習鄰居重要性；在精準率上表現佳 | 計算成本較高 | ★★★★☆ |
| **GCN** | 均勻鄰居聚合 | 簡單、快速 | 大度節點問題；易過平滑 | ★★★☆☆ |
| **EvolveGCN** | 時序 GCN | 捕捉動態圖變化 | 實作複雜 | ★★★★☆ |
| **GIN** | 圖同構網路 | 表達能力最強（理論）| 穩定性差 | ★★★☆☆ |
| **GLASS** | 子圖分類 | 在 Elliptic2 上最佳 | 需要子圖採樣 | ★★★★☆ |

#### 推薦：GraphSAGE 作為 BitoGuard 加分項

文獻支持（GCPAL, 2024）明確指出 GraphSAGE 在 AML 任務上最穩定。NVIDIA 的技術部落格（2023）也具體展示了 GraphSAGE + XGBoost 的兩段式架構：

```
步驟 1：以 user_id 建構交易圖（用戶為節點，crypto_transfer internal 為邊）
步驟 2：GraphSAGE 訓練產生每個用戶的圖嵌入向量（graph embedding）
步驟 3：將圖嵌入向量與原始表格特徵拼接
步驟 4：XGBoost 最終分類
```

這個架構的優點：
- GraphSAGE 嵌入捕捉「帳戶在圖中的位置」（是否在洗錢叢集附近）
- 最終分類仍用 XGBoost，保留 SHAP 可解釋性
- 圖視覺化本身即是評分加分項的展示材料

---

### 2.3 Ensemble 架構

#### 方案 A：傳統表格 Ensemble（基礎方案）

```
XGBoost + LightGBM + CatBoost
→ Soft Voting 或 Stacking（MetaLearner = LightGBM）
```

文獻支持：IEEE-CIS Fraud Detection 競賽第一名（2019）即使用 XGBoost + CatBoost + LightGBM 的 ensemble。

#### 方案 B：Tabular + Graph Hybrid Ensemble（進階加分方案）

```
[表格路徑] XGBoost(原始特徵) → P_tabular
[圖路徑]   GraphSAGE(轉帳圖) → embedding → XGBoost → P_graph
→ Soft Voting: P_final = α·P_tabular + (1-α)·P_graph
```

支持此方案的文獻：
- NVIDIA (2023)：「GNNs convert the transaction data into embeddings, which are then fed to an XGBoost model to predict fraud score」
- Ensemble GNN paper (2025)：整合 GCN + GAT + GIN 的 ensemble 達到 F2=74.8%

---

## 三、特徵工程參考

### 3.1 文獻中常見的 AML 特徵類別

基於文獻綜整（Weber et al. 2019, Ouyang et al. 2024, FATF AML Red Flags 2023），加密貨幣 AML 常見特徵分為以下五類：

#### 類別一：交易統計特徵（Transaction Statistics）

| 特徵 | 業界/文獻描述 | BitoGuard 對應 |
|------|------------|--------------|
| 交易次數（入/出） | Frequency of transactions | `twd_in_count`, `crypto_in_count` |
| 交易金額（總計/均值/最大值） | Amount volume & variability | `twd_in_sum`, `twd_out_mean`, `twd_max_single` |
| 淨流量 | Net flow direction | `twd_net_flow` |
| 出入比 | Throughput ratio | `twd_out_in_ratio` |

**差異**：文獻中通常只有「鏈上」交易，BitoGuard 還有法幣（twd）交易，是獨特資訊維度。

#### 類別二：時序/速度特徵（Temporal/Velocity Features）

| 特徵 | 業界描述 | BitoGuard 對應 |
|------|---------|--------------|
| 交易速度激增 | Velocity spikes（24–72 小時） | `avg_tx_per_active_day` |
| 存款後提款速度 | Rapid withdrawal after deposit | `deposit_to_withdraw_speed` |
| 帳戶年齡 | Account age | `account_age_days` |
| 首末交易間隔 | Transaction lifespan | `first_to_last_tx_days` |
| 異常時段 | Off-hours activity | `night_tx_ratio`, `weekend_tx_ratio` |
| KYC 完成速度 | KYC completion speed | `kyc1_speed_hours`, `kyc2_speed_days` |

**文獻缺口補充**：企劃書規劃了 `kyc2_speed_days`（KYC 完成速度），這在文獻中極少見，但業界紅旗報告（ComplyAdvantage, 2023）指出「迅速完成 KYC 後立即大額交易」是人頭戶典型行為，此特徵設計具有實務依據。

#### 類別三：圖/網絡特徵（Graph/Network Features）

| 特徵 | 文獻描述 | BitoGuard 對應 |
|------|---------|--------------|
| 關聯帳戶數 | Number of counterparties | `crypto_relation_user_nunique` |
| 錢包地址多樣性 | Wallet address diversity | `crypto_wallet_nunique` |
| 跨鏈行為 | Cross-chain behavior | `crypto_protocol_nunique` |
| 圖中心性（需 GNN） | Betweenness/PageRank | GNN 嵌入可自動捕捉 |

**文獻支持的新特徵建議**（未在企劃書中）：

1. **洗錢叢集距離**（Money Laundering Cluster Distance）：若有已知黑名單用戶，計算目標用戶在圖中與最近黑名單的跳數（hop distance）。文獻顯示這是最具預測力的圖特徵之一。

2. **PageRank 評分**：在轉帳圖中計算用戶的 PageRank，高 PageRank 的「中間人」帳戶是洗錢網絡的核心節點。

3. **入度/出度比不對稱性**：若某用戶大量接收小額轉帳再集中大額轉出（Smurfing 模式），入度/出度比會異常。

#### 類別四：身份/KYC 特徵（Identity Features）

| 特徵 | 文獻描述 | BitoGuard 對應 |
|------|---------|--------------|
| 職業與收入來源 | Occupation & income source | `career`, `income_source` |
| 年齡 | Age | `age` |
| 性別 | Gender | `sex` |
| 地理位置 | Geographic risk | 未包含（台灣境內） |

**文獻補充**：Google Cloud AML AI 的輸入模型明確列出以下 KYC 欄位為風險評分輸入：`birth_date`, `nationalities`, `occupation`, `assets_value_range`, `education_level_code`。BitoGuard 已涵蓋大部分，但缺少資產規模資訊。

#### 類別五：行為模式特徵（Behavioral Pattern Features）

| 特徵 | 業界描述 | BitoGuard 對應 |
|------|---------|--------------|
| 使用終端多樣性 | Multi-device usage | `trading_source_nunique` |
| IP 多樣性 | Multi-IP activity | `twd_ip_nunique` |
| 市價單偏好 | Urgency to execute | `trading_market_ratio` |
| 幣種多樣性 | Multi-currency usage | `crypto_currency_nunique` |

### 3.2 文獻中的時序特徵創新做法

#### 滾動窗口特徵（Rolling Window Aggregation）

業界最佳實踐（IBM, Google Cloud AML AI）強調使用滾動窗口捕捉行為基線的偏離：

```
特徵範例：
- 過去 7 天交易次數 vs. 過去 30 天日均交易次數（速度比）
- 過去 7 天最大單筆金額 vs. 歷史均值的倍數
- 最近 3 筆交易的時間間隔均值（急迫性指標）
```

**建議**：若資料量允許，可針對 `twd_transfer` 和 `crypto_transfer` 計算這類相對速度特徵，而非只有累計絕對值。

#### 時序順序特徵（Sequential Pattern）

EvolveGCN（Weber et al.）和 T-GCN（2022）研究發現，「資金移動的時序順序」比靜態圖更具預測力：

- 入金 → 立即出金（法幣套現）
- 入金 → 內轉分散 → 外部提領（典型分層）
- 外部入幣 → 閃兌 → 法幣提領（快速法幣化）

**建議**：`deposit_to_withdraw_speed` 已捕捉部分資訊；可進一步區分「入金後第一筆出金」的時間，比「平均間隔」更能捕捉人頭戶模式。

---

## 四、不平衡處理方法評估

### 4.1 BitoGuard 不平衡程度定性

- 訓練集：正常 49,377 vs. 黑名單 1,640，比例約 **30:1**
- 在 AML 文獻中屬於「高度不平衡」（extreme imbalance，比 Elliptic 的 9:1 更嚴重）
- Elliptic2 的 43:1 比 BitoGuard 更極端，但方法論可直接參考

### 4.2 各方法的文獻評估

#### 方法一：class_weight（已在企劃書中）

**評估**：最基礎且有效的方法。XGBoost 的 `scale_pos_weight` 參數（設為 30）等同於給少數類增加 30 倍損失權重。文獻一致支持，可作為基線。

**限制**：只改變損失函數權重，不改變資料分佈，對決策邊界的影響有限。

#### 方法二：SMOTE（已在企劃書中）

**評估**：研究顯示 SMOTE 在 AML 場景的效果**好壞參半**：
- 優點：增加少數類樣本多樣性
- 缺點：合成樣本可能不符合真實人頭戶的業務邏輯；在極端不平衡（>20:1）時效果不穩定
- 重要注意：必須在訓練集上做 SMOTE，不能在全資料集做，否則會造成資料洩漏

**替代建議**：考慮 **ADASYN**（自適應合成），它會對難以分類的樣本生成更多合成樣本，在 AML 場景的一些研究中優於標準 SMOTE。

#### 方法三：Focal Loss（推薦添加）

**原理**：針對「容易分類的樣本」降低損失權重，讓模型專注在難以分類的樣本（通常是少數類的邊界樣本）。

```python
FL(pt) = -α(1-pt)^γ · log(pt)
# γ=2 時，pt=0.9（容易）的損失權重降至 (1-0.9)^2 = 0.01
# γ=2 時，pt=0.1（困難）的損失權重維持在 (1-0.1)^2 = 0.81
```

**文獻支持**：
- XGBoost 的 Modified Focal Loss（Trisanto et al., 2021）：在信用卡詐欺偵測上改善輕度不平衡場景
- Focal-aware Cost-sensitive Boosted Tree（ScienceDirect, 2022）：在信用評分中的不平衡問題上顯著改善
- CRP-AML framework（2025）：使用 Focal Loss 配合 Conv1D，少數類 F1 達 82.51%

**實作方式**（XGBoost）：

```python
# 自定義 Focal Loss for XGBoost
def focal_loss_obj(y_pred, y_true):
    gamma = 2.0
    alpha = 0.25
    pt = 1 / (1 + np.exp(-y_pred))
    focal_weight = alpha * (1 - pt) ** gamma
    grad = focal_weight * (pt - y_true)
    hess = focal_weight * pt * (1 - pt)
    return grad, hess
```

**建議**：Focal Loss 作為 class_weight 的替代或補充，重點調整 γ（建議從 0.5 開始，不宜過大）。

#### 方法四：閾值優化（已在企劃書中，且最重要）

**評估**：在 AML 場景下，調整分類閾值是**比改變資料更有效且風險更低**的方法。

**文獻最佳實踐**：CRP-AML（2025）提出 **Recall-First Threshold（RFT）**：
1. 先設定最低可接受 Recall（如 0.8）
2. 在滿足 Recall 約束下，最大化 Precision
3. 這比「最大化 F1 的閾值」更符合 AML 業務邏輯（寧可多查，不要漏掉）

**建議**：企劃書已規劃 Precision-Recall Curve 選閾值，這是正確方向。可明確在簡報中說明「採用 Recall-First 策略」，與業界最佳實踐對齊。

#### 方法五：Anomaly Detection 混合方法（進階選項）

部分研究將 AML 問題拆分為兩階段：

```
階段一：Isolation Forest / Autoencoder → 篩出「異常用戶」候選
階段二：XGBoost 對異常候選進行二元分類
```

**優點**：第一階段不依賴標籤，可以利用全部 63,770 個用戶的資訊
**缺點**：異常不等於黑名單，第一階段可能引入大量雜訊

**評估**：對 BitoGuard 而言，這個方法風險較高，不建議作為主要路線，但可作為特徵工程的補充（用 Isolation Forest score 作為一個輸入特徵）。

### 4.3 不平衡處理策略總結建議

針對 BitoGuard 的 30:1 不平衡情境，建議以下優先順序：

| 優先級 | 方法 | 預期效果 | 實作複雜度 |
|--------|------|---------|-----------|
| P1（必做）| class_weight='balanced' | 穩定基線改善 | 極低 |
| P1（必做）| 閾值優化（PR Curve）| 最直接的精準率/召回率控制 | 低 |
| P2（建議）| Focal Loss（γ=0.5–2.0）| 針對邊界樣本改善 | 中 |
| P3（備選）| SMOTE 或 ADASYN | 增加少數類多樣性 | 中 |
| P4（進階）| Isolation Forest score 作為特徵 | 無監督異常分數補充 | 中高 |

---

## 五、可解釋性方法評估

### 5.1 SHAP 在不平衡場景下的已知問題

#### 問題一：KernelSHAP 的邊際採樣問題

Aas, Jullum & Løland（2021）指出：當使用邊際採樣的 KernelSHAP 時，在不平衡資料集中可能將高權重放在「不可能出現」的樣本組合上，導致特徵貢獻估計失真。

**BitoGuard 的緩解方案**：企劃書使用的是 **TreeExplainer**（非 KernelSHAP），TreeExplainer 基於決策樹路徑計算，不依賴採樣，**此問題不適用**。

#### 問題二：SHAP 特徵選擇 vs. 內建特徵重要性

2024 年研究指出：在大型資料集上，SHAP 的計算成本高，且對小特徵子集的選擇效果有時不如 XGBoost 內建的 feature_importance。但對大特徵子集（如 30+ 特徵），SHAP 優於內建方法。

**BitoGuard 的情況**：預期保留 30–40 個特徵，SHAP 在此規模下是合適的選擇。

#### 問題三：少數類的 SHAP 穩定性

這正是企劃書中 SSR（Stable Sample Ratio）評測要解決的問題。文獻中尚未有系統性研究此問題（這是 BitoGuard 的**原創貢獻**），但理論上：
- 少數類（黑名單用戶）在決策邊界附近的比例更高
- 邊界附近的 SHAP 值對特徵值微小變化更敏感
- 因此，黑名單用戶的 SSR 可能低於正常用戶的 SSR

**建議**：在 SSR 分析中，分別報告正常用戶和黑名單用戶的 SSR，這個差異本身就是有意義的發現。

### 5.2 其他在金融 AML 場景被驗證的可解釋性方法

#### LIME（Local Interpretable Model-agnostic Explanations）

- **特性**：建立局部線性代理模型
- **AML 中的使用**：IEEE XAI for Financial Fraud Detection (2025) 指出，LIME 適合單一用戶的「局部解釋」，但一致性低於 SHAP
- **與 SHAP 的差異**：LIME 解釋可能每次結果不同（隨機採樣）；SHAP 基於博弈論，具有唯一性
- **建議**：BitoGuard 已選擇 SHAP，LIME 可作為輔助驗證（比較兩者的 Top-3 特徵是否一致）

#### Integrated Gradients（深度學習專用）

適用於 TabNet 等深度學習模型，對 XGBoost 不適用。

#### SHAP Interaction Values

XGBoost + SHAP 可計算**特徵交互作用值**（SHAP interaction values），例如：
- `age` × `income_source` 的交互作用：年輕人 + 無固定收入來源的組合效應
- `night_tx_ratio` × `twd_out_in_ratio` 的組合：深夜出金的異常程度

**建議**：在 SHAP 分析中加入 Dependence Plot（已在企劃書中規劃），配合 interaction_index 參數可視覺化重要交互作用。

#### Counterfactual Explanations（反事實解釋）

近年在 AML 場景出現新應用：「這個用戶需要哪些特徵改變，才能從黑名單變為正常？」

```
範例反事實解釋：
「若該用戶的 deposit_to_withdraw_speed 從 0.5 小時增加至 24 小時，
 且 crypto_relation_user_nunique 從 15 降低至 3，模型會將其分類為正常。」
```

**文獻支持**：Explainable AI for Forensic Analysis (MDPI, 2025) 提及此方法在金融場景的應用。

**建議**：此方法可加強 Live Demo 的說服力（讓評審理解「怎樣才算正常」），但實作複雜度較高，列為備選。

### 5.3 可解釋性策略建議

針對 BitoGuard 的評分結構（30% 為可解釋性），建議分層展示：

| 層級 | 方法 | 對應評分點 |
|------|------|-----------|
| 全局 | SHAP Summary Plot（蜂群圖）+ Bar Plot | 展示整體風險因子 |
| 用戶 | SHAP Waterfall Plot + Force Plot | Live Demo 單一用戶解釋 |
| 穩定性 | SSR 評測結果（跨擾動強度曲線）| 展示解釋可信度 |
| 業務語言 | 模板式自然語言風險報告 | 展示實務可用性 |

---

## 六、競賽與業界實踐

### 6.1 Kaggle 相關競賽

#### IEEE-CIS Fraud Detection（2019，但方法至今仍適用）

- **資料**：Vesta Corporation 的電商交易詐欺，約 590K 筆，3.5% 詐欺率
- **第一名方案**（Chris Deotte, NVIDIA）：
  - XGBoost + CatBoost + LightGBM **Ensemble**
  - 特徵工程：從交易欄位中提取 UID（用戶識別組合），再做 GroupBy 聚合特徵
  - **GroupBy 特徵**是最關鍵的特徵工程：`card1+addr1+D1` 組合識別同一張信用卡用戶
  - AUC-ROC 約 0.93

**可借鑑之處**：BitoGuard 的用戶識別是現成的（`user_id`），但 IEEE-CIS 的 UID 構建邏輯（「哪些帳號是同一個人操作的」）類似於識別多個 IP 使用同一帳戶、或多個帳戶使用相同行為模式。

#### AML 2021 Kaggle Competition

Kaggle 上的 AML 競賽，部分參與者採用 GNN 方法（見 kaggle.com/code/issacchanjj/anti-money-laundering-detection-with-gnn），多數 Top 方案仍以 LightGBM + 豐富特徵工程為主。

#### SAML-D 合成資料集

Kaggle 上的 Anti Money Laundering Transaction Data（SAML-D），提供合成的交易監控資料，可作為方法測試的輔助資源。

### 6.2 業界實踐

#### Google Cloud AML AI（2023 年發布）

**技術方法**：
- 客戶中心化風險評分（Customer-centric risk scoring）
- 輸入：交易資料（支援 WIRE, CARD, CASH, CHECK, CRYPTO 等類型） + KYC 資料（職業、年齡、資產規模等）
- 輸出：整合機器學習生成的客戶風險分數（替代傳統規則式警報）
- 技術：基於 Vertex AI 和 BigQuery，包含圖分析和行為基線

**HSBC 實際效果**：
- 可疑活動識別率提升 **2–4 倍**
- 警報數量減少 **60%**（假陽性大幅降低）
- 調查時間從數週縮短至約 **8 天**

**與 BitoGuard 的關聯**：Google AML AI 的輸入資料模型（KYC + 交易類型分類）與 BitoGuard 的資料結構高度一致。這是業界對「KYC + 交易行為組合特徵」有效性的有力佐證。

#### Elliptic（業界先驅）

**技術核心**：
- 圖分析與機器學習結合，追蹤超過 20 億個已標籤地址
- 支援 100+ 加密資產，每月處理 200 萬次篩查
- Holistic 技術：次秒級全鏈路追蹤

**Elliptic2 資料集**：Elliptic 本身也是學術研究的重要推動者，與 MIT、IBM 合作釋出 Elliptic2 資料集（見 1.1 節）。

#### Chainalysis

**技術方法**：
- Knowledge Graph：連接超過 24 萬億美元的鏈上資金流動
- KYT（Know Your Transaction）：實時交易監控，基於規則 + ML 的風險評分
- Reactor：鏈上調查工具，可視覺化資金流向

**與 BitoGuard 的差異**：Chainalysis 和 Elliptic 主要做**鏈上地址追蹤**，而 BitoGuard 做的是**交易所內用戶行為分類**，兩者互補而非競爭。BitoGuard 的法幣交易資料（twd_transfer）是 Chainalysis 無法獲取的獨特信息。

#### Feedzai / ComplyAdvantage

**行為基線方法**：
- 建立用戶行為基線（歷史交易均值、頻率、對象分佈）
- 對當前交易與基線的偏離程度評分
- 核心概念與 `deposit_to_withdraw_speed`（入金到出金的速度偏離）一致

---

## 七、與 BitoGuard 的綜合關聯性分析

### 7.1 企劃書現有方案的文獻支持度評估

| 企劃書方案 | 文獻支持度 | 評估 |
|-----------|-----------|------|
| XGBoost/LightGBM 作為主模型 | ★★★★★ | 在 AML tabular 任務上持續是 SOTA，完全合理 |
| SHAP TreeExplainer | ★★★★★ | 最適合 tree-based 模型的可解釋性方法，規避了 KernelSHAP 的採樣問題 |
| Stratified K-Fold 交叉驗證 | ★★★★★ | AML 不平衡場景的標準做法 |
| SSR 穩定性評測 | 原創性高（文獻稀少）| 具備學術新穎性，直接回應「SHAP 在不平衡場景可靠性」的研究缺口 |
| SMOTE | ★★★☆☆ | 有效但在高度不平衡場景效果不穩定，建議搭配 Focal Loss |
| class_weight | ★★★★★ | 最穩健的基礎方案 |
| Precision-Recall 閾值選擇 | ★★★★★ | 業界最佳實踐，與 RFT 方法一致 |
| 關聯圖譜加分項（GNN）| ★★★★☆ | GraphSAGE + XGBoost 兩段式是有文獻支持的實用架構 |

### 7.2 建議補充到企劃書的方法

1. **Focal Loss 替代/補充 SMOTE**：文獻顯示在極端不平衡下效果更穩定，實作簡單
2. **CatBoost 加入 Ensemble**：對 career（31 類）、income_source（10 類）等類別特徵處理更佳，可豐富 ensemble 多樣性
3. **圖特徵手工提取（輕量版 GNN 方案）**：若不做完整 GNN，可手工計算用戶在轉帳圖中的度數（degree）、PageRank、與已知黑名單的跳數，作為表格特徵輸入 XGBoost
4. **SHAP Interaction Values**：在簡報中展示 `night_tx_ratio` × `twd_out_in_ratio` 等特徵交互的視覺化，強化業務解釋力
5. **分層 SSR 報告**：分別報告黑名單用戶和正常用戶的 SSR，說明「解釋在哪些用戶群體上更穩定」

---

## 八、具體可借鑑的行動建議

### 短期（核心得分，低風險）

1. **特徵工程補充**：增加「與已知黑名單用戶的最短轉帳路徑跳數」特徵。若企劃書中的 `crypto_relation_user_id`（內轉關係）能建圖，可直接用 BFS 計算。這是文獻中最具預測力的圖特徵，且不需要完整 GNN。

2. **不平衡處理升級**：在 XGBoost 中使用自定義 Focal Loss（γ=1.0, α 按不平衡比設定），替換或補充現有 SMOTE。

3. **模型加入 CatBoost**：作為 ensemble 第三成員，利用其對 career 和 income_source 類別特徵的原生支持。

4. **SSR 分析分組呈現**：分別計算黑名單用戶和正常用戶的 SSR，在簡報中說明兩組的差異及業務含義。

### 中期（加分項，中等風險）

5. **輕量圖特徵**：利用 NetworkX 對 `crypto_transfer`（sub_kind=1 的內轉）建構用戶圖，計算以下特徵：
   - 用戶的入度（incoming_internal_degree）和出度（outgoing_internal_degree）
   - PageRank score（資金流向的中心性）
   - 與黑名單用戶的最短路徑（BFS hop count，在訓練集中可用）
   這些圖特徵加入 XGBoost 特徵向量，不需要實作完整 GNN。

6. **GraphSAGE 視覺化圖譜**：作為加分項，使用 DGL 或 PyG 的 GraphSAGE 在內轉圖上訓練，主要目的是產生視覺化（Gephi 或 vis.js），展示洗錢叢集結構。評分標準中「視覺化關聯圖譜」的加分條件可以此達成。

### 長期（原創研究貢獻）

7. **Elliptic2 啟發的子圖特徵**：識別「有互相轉帳的用戶群組」（subgraph），計算群組層級特徵（群組大小、群組黑名單比例）作為個別用戶的特徵。

8. **SHAP 不平衡可靠性的系統研究**：SSR 評測結果可整理為學術貢獻，對比不同不平衡比例下的 SSR 衰減曲線，對業界 SHAP + AML 的討論有實質貢獻。

---

## 參考資料

### 學術論文

1. Weber, M., Domeniconi, G., et al. (2019). Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics. *arXiv:1908.02591*. [連結](https://arxiv.org/pdf/1908.02591)

2. Ouyang, J., et al. (2024). Bitcoin Money Laundering Detection via Subgraph Contrastive Learning. *Entropy*, 26(3), 211. [連結](https://pmc.ncbi.nlm.nih.gov/articles/PMC10969714/)

3. Bellei, C., Xu, M., et al. (2024). The Shape of Money Laundering: Subgraph Representation Learning on the Blockchain with the Elliptic2 Dataset. *arXiv:2404.19109*. KDD MLF '24. [連結](https://arxiv.org/html/2404.19109v2)

4. Altman, E., et al. (2023). Realistic Synthetic Financial Transactions for Anti-Money Laundering Models. *NeurIPS 2023 Datasets and Benchmarks Track*. [連結](https://proceedings.neurips.cc/paper_files/paper/2023/file/5f38404edff6f3f642d6fa5892479c42-Paper-Datasets_and_Benchmarks.pdf)

5. Anonymous (2025). Explainable and fair anti-money laundering models using a reproducible SHAP framework for financial institutions. *Discover Artificial Intelligence*, Springer. [連結](https://link.springer.com/article/10.1007/s44163-026-00944-7)

6. Sui, M., Su, Y., Shen, J. (2024). Intelligent Anti-Money Laundering on Cryptocurrency: A CNN-GNN Fusion Approach. *SSRN:6006154*. [連結](https://papers.ssrn.com/sol3/Delivery.cfm/6006154.pdf?abstractid=6006154&mirid=1)

7. Anonymous (2024). Anti-Money Laundering in Cryptocurrency via Multi-Relational Graph Neural Network. *PAKDD 2023, Springer LNCS*. [連結](https://link.springer.com/chapter/10.1007/978-3-031-33377-4_10)

8. Anonymous (2024). Graph Contrastive Pre-training for Anti-money Laundering (GCPAL). *International Journal of Computational Intelligence Systems*. [連結](https://link.springer.com/article/10.1007/s44196-024-00720-4)

9. Anonymous (2022). Graph-Based LSTM for Anti-money Laundering: Experimenting Temporal Graph Convolutional Network with Bitcoin Data. *Neural Processing Letters*, Springer. [連結](https://link.springer.com/article/10.1007/s11063-022-10904-8)

10. Anonymous (2025). Deep Learning Approaches for Anti-Money Laundering on Mobile Transactions: Review, Framework, and Directions. *arXiv:2503.10058*. [連結](https://arxiv.org/html/2503.10058v1)

11. Anonymous (2024). Graph Network Models To Detect Illicit Transactions In Block Chain. *arXiv:2410.07150*. [連結](https://arxiv.org/html/2410.07150v1)

12. Anonymous (2023). Detecting anomalous cryptocurrency transactions: An AML/CFT application of machine learning-based forensics. *Electronic Markets*, Springer. [連結](https://link.springer.com/article/10.1007/s12525-023-00654-3)

13. Anonymous (2024). A plug-and-play data-driven approach for anti-money laundering in bitcoin. *Expert Systems with Applications*, ScienceDirect. [連結](https://www.sciencedirect.com/science/article/abs/pii/S0957417424029397)

14. Anonymous (2025). Normalisation and Initialisation Strategies for Graph Neural Networks in Blockchain Anomaly Detection. *arXiv:2602.23599*. [連結](https://arxiv.org/html/2602.23599)

15. Trisanto, A., et al. (2021). Modified Focal Loss in Imbalanced XGBoost for Credit Card Fraud Detection. *Semantic Scholar*. [連結](https://www.semanticscholar.org/paper/Modified-Focal-Loss-in-Imbalanced-XGBoost-for-Card-Trisanto-Jakarta/8b8eaa039d664658d98d84c87346aa6b1e16036c)

16. Anonymous (2022). A focal-aware cost-sensitive boosted tree for imbalanced credit scoring. *Expert Systems with Applications*, ScienceDirect. [連結](https://www.sciencedirect.com/science/article/abs/pii/S0957417422013379)

17. Anonymous (2024). An imbalanced learning method based on graph tran-smote for fraud detection. *Scientific Reports*. [連結](https://www.nature.com/articles/s41598-024-67550-4)

18. Aas, K., Jullum, M., & Løland, A. (2021). Explaining individual predictions when features are dependent: More accurate approximations to Shapley values. *Artificial Intelligence*, 298, 103502.

19. Anonymous (2024). Feature selection strategies: a comparative analysis of SHAP-value and importance-based methods. *Journal of Big Data*, Springer. [連結](https://link.springer.com/article/10.1186/s40537-024-00905-w)

20. Anonymous (2022). Fraud Detection in Mobile Payment Systems using an XGBoost-based Framework. *Information Systems Frontiers*, Springer. [連結](https://link.springer.com/article/10.1007/s10796-022-10346-6)

### 資料集

21. Elliptic Data Set. Kaggle. [連結](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

22. GitHub - MITIBMxGraph/Elliptic2. [連結](https://github.com/MITIBMxGraph/Elliptic2)

23. IBM AML Data. GitHub. [連結](https://github.com/IBM/AML-Data)

24. SAML-D: Anti Money Laundering Transaction Data. Kaggle. [連結](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml)

### 業界資源

25. Google Cloud. (2023). Anti Money Laundering AI. [連結](https://cloud.google.com/anti-money-laundering-ai)

26. Google Cloud. (2023). AML Input Data Model Documentation. [連結](https://docs.cloud.google.com/financial-services/anti-money-laundering/docs/reference/schemas/aml-input-data-model)

27. NVIDIA Technical Blog. (2023). Supercharging Fraud Detection in Financial Services with Graph Neural Networks. [連結](https://developer.nvidia.com/blog/supercharging-fraud-detection-in-financial-services-with-graph-neural-networks/)

28. NVIDIA Technical Blog. (2021). Leveraging Machine Learning to Detect Fraud: Tips to Developing a Winning Kaggle Solution. [連結](https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution/)

29. ComplyAdvantage. (2023). Crypto AML Red Flags. [連結](https://complyadvantage.com/insights/crypto-aml-red-flags/)

30. GitHub - safe-graph/graph-fraud-detection-papers: A curated list of Graph/Transformer-based fraud, anomaly, and outlier detection papers & resources. [連結](https://github.com/safe-graph/graph-fraud-detection-papers)

31. IBM Research. (2023). Realistic Synthetic Financial Transactions for Anti-Money Laundering Models. [連結](https://research.ibm.com/publications/realistic-synthetic-financial-transactions-for-anti-money-laundering-models)

32. Fraud Detection Handbook (Chapters on Cost-Sensitive Learning). [連結](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_6_ImbalancedLearning/CostSensitive.html)

---

## 附錄 A：術語對照表

| 英文術語 | 中文說明 |
|---------|---------|
| AML (Anti-Money Laundering) | 反洗錢 |
| CFT (Countering the Financing of Terrorism) | 反恐融資 |
| Mule Account | 人頭戶（被用於轉移非法資金的帳號） |
| Class Imbalance | 類別不平衡 |
| SMOTE | 合成少數過採樣技術 |
| Focal Loss | 焦點損失（針對難分樣本的加權損失函數） |
| GNN (Graph Neural Network) | 圖神經網路 |
| GCN (Graph Convolutional Network) | 圖卷積網路 |
| GAT (Graph Attention Network) | 圖注意力網路 |
| GraphSAGE | 可擴展圖採樣聚合網路 |
| SHAP (SHapley Additive exPlanations) | 基於 Shapley 值的加性解釋方法 |
| SSR (Stable Sample Ratio) | 穩定樣本比例（BitoGuard 自定義指標） |
| XAI (Explainable AI) | 可解釋人工智慧 |
| Illicit Class | 非法類別（洗錢/黑名單） |
| Subgraph Classification | 子圖分類 |
| Peel-chain | 比特幣鏈式剝皮洗錢模式 |
| Smurfing | 分散式小額交易（避免報告閾值的洗錢手法） |
| PageRank | 網頁排名演算法（應用於圖的節點中心性計算） |
| AUPRC | 精確率-召回率曲線下面積 |

---

*本報告由 Research Analyst 完成，調研截止日期 2026-03-14。*
*報告路徑：`docs/research/aml-model-literature-review.md`*
