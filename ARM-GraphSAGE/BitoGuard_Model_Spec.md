# BitoGuard — 模型企劃書與完整技術規格

> **專案名稱**: BitoGuard  
> **隸屬平台**: BitoEx 幣託交易所  
> **目標賽事**: AWS 詐欺偵測競賽  
> **版本**: v1.0  
> **撰寫日期**: 2026-03-16  

---

## 目錄

1. [專案概述](#1-專案概述)
2. [資料集描述](#2-資料集描述)
3. [核心模型架構：GraphSAGE-ARM-CF](#3-核心模型架構graphsage-arm-cf)
4. [模組一：特徵工程 Pipeline](#4-模組一特徵工程-pipeline)
5. [模組二：Association Rule Mining（ARM）](#5-模組二association-rule-miningarm)
6. [模組三：Causal Forest Extension（CF）](#6-模組三causal-forest-extensioncf)
7. [模組四：異質圖建構](#7-模組四異質圖建構)
8. [模組五：GNN 主模型](#8-模組五gnn-主模型)
9. [模組六：Stacking 輸出頭](#9-模組六stacking-輸出頭)
10. [模組七：可解釋性系統](#10-模組七可解釋性系統)
11. [完整訓練流程](#11-完整訓練流程)
12. [AWS 部署架構](#12-aws-部署架構)
13. [評估指標與實驗設計](#13-評估指標與實驗設計)
14. [專案時程與里程碑](#14-專案時程與里程碑)
15. [分工建議](#15-分工建議)
16. [附錄：關鍵技術選型理由](#16-附錄關鍵技術選型理由)

---

## 1. 專案概述

### 1.1 問題定義

BitoEx 幣託交易所面臨的詐欺類型主要分為兩類：

| 類型 | 描述 | 偵測難點 |
|------|------|---------|
| **詐欺帳戶（Fraud Account）** | 主動發起詐騙交易的帳戶 | 行為初期與正常用戶相似 |
| **騾子帳戶（Mule Account）** | 被利用轉移非法資金的帳戶 | 本人可能不知情，行為被動 |

傳統規則引擎依賴人工設定閾值，難以應對不斷演化的詐欺模式。本專案提出 **GraphSAGE-ARM-CF** 模型，結合圖神經網路、關聯規則挖掘與因果推斷，同時捕捉用戶的**個體行為特徵**與**社群結構信號**。

### 1.2 核心貢獻

1. **圖結構建模**：將加密貨幣轉帳關係轉化為異質圖，讓 GNN 能學習詐欺帳戶在社群中的結構特徵
2. **關聯規則注入**：透過 ARM 挖掘詐欺行為的共現模式，作為邊權重與節點特徵注入 GNN
3. **因果效應估計**：Causal Forest 計算每個用戶的條件平均處理效應（CATE），區分「相關性」與「因果性」，減少誤報
4. **可解釋性**：每個詐欺判斷都附有 ARM 規則證據、SHAP 特徵貢獻與 Anthropic API 生成的自然語言風險報告

### 1.3 模型命名

```
GraphSAGE-ARM-CF
│           │   │
│           │   └─ Causal Forest Extension
│           └───── Association Rule Mining
└─────────────────  GraphSAGE (圖神經網路骨幹)
```

---

## 2. 資料集描述

### 2.1 資料表總覽

| 資料表 | 行數 | 欄位數 | 說明 |
|--------|------|--------|------|
| `user_info` | 63,770 | 9 | 用戶基本資料、KYC 等級、帳戶來源 |
| `twd_transfer` | 195,601 | 6 | 台幣存提款紀錄（自身行為，無對手方） |
| `usdt_swap` | 53,841 | 6 | USDT 兌換紀錄 |
| `usdt_twd_trading` | 217,634 | 9 | USDT/TWD 交易紀錄 |
| `crypto_transfer` | — | — | 加密貨幣轉帳（含 from/to wallet 與 relation_user_id） |

### 2.2 欄位語意說明

#### `user_info`
```
user_id              — 主鍵
sex                  — 性別（encoded）
age                  — 年齡
career               — 職業類別（encoded）
income_source        — 資金來源（encoded）
confirmed_at         — 帳號確認時間
level1_finished_at   — KYC Level 1 完成時間
level2_finished_at   — KYC Level 2 完成時間
user_source          — 帳號來源管道
```

#### `twd_transfer`
```
user_id              — 操作用戶
kind                 — 操作類型（存/提款）
ori_samount          — 金額（scaled）
source_ip_hash       — 來源 IP 雜湊值
created_at           — 交易時間
```
> ⚠️ **重要**：`twd_transfer` 無對手方欄位，**只能作為節點特徵**，不作為圖邊

#### `crypto_transfer`（圖邊的唯一來源）
```
user_id              — 發起方用戶 ID
relation_user_id     — 對手方用戶 ID（可為 null）
from_wallet          — 發送錢包地址
to_wallet            — 接收錢包地址
amount               — 轉帳金額
created_at           — 交易時間
```

### 2.3 資料前處理原則

1. `crypto_transfer` 過濾：移除 `relation_user_id` 為 null 的記錄、金額為 0 的記錄
2. 所有金額欄位：log1p 轉換處理右偏分布
3. 時間欄位：拆解為 hour-of-day、day-of-week、days-since-registration 等衍生特徵
4. IP Hash：計算每用戶唯一 IP 數量、IP 重用率（多帳戶指標）

---

## 3. 核心模型架構：GraphSAGE-ARM-CF

### 3.1 架構總覽

```
┌─────────────────────────────────────────────────────────┐
│                      原始資料                            │
│  user_info · twd_transfer · usdt_swap · usdt_twd_trading│
│  · crypto_transfer                                       │
└──────────────────────────┬──────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
   │  模組一      │  │  模組二      │  │  模組三      │
   │ 特徵工程     │  │    ARM       │  │  Causal     │
   │ (~43 維)    │  │ 規則挖掘     │  │  Forest     │
   │             │  │ (行為模式)   │  │  (CATE)     │
   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
          └────────────────┼────────────────┘
                           ▼
                  ┌─────────────────┐
                  │   特徵融合       │
                  │  節點特徵矩陣 X  │
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │  模組四          │
                  │  異質圖建構      │
                  │  HeteroData     │
                  │  user · wallet  │
                  │  u→w, w→u, u→u │
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │  模組五          │
                  │  GNN 主模型     │
                  │  HeteroConv     │
                  │  + GATv2Conv    │
                  │  + ARM 邊權重   │
                  │  + CF 門控      │
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │  模組六          │
                  │  Stacking 輸出頭 │
                  │  LightGBM      │
                  │  meta-learner  │
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │  模組七          │
                  │  可解釋性        │
                  │  GNNExplainer   │
                  │  SHAP + ARM     │
                  │  Anthropic API  │
                  └─────────────────┘
```

### 3.2 與原版 GraphSAGE 的差異對照

| 維度 | 原版 GraphSAGE | GraphSAGE-ARM-CF |
|------|--------------|-----------------|
| 框架 | TensorFlow 1.x | PyTorch + PyG |
| 圖類型 | 同質圖 | 異質圖（HeteroConv） |
| 聚合器 | Mean / MaxPool / LSTM | GATv2Conv（attention-based） |
| 邊權重 | 無（uniform） | ARM confidence score 加權 |
| 節點特徵 | 外部給定固定特徵 | 動態融合 tabular + ARM flags + CATE |
| 輸出 | Softmax 分類 | GNN embedding → LightGBM stacking |
| 可解釋性 | 無 | CF-GNNExplainer + SHAP + 自然語言報告 |
| 因果推斷 | 無 | Causal Forest CATE 作為節點特徵與門控 |

---

## 4. 模組一：特徵工程 Pipeline

### 4.1 對應腳本

`bitoguard_feature_pipeline.py` → 輸出 `user_features.csv`

### 4.2 特徵群組

#### Group A：用戶基本資料（來自 `user_info`）
| 特徵名稱 | 說明 | 詐欺信號 |
|---------|------|---------|
| `age` | 年齡 | 極年輕 / 極年長帳戶可疑 |
| `career` | 職業類別 | 特定職業與詐欺高度相關 |
| `income_source` | 資金來源 | 來源不明確 → 高風險 |
| `kyc_days_to_l1` | 從帳號建立到完成 KYC L1 的天數 | 極快完成 → 可能批量申請 |
| `kyc_days_to_l2` | 從 L1 到 L2 的天數 | — |
| `user_source` | 帳號來源管道 | 特定管道詐欺比例高 |

#### Group B：台幣存提款行為（來自 `twd_transfer`）
| 特徵名稱 | 說明 | 詐欺信號 |
|---------|------|---------|
| `twd_tx_count` | 總交易次數 | — |
| `twd_deposit_count` | 存款次數 | — |
| `twd_withdraw_count` | 提款次數 | — |
| `twd_total_deposit` | 總存款金額（log1p） | — |
| `twd_total_withdraw` | 總提款金額（log1p） | — |
| `twd_withdraw_ratio` | 提款比例 | 近 1.0 → 快速清空 |
| `twd_night_tx_ratio` | 夜間（22:00–06:00）交易比例 | 高 → 異常 |
| `twd_unique_ip_count` | 唯一 IP 數量 | 高 → 多裝置 / 多人操作 |
| `twd_ip_reuse_rate` | IP 被多帳戶共用率 | 高 → 批量帳戶 |
| `twd_weekend_ratio` | 週末交易比例 | — |
| `twd_avg_amount` | 平均單筆金額 | — |
| `twd_amount_std` | 金額標準差 | 極低 → 固定金額操作 |

#### Group C：USDT 交易行為（來自 `usdt_twd_trading`）
| 特徵名稱 | 說明 | 詐欺信號 |
|---------|------|---------|
| `trade_count` | 交易次數 | — |
| `trade_buy_ratio` | 買入比例 | 趨近 1.0 → 只買不賣 |
| `trade_avg_amount` | 平均交易金額 | — |
| `trade_market_ratio` | 市價單比例 | 高 → 急於成交 |
| `trade_source_entropy` | 交易來源平台熵值 | 低 → 單一來源 |
| `trade_speed_score` | 交易頻率分數（次/天） | 極高 → 自動化操作 |

#### Group D：USDT 兌換行為（來自 `usdt_swap`）
| 特徵名稱 | 說明 | 詐欺信號 |
|---------|------|---------|
| `swap_count` | 兌換次數 | — |
| `swap_total_amount` | 總兌換金額（log1p） | — |
| `swap_kind_diversity` | 兌換幣種多樣性 | 低多樣性 → 專注特定幣種 |
| `swap_avg_interval_hours` | 平均兌換間隔（小時） | 極短 → 自動化 |

### 4.3 輸出格式

```csv
user_id, age, career, income_source, kyc_days_to_l1, kyc_days_to_l2, user_source,
twd_tx_count, twd_deposit_count, ...,
trade_count, trade_buy_ratio, ...,
swap_count, swap_total_amount, ...
```

總維度：**~43 維**，每行對應一個 `user_id`

---

## 5. 模組二：Association Rule Mining（ARM）

### 5.1 目標

挖掘詐欺帳戶在**行為模式上的共現規律**，轉化為：
1. 每個用戶命中的規則數量 → 節點特徵
2. 規則的 confidence score → 圖邊的 message passing 權重

### 5.2 Itemset 設計

將每個用戶的連續特徵離散化為 **布林事件**：

```python
BEHAVIORAL_ITEMS = {
    "HIGH_WITHDRAW_RATIO"  : lambda u: u.twd_withdraw_ratio > 0.85,
    "NIGHT_TRANSACTIONS"   : lambda u: u.twd_night_tx_ratio > 0.4,
    "MULTI_IP"             : lambda u: u.twd_unique_ip_count > 5,
    "IP_SHARED"            : lambda u: u.twd_ip_reuse_rate > 0.3,
    "BUY_ONLY"             : lambda u: u.trade_buy_ratio > 0.9,
    "HIGH_SPEED_TRADE"     : lambda u: u.trade_speed_score > 10,
    "MARKET_ORDER_HEAVY"   : lambda u: u.trade_market_ratio > 0.7,
    "FAST_SWAP"            : lambda u: u.swap_avg_interval_hours < 1,
    "QUICK_KYC"            : lambda u: u.kyc_days_to_l1 < 1,
    "LOW_AGE"              : lambda u: u.age < 22,
    "HIGH_AMOUNT_SINGLE"   : lambda u: u.twd_avg_amount > 500000,
    "FIXED_AMOUNT"         : lambda u: u.twd_amount_std < 100,
}
```

### 5.3 挖掘流程

```python
# 步驟 1：建立交易矩陣
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# 步驟 2：FP-Growth 挖掘頻繁項目集
frequent_itemsets = fpgrowth(
    df_items,
    min_support=0.01,     # 至少 1% 用戶命中
    use_colnames=True
)

# 步驟 3：生成關聯規則
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.6      # 置信度 >= 60%
)

# 步驟 4：保留高 lift 規則（排除假相關）
high_quality_rules = rules[rules["lift"] > 2.0]
```

### 5.4 整合到模型

#### 節點特徵層

```python
# 每個用戶新增特徵
user_features["arm_rule_hit_count"]      # 命中規則數量
user_features["arm_max_confidence"]      # 命中規則中最高 confidence
user_features["arm_avg_lift"]            # 命中規則的平均 lift
user_features["arm_fraud_score"]         # 加權詐欺風險分 = Σ(confidence × lift)
```

#### 邊權重層（ARM-guided attention bias）

```python
# 在圖建構時，若兩個 user 同時命中相同高置信規則
# 則這條邊的 attention bias 提高
def compute_arm_edge_weight(user_i, user_j, rules):
    shared_rules = get_shared_rule_hits(user_i, user_j, rules)
    if not shared_rules:
        return 1.0  # 預設權重
    max_conf = max(r.confidence for r in shared_rules)
    max_lift = max(r.lift for r in shared_rules)
    return 1.0 + max_conf * log(max_lift)
```

### 5.5 預期發現的規則範例

```
{NIGHT_TRANSACTIONS, HIGH_WITHDRAW_RATIO} → fraud
    support=0.031, confidence=0.847, lift=6.2

{MULTI_IP, BUY_ONLY, FAST_SWAP} → fraud
    support=0.018, confidence=0.791, lift=5.8

{QUICK_KYC, HIGH_AMOUNT_SINGLE, MARKET_ORDER_HEAVY} → fraud
    support=0.012, confidence=0.923, lift=9.1
```

---

## 6. 模組三：Causal Forest Extension（CF）

### 6.1 動機

傳統 ML 模型學習「詐欺帳戶的行為與哪些特徵**相關**」，但相關性不等於因果性。例如：
- 騾子帳戶可能是被**誘導**進行大額交易（受害者）
- 正常帳戶可能因為業務需求而有高頻交易

Causal Forest 能估計：**在排除個人背景因素後，某個行為改變（treatment）對詐欺風險的純增量影響（CATE）**。

### 6.2 定義 Treatment / Outcome / Covariates

```
Treatment T：帳戶近期（30天內）USDT 交易量 > 歷史均值的 2 倍
            → 代表「突然異常增加的交易活動」

Outcome Y：  詐欺標籤（若有）或 Isolation Forest 異常分數

Covariates X：所有基本特徵（age, career, income_source, 歷史行為均值等）
              → 控制個人背景差異
```

### 6.3 實作

```python
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

cf_model = CausalForestDML(
    model_y=GradientBoostingRegressor(n_estimators=100),
    model_t=GradientBoostingClassifier(n_estimators=100),
    n_estimators=200,
    min_samples_leaf=10,
    random_state=42
)

cf_model.fit(Y=y_outcome, T=t_treatment, X=X_covariates)

# 計算每個用戶的 CATE（條件平均處理效應）
cate_scores = cf_model.effect(X_covariates)
```

### 6.4 CATE 的使用方式

| 用途 | 說明 |
|------|------|
| **節點特徵** | `cate_score` 直接加入節點特徵矩陣 |
| **CF 門控聚合** | 在 GNN readout 時，高 CATE 節點的 embedding 貢獻更大 |
| **可解釋性** | 向調查人員說明「這個帳戶的交易活動增加對風險的純增量貢獻」 |

### 6.5 CATE 解讀

```
CATE > 0  →  該交易活動增加確實導致風險上升（主動詐欺可能性高）
CATE ≈ 0  →  交易活動與詐欺風險無因果關係（可能只是相關）
CATE < 0  →  交易活動增加反而降低風險（正常業務擴張）
```

---

## 7. 模組四：異質圖建構

### 7.1 對應腳本

`bitoguard_hetero_graph.py` → 輸出 `PyG HeteroData` 物件

### 7.2 圖結構定義

```
節點類型：
  - user    ：每個 user_id 為一個節點，特徵 = 融合後節點特徵矩陣
  - wallet  ：每個唯一 wallet 地址為一個節點，特徵 = one-hot or degree features

邊類型（來自 crypto_transfer）：
  - (user, sends_to, wallet)     : user_id → to_wallet
  - (wallet, receives_from, user): from_wallet → user_id（反向）
  - (user, transacts_with, user) : user_id → relation_user_id
  - (user, transacts_with_rev, user): 反向邊（雙向 message passing）

邊屬性：
  - amount（log1p）
  - arm_weight：ARM 計算的邊權重
  - time_delta：交易時間距離現在的天數
```

### 7.3 圖統計（預計）

| 統計項目 | 預期值 |
|---------|--------|
| 用戶節點數 | ~63,770 |
| 錢包節點數 | ~數萬 |
| user→wallet 邊數 | 依 crypto_transfer 記錄數 |
| user→user 邊數 | 依 relation_user_id 非 null 記錄 |

### 7.4 前處理規則

```python
# 過濾條件
crypto_df = crypto_df[crypto_df["relation_user_id"].notna()]
crypto_df = crypto_df[crypto_df["amount"] > 0]

# 可選：時間窗口過濾（取最近 N 天）
WINDOW_DAYS = 180
cutoff = crypto_df["created_at"].max() - timedelta(days=WINDOW_DAYS)
crypto_df = crypto_df[crypto_df["created_at"] >= cutoff]
```

---

## 8. 模組五：GNN 主模型

### 8.1 架構設計

```python
class BitoGuardGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=4):
        super().__init__()
        
        # Layer 1: Heterogeneous convolution with GATv2
        self.conv1 = HeteroConv({
            ("user", "sends_to", "wallet")    : GATv2Conv(in_dim, hidden_dim, heads=num_heads),
            ("wallet", "receives_from", "user"): GATv2Conv(in_dim, hidden_dim, heads=num_heads),
            ("user", "transacts_with", "user") : GATv2Conv(in_dim, hidden_dim, heads=num_heads),
        }, aggr="sum")
        
        # Layer 2
        self.conv2 = HeteroConv({
            ("user", "sends_to", "wallet")    : GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=1),
            ("wallet", "receives_from", "user"): GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=1),
            ("user", "transacts_with", "user") : GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=1),
        }, aggr="sum")
        
        # CF-gated readout：用 CATE score 作為門控係數
        self.cf_gate = nn.Linear(1, hidden_dim)
        
        # 最終 user embedding projection
        self.proj = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x_dict, edge_index_dict, edge_attr_dict, cate_scores):
        # Layer 1
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = {k: self.dropout(v) for k, v in x_dict.items()}
        
        # Layer 2
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        
        # CF-gated readout（只對 user 節點）
        user_emb = x_dict["user"]
        gate = torch.sigmoid(self.cf_gate(cate_scores.unsqueeze(-1)))
        user_emb = user_emb * gate  # element-wise gate
        
        return self.proj(user_emb)
```

### 8.2 ARM 邊權重整合

```python
# 在 GATv2Conv 的 message passing 中注入 ARM edge weight
# PyG 支援 edge_attr 傳入 GATv2Conv

class ARMWeightedGATv2Conv(GATv2Conv):
    def message(self, x_j, alpha, edge_attr):
        # alpha: attention weight from GATv2
        # edge_attr[:,0]: ARM confidence weight
        arm_weight = edge_attr[:, 0].unsqueeze(-1)
        alpha_adjusted = alpha * arm_weight
        return alpha_adjusted * x_j
```

### 8.3 Loss 函數

```python
# 詐欺偵測為高度不平衡分類問題
# 使用 Focal Loss 處理類別不平衡

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()
```

### 8.4 超參數設定

| 超參數 | 預設值 | 說明 |
|--------|--------|------|
| `hidden_dim` | 128 | 隱藏層維度 |
| `num_heads` | 4 | GATv2 attention heads |
| `out_dim` | 64 | 輸出 embedding 維度 |
| `dropout` | 0.3 | Dropout rate |
| `lr` | 0.001 | Adam 學習率 |
| `weight_decay` | 1e-4 | L2 正則化 |
| `epochs` | 100 | 最大訓練輪數 |
| `batch_size` | 1024 | Mini-batch（NeighborLoader） |
| `num_neighbors` | [10, 5] | 兩層各採樣鄰居數（GraphSAGE 風格） |

---

## 9. 模組六：Stacking 輸出頭

### 9.1 設計原理

GNN embedding 捕捉圖結構信號，但可能遺漏部分個體特徵。透過 **Stacking**，讓 LightGBM 學習如何融合兩種互補的信號。

```
輸入 = concat(
    GNN 64維 embedding,     # 圖結構信號
    tabular 43維特徵,        # 個體行為信號
    ARM 4維特徵,             # 規則命中信號
    CATE 1維分數             # 因果效應信號
)
總輸入維度 = 64 + 43 + 4 + 1 = 112 維
```

### 9.2 LightGBM 設定

```python
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=-1,
    scale_pos_weight=neg_count / pos_count,  # 處理類別不平衡
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42
)
```

### 9.3 訓練策略

```
第一階段：訓練 GNN（80% 訓練集）
第二階段：用 GNN 對剩餘 20% 生成 out-of-fold embedding
第三階段：用 out-of-fold embedding + 原始特徵訓練 LightGBM
第四階段：最終預測 = LightGBM(GNN_embedding + 原始特徵)
```

---

## 10. 模組七：可解釋性系統

### 10.1 三層可解釋架構

```
層次 1 [全局]：哪些特徵對詐欺偵測最重要？
  → SHAP global importance（LightGBM）
  → ARM 規則的 support/confidence/lift 排行

層次 2 [個案]：這個帳戶為什麼被標記為詐欺？
  → SHAP waterfall plot（個別用戶）
  → 命中的 ARM 規則清單
  → CF-GNNExplainer 找出關鍵鄰居

層次 3 [自然語言]：生成可讀報告
  → Anthropic API → 自然語言風險診斷報告
```

### 10.2 CF-GNNExplainer 整合

```python
from torch_geometric.explain import Explainer, GNNExplainer

explainer = Explainer(
    model=bitoguard_gnn,
    algorithm=GNNExplainer(epochs=200),
    explanation_type="model",
    node_mask_type="attributes",
    edge_mask_type="object",
    model_config=dict(
        mode="binary_classification",
        task_level="node",
        return_type="probs"
    )
)

# 為高風險帳戶生成解釋
explanation = explainer(
    x=data.x_dict,
    edge_index=data.edge_index_dict,
    index=high_risk_node_idx
)
```

### 10.3 Anthropic API 風險報告生成

```python
import anthropic

def generate_risk_report(user_id, fraud_score, shap_values, arm_rules, cate_score):
    client = anthropic.Anthropic()
    
    prompt = f"""
    你是一位加密貨幣詐欺分析師。請根據以下資訊，撰寫一份針對用戶 {user_id} 的風險診斷報告：
    
    詐欺風險分數：{fraud_score:.3f}（0=低風險，1=高風險）
    
    CATE 因果效應分數：{cate_score:.3f}
    （>0 表示近期交易量增加對風險有正向因果影響）
    
    前三大風險特徵（SHAP）：
    {format_shap(shap_values)}
    
    命中的關聯規則：
    {format_arm_rules(arm_rules)}
    
    請用繁體中文撰寫 200 字以內的報告，說明：
    1. 主要風險指標
    2. 可能的詐欺類型（主動詐欺 vs 騾子帳戶）
    3. 建議的後續調查方向
    """
    
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return message.content[0].text
```

---

## 11. 完整訓練流程

### 11.1 流程圖

```
Step 1: 資料前處理
  ├─ 讀取所有 CSV
  ├─ 清洗 crypto_transfer（過濾 null relation_user_id、零金額）
  └─ 輸出清洗後資料

Step 2: 特徵工程 [bitoguard_feature_pipeline.py]
  └─ 輸出 user_features.csv（~43 維）

Step 3: ARM 挖掘 [arm_mining.py]
  ├─ 離散化特徵為 behavioral items
  ├─ FP-Growth 挖掘頻繁項目集
  ├─ 生成關聯規則（min_confidence=0.6, min_lift=2.0）
  ├─ 計算每用戶 ARM 特徵（hit_count, max_conf, avg_lift, fraud_score）
  └─ 計算邊的 ARM weight

Step 4: Causal Forest 訓練 [causal_forest.py]
  ├─ 定義 Treatment（USDT 交易量突增）
  ├─ 訓練 CausalForestDML
  └─ 輸出每用戶 CATE 分數

Step 5: 特徵融合 [feature_fusion.py]
  └─ concat(tabular, arm_features, cate_scores) → node_features.csv

Step 6: 異質圖建構 [bitoguard_hetero_graph.py]
  ├─ 建立 user / wallet 節點
  ├─ 建立 u→w, w→u, u→u 邊
  ├─ 加入 ARM edge weights
  └─ 輸出 graph.pt

Step 7: GNN 訓練 [bitoguard_gnn_model.py]
  ├─ 載入 graph.pt + node_features.csv
  ├─ 訓練 BitoGuardGNN（Focal Loss, Adam）
  ├─ 驗證（PR-AUC, ROC-AUC）
  └─ 輸出 gnn_embeddings.npy

Step 8: Stacking [stacking_head.py]
  ├─ 合併 gnn_embeddings + 原始特徵
  ├─ 訓練 LightGBM meta-learner
  └─ 輸出最終 fraud_scores.csv

Step 9: 可解釋性 [explainability.py]
  ├─ SHAP 分析
  ├─ CF-GNNExplainer 找關鍵鄰居
  └─ 為高風險帳戶生成 Anthropic API 報告

Step 10: 評估與報告
  └─ 計算 PR-AUC, ROC-AUC, F1, Precision@K
```

### 11.2 執行指令

```bash
# 建議執行順序
python bitoguard_feature_pipeline.py \
    --data_dir ./data \
    --output user_features.csv

python arm_mining.py \
    --features user_features.csv \
    --min_support 0.01 \
    --min_confidence 0.6 \
    --output arm_rules.pkl arm_features.csv

python causal_forest.py \
    --features user_features.csv \
    --output cate_scores.csv

python feature_fusion.py \
    --tabular user_features.csv \
    --arm arm_features.csv \
    --cate cate_scores.csv \
    --output node_features.csv

python bitoguard_hetero_graph.py \
    --node_features node_features.csv \
    --arm_rules arm_rules.pkl \
    --crypto_transfer crypto_transfer.csv \
    --output graph.pt

python bitoguard_gnn_model.py \
    --graph graph.pt \
    --epochs 100 \
    --hidden_dim 128 \
    --output gnn_embeddings.npy

python stacking_head.py \
    --embeddings gnn_embeddings.npy \
    --features node_features.csv \
    --output fraud_scores.csv

python explainability.py \
    --model bitoguard_gnn.pt \
    --scores fraud_scores.csv \
    --top_k 100
```

---

## 12. AWS 部署架構

### 12.1 訓練環境

```
Amazon SageMaker Training Job
  ├─ Instance type: ml.g4dn.xlarge（T4 GPU，適合 GNN 訓練）
  ├─ Framework: PyTorch 2.x + PyG
  └─ 資料儲存: Amazon S3

Amazon SageMaker Notebook
  └─ 用於 ARM 挖掘 + Causal Forest（CPU-heavy）
```

### 12.2 推理環境

```
Amazon SageMaker Endpoint
  ├─ 即時推理（Real-time inference）
  ├─ Instance type: ml.m5.xlarge
  └─ Model artifact: 打包 GNN + LightGBM + Scaler

Amazon API Gateway + AWS Lambda
  └─ REST API → 觸發推理 + 生成 Anthropic API 報告
```

### 12.3 展示 Dashboard

```
Amazon CloudWatch
  └─ 模型監控（data drift, fraud rate）

AWS Amplify + React
  └─ Demo Dashboard
     ├─ 用戶風險分布圖
     ├─ 詐欺帳戶網路圖（D3.js）
     ├─ 個案解釋面板（SHAP + ARM rules）
     └─ Anthropic API 生成報告
```

---

## 13. 評估指標與實驗設計

### 13.1 主要評估指標

| 指標 | 說明 | 目標值 |
|------|------|--------|
| **PR-AUC** | Precision-Recall 曲線面積（主指標，適合不平衡資料） | > 0.85 |
| **ROC-AUC** | ROC 曲線面積 | > 0.92 |
| **Precision@100** | 前 100 高風險帳戶中詐欺比例 | > 0.80 |
| **Recall@5%FPR** | 假陽性率 5% 時的召回率 | > 0.75 |

### 13.2 消融實驗（Ablation Study）

| 模型配置 | 說明 |
|---------|------|
| Baseline: LightGBM only | 僅用 tabular 特徵 |
| + GNN | 加入圖結構 embedding |
| + ARM features | 加入 ARM 節點特徵 |
| + ARM edge weights | 加入 ARM 邊權重（message passing） |
| + Causal Forest | 加入 CATE 分數 |
| **Full model** | GraphSAGE-ARM-CF 完整版 |

### 13.3 資料分割策略

```
時間序列分割（Time-based split，避免 data leakage）：
  訓練集：前 70% 時間的交易
  驗證集：中間 15%（超參數調整）
  測試集：最後 15%（最終評估）
```

---

## 14. 專案時程與里程碑

### 14.1 整體時程（2 週衝刺）

```
Week 1
├─ Day 1-2：資料 EDA + 前處理 + 特徵工程完成
├─ Day 3：ARM mining 模組完成 + 規則驗證
├─ Day 4：Causal Forest 模組完成 + CATE 分析
└─ Day 5：異質圖建構 + 初版 GNN 訓練

Week 2
├─ Day 6-7：GNN 調參 + Stacking 頭完成
├─ Day 8：可解釋性系統 + Anthropic API 整合
├─ Day 9：AWS 部署 + SageMaker 訓練任務
└─ Day 10：Demo Dashboard + 簡報準備
```

### 14.2 里程碑定義

| 里程碑 | 完成標準 |
|--------|---------|
| M1：資料就緒 | 所有 CSV 清洗完成，`user_features.csv` 生成 |
| M2：ARM 完成 | 至少找到 20 條高質量規則（conf>0.6, lift>2） |
| M3：CF 完成 | CATE 分數分布符合預期（非退化解） |
| M4：GNN 收斂 | 驗證集 ROC-AUC > 0.85 |
| M5：Stacking 完成 | 完整 pipeline 端對端可執行 |
| M6：部署完成 | SageMaker Endpoint 可用，API 回應 < 500ms |
| M7：Demo 完成 | Dashboard 展示 + 簡報完成 |

---

## 15. 分工建議

| 成員背景 | 負責模組 |
|---------|---------|
| ML / 特徵工程 | 模組一（特徵工程）+ 模組六（Stacking） |
| ML / 統計背景 | 模組二（ARM）+ 模組三（Causal Forest） |
| Deep Learning | 模組五（GNN 主模型）+ 模組七（GNNExplainer） |
| Software Engineering | 模組四（圖建構）+ 模組十二（AWS 部署）+ Dashboard |
| 全組共同 | 模組十三（評估）+ 簡報 |

---

## 16. 附錄：關鍵技術選型理由

### 為什麼選 GATv2 而非原版 GraphSAGE？

原版 GraphSAGE 的 Mean Aggregator 對所有鄰居給予相同權重，但在詐欺圖中：
- 一個騾子帳戶可能連接到數百個正常帳戶和少數詐欺帳戶
- Attention 機制能讓 GNN「專注」於詐欺鄰居的信號
- GATv2 修復了原版 GAT 的「靜態注意力」問題，更適合動態圖

### 為什麼選 FP-Growth 而非 Apriori？

- `crypto_transfer` 資料量大，Apriori 的候選項目集生成開銷過高
- FP-Growth 不生成候選集，時間複雜度更低
- `mlxtend` 提供 Pandas 友好的實作

### 為什麼用 Causal Forest 而非普通特徵？

詐欺偵測中的「混淆因素（confounding）」問題嚴重：
- 高資產用戶本來就交易頻繁，不能直接用「交易量高」作為詐欺信號
- Causal Forest 在控制個人背景（age, career, income_source）後，估計純粹由行為改變帶來的風險增量
- 這讓模型對「正常高交易用戶」的誤報率大幅降低

### 為什麼使用 Stacking 而非直接用 GNN 輸出？

- GNN 擅長捕捉圖結構信號，但對純 tabular 特徵的建模能力不如梯度提升
- LightGBM 能自動處理特徵交互，且對缺失值魯棒
- Stacking 是「取兩者之長」的標準做法，在學術界和工業界均已驗證有效

---

*文件版本：1.0 | 最後更新：2026-03-16 | 專案：BitoGuard @ AWS Competition*
