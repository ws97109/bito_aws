# BitoGuard 系統架構詳解

本文檔提供 BitoGuard GraphSAGE-ARM-CF 系統的完整技術架構說明。

---

## 系統架構圖

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         BitoGuard System Architecture                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐           ┌───────────────┐         ┌───────────────┐
│  Raw Data     │           │  Raw Data     │         │  Raw Data     │
│  Layer        │           │  Layer        │         │  Layer        │
│               │           │               │         │               │
│ • user_info   │           │ • twd_transfer│         │• crypto_      │
│ • usdt_swap   │           │ • usdt_trading│         │  transfer     │
└───────┬───────┘           └───────┬───────┘         └───────┬───────┘
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────────────────────┐
        │              Feature Engineering Layer                     │
        │  (bitoguard_feature_pipeline.py)                          │
        │                                                            │
        │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
        │  │ Group A  │  │ Group B  │  │ Group C  │  │ Group D  │ │
        │  │  Basic   │  │   TWD    │  │  Trading │  │   Swap   │ │
        │  │ Features │  │ Transfer │  │ Features │  │ Features │ │
        │  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │
        └───────────────────────────┬───────────────────────────────┘
                                    │
                        user_features.csv (~43 dims)
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌──────────────┐          ┌──────────────┐          ┌──────────────┐
│ ARM Module   │          │ CF Module    │          │ Feature      │
│ (Module 2)   │          │ (Module 3)   │          │ Fusion       │
│              │          │              │          │              │
│ FP-Growth    │          │ Causal       │          │ Merge all    │
│ Association  │          │ Forest DML   │          │ features     │
│ Rules        │          │              │          │              │
└──────┬───────┘          └──────┬───────┘          └──────┬───────┘
       │                         │                         │
       │ arm_features.csv        │ cate_scores.csv        │
       │ arm_rules.pkl           │                         │
       └─────────────────────────┼─────────────────────────┘
                                 │
                    node_features.csv (~48 dims)
                                 │
                                 ▼
        ┌────────────────────────────────────────────────────┐
        │      Heterogeneous Graph Construction              │
        │      (bitoguard_hetero_graph.py)                   │
        │                                                     │
        │  Nodes:  [user]  [wallet]                         │
        │                                                     │
        │  Edges:  user ──sends_to──> wallet                │
        │          wallet ──receives_from──> user           │
        │          user ──transacts_with──> user            │
        │                                                     │
        │  Edge Attrs: amount, ARM weights                   │
        └────────────────────────┬───────────────────────────┘
                                 │
                            graph.pt
                                 │
                                 ▼
        ┌────────────────────────────────────────────────────┐
        │         BitoGuardGNN Model (Module 5)              │
        │      (models/bitoguard_gnn.py)                     │
        │                                                     │
        │  ┌─────────────────────────────────────┐          │
        │  │   Input Projection Layer            │          │
        │  │   user: X_u → H, wallet: X_w → H   │          │
        │  └────────────────┬────────────────────┘          │
        │                   │                                 │
        │  ┌────────────────▼────────────────────┐          │
        │  │   HeteroConv Layer 1                │          │
        │  │   GATv2Conv (multi-head attention)  │          │
        │  │   + ARM edge weights                │          │
        │  └────────────────┬────────────────────┘          │
        │                   │                                 │
        │  ┌────────────────▼────────────────────┐          │
        │  │   HeteroConv Layer 2                │          │
        │  │   GATv2Conv (single-head)           │          │
        │  └────────────────┬────────────────────┘          │
        │                   │                                 │
        │  ┌────────────────▼────────────────────┐          │
        │  │   CF-Gated Readout                  │          │
        │  │   gate = σ(W_cf · CATE)            │          │
        │  │   h_user = h_user ⊙ gate           │          │
        │  └────────────────┬────────────────────┘          │
        │                   │                                 │
        │  ┌────────────────▼────────────────────┐          │
        │  │   Output Projection                 │          │
        │  │   h_user → z_user (64-dim)         │          │
        │  └─────────────────────────────────────┘          │
        └────────────────────────┬───────────────────────────┘
                                 │
                    user_embeddings (64-dim)
                                 │
                                 ▼
        ┌────────────────────────────────────────────────────┐
        │         Classification Head                         │
        │         (Focal Loss for imbalanced data)           │
        │                                                     │
        │  Logits = Linear(z_user)                           │
        │  Loss = -α(1-p_t)^γ log(p_t)                      │
        └────────────────────────┬───────────────────────────┘
                                 │
                                 ▼
                      Fraud Probability Scores
```

---

## 模組依賴關係

```
Module 1 (Feature Eng)
    │
    ├─> Module 2 (ARM Mining)
    │        │
    │        └─> arm_features.csv, arm_rules.pkl
    │
    ├─> Module 3 (Causal Forest)
    │        │
    │        └─> cate_scores.csv
    │
    └─> Feature Fusion
             │
             └─> node_features.csv
                     │
                     ▼
            Module 4 (Graph Construction)
                     │
                     └─> graph.pt
                            │
                            ▼
                   Module 5 (BitoGuardGNN)
                            │
                            └─> embeddings + predictions
```

---

## 資料流詳解

### 1. 原始資料層 (Raw Data Layer)

| 資料表 | 行數 | 關鍵欄位 | 用途 |
|--------|------|---------|------|
| `user_info` | 63,770 | user_id, age, career, KYC timing | 用戶基本特徵 |
| `twd_transfer` | 195,601 | user_id, kind, amount, IP | TWD 行為特徵 |
| `usdt_twd_trading` | 217,634 | user_id, side, order_type | 交易行為特徵 |
| `usdt_swap` | 53,841 | user_id, kind, amount | 兌換行為特徵 |
| `crypto_transfer` | — | user_id, relation_user_id, wallets | **圖邊來源** |

### 2. 特徵工程層 (Feature Engineering Layer)

**輸入**: 5 個原始 CSV 檔案

**處理步驟**:
1. 載入資料
2. 時間特徵提取 (hour, day_of_week, is_night, is_weekend)
3. 金額 log1p 轉換
4. IP 特徵計算 (unique count, reuse rate)
5. 聚合統計 (count, mean, std, ratio)

**輸出**: `user_features.csv` (63,770 × 43)

**程式碼位置**: `bitoguard_feature_pipeline.py`

### 3. ARM 層 (Association Rule Mining Layer)

**輸入**: `user_features.csv`

**處理步驟**:
1. **離散化**: 連續特徵 → 布林事件
   ```python
   HIGH_WITHDRAW_RATIO = (twd_withdraw_ratio > 0.85)
   NIGHT_TRANSACTIONS  = (twd_night_tx_ratio > 0.4)
   MULTI_IP            = (twd_unique_ip_count > 5)
   ...共 12 個 items
   ```

2. **Transaction 建構**:
   ```python
   User A: [HIGH_WITHDRAW_RATIO, NIGHT_TRANSACTIONS, MULTI_IP]
   User B: [BUY_ONLY, QUICK_KYC]
   ...
   ```

3. **FP-Growth 挖掘**:
   ```
   {NIGHT_TRANSACTIONS, HIGH_WITHDRAW_RATIO} → fraud
     support=0.031, confidence=0.847, lift=6.2
   ```

4. **特徵計算**:
   - `arm_rule_hit_count`: 用戶命中的規則數量
   - `arm_max_confidence`: 最高置信度
   - `arm_avg_lift`: 平均 lift
   - `arm_fraud_score`: 加權風險分數

**輸出**:
- `arm_rules.pkl`: DataFrame of rules
- `arm_features.csv` (63,770 × 4)

**程式碼位置**: `arm_mining.py`

### 4. 因果推斷層 (Causal Forest Layer)

**輸入**: `user_features.csv`

**因果模型定義**:
```
Treatment T: 近期 USDT 交易量 > 2 × 歷史均值
Outcome Y:   詐欺標籤 (或異常分數)
Covariates X: age, career, income_source, 歷史行為均值

CATE(X) = E[Y | T=1, X] - E[Y | T=0, X]
```

**估計方法**: CausalForestDML (econml)
- Model Y: GradientBoostingRegressor
- Model T: GradientBoostingClassifier
- Forest: 200 estimators

**輸出**: `cate_scores.csv` (63,770 × 1)

**程式碼位置**: `causal_forest.py`

### 5. 特徵融合層 (Feature Fusion Layer)

**輸入**:
- `user_features.csv` (43 dims)
- `arm_features.csv` (4 dims)
- `cate_scores.csv` (1 dim)

**操作**: Left join on `user_id`

**輸出**: `node_features.csv` (63,770 × 48)

**程式碼位置**: `feature_fusion.py`

### 6. 異質圖建構層 (Heterogeneous Graph Layer)

**輸入**:
- `node_features.csv`: 節點特徵
- `crypto_transfer.csv`: 圖邊資料
- `arm_rules.pkl`: (Optional) 計算邊權重

**圖結構**:

```python
data = HeteroData()

# User nodes
data['user'].x = user_features  # [num_users, 48]
data['user'].num_nodes = 63770

# Wallet nodes
data['wallet'].x = wallet_features  # [num_wallets, 100]
data['wallet'].num_nodes = num_wallets

# Edges
data['user', 'sends_to', 'wallet'].edge_index = ...
data['user', 'sends_to', 'wallet'].edge_attr = [amount, arm_weight]

data['wallet', 'receives_from', 'user'].edge_index = ...
data['wallet', 'receives_from', 'user'].edge_attr = [amount, arm_weight]

data['user', 'transacts_with', 'user'].edge_index = ...
data['user', 'transacts_with', 'user'].edge_attr = [amount, arm_weight]
```

**邊權重計算** (如果有 ARM rules):
```python
def compute_arm_edge_weight(user_i, user_j, rules):
    shared_rules = get_shared_rules(user_i, user_j, rules)
    if shared_rules:
        max_conf = max(r.confidence for r in shared_rules)
        max_lift = max(r.lift for r in shared_rules)
        return 1.0 + max_conf * log(max_lift)
    return 1.0
```

**輸出**: `graph.pt` (PyG HeteroData object)

**程式碼位置**: `bitoguard_hetero_graph.py`

### 7. GNN 模型層 (BitoGuardGNN Layer)

**輸入**: `graph.pt`

**模型結構**:

```python
class BitoGuardGNN(nn.Module):
    def __init__(self, ...):
        # Input projection
        self.input_projs = {
            'user': Linear(48, 128),
            'wallet': Linear(100, 128)
        }

        # Layer 1: Multi-head GATv2
        self.conv1 = HeteroConv({
            ('user', 'sends_to', 'wallet'): GATv2Conv(
                128, 128, heads=4, edge_dim=1
            ),
            ('wallet', 'receives_from', 'user'): GATv2Conv(...),
            ('user', 'transacts_with', 'user'): GATv2Conv(...)
        })

        # Layer 2: Single-head GATv2
        self.conv2 = HeteroConv({
            ('user', 'sends_to', 'wallet'): GATv2Conv(
                512, 128, heads=1, edge_dim=1
            ),
            ...
        })

        # CF gating
        self.cf_gate = Linear(1, 128)

        # Output
        self.output_proj = Linear(128, 64)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, cate_scores):
        # Project inputs
        x_dict = {k: self.input_projs[k](x) for k, x in x_dict.items()}

        # Layer 1
        x_dict = self.conv1(x_dict, edge_index_dict, edge_attr_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        # Layer 2
        x_dict = self.conv2(x_dict, edge_index_dict, edge_attr_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        # CF-gated readout
        user_emb = x_dict['user']
        gate = torch.sigmoid(self.cf_gate(cate_scores))
        user_emb = user_emb * gate

        # Output projection
        return self.output_proj(user_emb)
```

**創新點**:
1. **HeteroConv**: 不同邊類型使用不同的 GATv2Conv
2. **ARM edge weights**: 通過 `edge_attr` 注入 attention
3. **CF gating**: CATE 分數調節 embedding 貢獻

**輸出**: User embeddings [num_users, 64]

**程式碼位置**: `models/bitoguard_gnn.py`

---

## 訓練流程

```
┌──────────────────────────────────────────┐
│  1. Load HeteroData                      │
│     hetero_data = torch.load(graph.pt)  │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│  2. Create NeighborLoader                │
│     - Sampling: [10, 5] (2-hop)         │
│     - Batch size: 1024                   │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│  3. Initialize Model                     │
│     - BitoGuardGNN                       │
│     - FocalLoss (α=0.25, γ=2.0)         │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│  4. Training Loop                        │
│     for epoch in range(epochs):         │
│       - Forward pass                     │
│       - Compute focal loss               │
│       - Backward + optimizer step        │
│       - Validation                       │
│       - Early stopping check             │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│  5. Save Best Model                      │
│     torch.save(model, best_model.pt)    │
└──────────────────────────────────────────┘
```

**程式碼位置**: `train_example.py`

---

## 目錄結構

```
bitoguard/
│
├── configs/
│   └── config.yaml                    # 主配置文件
│
├── models/
│   ├── __init__.py
│   └── bitoguard_gnn.py               # GNN 模型定義
│       ├── class ARMWeightedGATv2Conv
│       ├── class BitoGuardGNN
│       ├── class FocalLoss
│       └── class BitoGuardClassifier
│
├── utils/
│   ├── __init__.py
│   └── utils.py                       # 工具函數
│       ├── load_config()
│       ├── set_seed()
│       ├── compute_metrics()
│       ├── log1p_transform()
│       ├── extract_time_features()
│       ├── calculate_ip_features()
│       ├── entropy()
│       └── class EarlyStopping
│
├── data/                              # (運行時生成)
│   ├── user_info.csv
│   ├── twd_transfer.csv
│   ├── usdt_swap.csv
│   ├── usdt_twd_trading.csv
│   └── crypto_transfer.csv
│
├── results/                           # (運行時生成)
│   ├── features/
│   │   ├── user_features.csv
│   │   ├── arm_features.csv
│   │   ├── arm_rules.pkl
│   │   ├── cate_scores.csv
│   │   ├── node_features.csv
│   │   └── gnn_embeddings.npy
│   ├── graphs/
│   │   └── graph.pt
│   ├── models/
│   │   └── best_model.pt
│   └── reports/
│
├── bitoguard_feature_pipeline.py     # Module 1
├── arm_mining.py                      # Module 2
├── causal_forest.py                   # Module 3
├── feature_fusion.py                  # Feature merge
├── bitoguard_hetero_graph.py          # Module 4
├── train_example.py                   # Training script
│
├── run_pipeline.sh                    # 一鍵執行腳本
├── requirements.txt                   # Python 依賴
│
├── README.md                          # 主文檔
├── COMPARISON_WITH_BASELINE.md        # vs Baseline 對比
├── QUICKSTART.md                      # 快速開始
├── ARCHITECTURE.md                    # 本文檔
│
└── __init__.py
```

---

## 計算流程時序圖

```
User --> bitoguard_feature_pipeline.py
           │
           └──> user_features.csv
                     │
                     ├──> arm_mining.py
                     │      │
                     │      └──> arm_features.csv, arm_rules.pkl
                     │
                     ├──> causal_forest.py
                     │      │
                     │      └──> cate_scores.csv
                     │
                     └──> feature_fusion.py
                              │
                              └──> node_features.csv
                                       │
                                       └──> bitoguard_hetero_graph.py
                                              │
                                              └──> graph.pt
                                                     │
                                                     └──> train_example.py
                                                            │
                                                            ├──> best_model.pt
                                                            └──> gnn_embeddings.npy
```

---

## 關鍵技術細節

### GATv2Conv Attention 機制

```python
# GATv2 的 attention 計算
α_ij = softmax_j(LeakyReLU(a^T [W·h_i || W·h_j || e_ij]))

# 其中:
# h_i, h_j: 節點 i, j 的特徵
# e_ij: 邊特徵 (ARM weight)
# W: 可學習權重矩陣
# a: 注意力向量
# ||: 拼接操作

# Message passing
h_i^(l+1) = σ(Σ_j α_ij · W·h_j)
```

### ARM Edge Weight 注入

```python
# 在 forward pass 時
edge_attr = torch.tensor([
    [amount, arm_weight],  # 每條邊的屬性
    ...
])

# GATv2Conv 內部使用 edge_attr
out = GATv2Conv(x, edge_index, edge_attr)
```

### CF Gating 機制

```python
# CATE scores: [num_users, 1]
gate = torch.sigmoid(W_cf · cate_scores)  # [num_users, hidden_dim]

# Element-wise gating
h_user_gated = h_user ⊙ gate  # [num_users, hidden_dim]
```

---

## 性能優化建議

### 記憶體優化
- 使用 `NeighborLoader` 進行 mini-batch 訓練
- 降低 `num_neighbors` 採樣數量
- 使用混合精度訓練 (`torch.cuda.amp`)

### 速度優化
- 使用 GPU (CUDA)
- 預計算並快取 ARM edge weights
- 使用 DataLoader 的 `num_workers` 並行載入

### 擴展性優化
- 分散式訓練 (PyTorch DDP)
- 圖分區 (Graph partitioning)
- Sparse tensor 操作

---

**文檔版本**: 1.0
**最後更新**: 2026-03-17
**作者**: BitoGuard Team
