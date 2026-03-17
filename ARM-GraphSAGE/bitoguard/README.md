# BitoGuard: GraphSAGE-ARM-CF Fraud Detection System

> **Version**: 1.0
> **Project**: BitoEx 幣託交易所詐欺偵測系統
> **Competition**: AWS 詐欺偵測競賽

---

## 專案概述

BitoGuard 是一個結合圖神經網路(GraphSAGE)、關聯規則挖掘(ARM)和因果推斷(Causal Forest)的詐欺偵測系統。相較於傳統的 GraphSAGE baseline，BitoGuard 提供了以下創新：

### 核心創新

| 維度 | Baseline GraphSAGE | BitoGuard (GraphSAGE-ARM-CF) |
|------|-------------------|---------------------------|
| 框架 | TensorFlow 1.x | **PyTorch + PyG** |
| 圖類型 | 同質圖 | **異質圖 (user + wallet 節點)** |
| 聚合器 | Mean/MaxPool | **GATv2Conv (attention-based)** |
| 邊權重 | Uniform | **ARM confidence scores** |
| 節點特徵 | 靜態 tabular | **Tabular + ARM + CATE 動態融合** |
| 因果推斷 | 無 | **Causal Forest CATE** |
| 可解釋性 | 無 | **GNNExplainer + SHAP + 自然語言報告** |

---

## 系統架構

```
┌─────────────────────────────────────────────────────────┐
│                    BitoGuard Pipeline                    │
└─────────────────────┬───────────────────────────────────┘
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
   ┌──────────┐ ┌──────────┐ ┌──────────┐
   │ Module 1 │ │ Module 2 │ │ Module 3 │
   │ 特徵工程  │ │   ARM    │ │ Causal   │
   │  (~43維) │ │ 規則挖掘  │ │  Forest  │
   └────┬─────┘ └────┬─────┘ └────┬─────┘
        └────────────┼────────────┘
                     ▼
            ┌────────────────┐
            │  特徵融合       │
            │ node_features  │
            └────────┬───────┘
                     │
            ┌────────▼────────┐
            │   Module 4      │
            │  異質圖建構      │
            │  HeteroData     │
            └────────┬────────┘
                     │
            ┌────────▼────────┐
            │   Module 5      │
            │  BitoGuardGNN   │
            │  (GATv2 + ARM)  │
            └────────┬────────┘
                     │
            ┌────────▼────────┐
            │  Fraud Scores   │
            │  & Embeddings   │
            └─────────────────┘
```

---

## 安裝與環境設定

### 系統需求

- Python >= 3.8
- CUDA 11.x (GPU 訓練，optional)

### 安裝步驟

```bash
# 1. 建立虛擬環境
conda create -n bitoguard python=3.9
conda activate bitoguard

# 2. 安裝 PyTorch (根據你的 CUDA 版本)
# CPU only:
pip install torch torchvision torchaudio

# GPU (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 安裝 PyTorch Geometric
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# 4. 安裝其他依賴
pip install pandas numpy scikit-learn pyyaml
pip install mlxtend  # For ARM (FP-Growth)
pip install econml   # For Causal Forest
pip install lightgbm # For stacking
pip install shap     # For explainability (optional)
```

---

## 快速開始

### 1. 準備資料

將資料放置在 `Data/` 目錄下：
```
Data/
├── user_info.csv
├── twd_transfer.csv
├── usdt_swap.csv
├── usdt_twd_trading.csv
└── crypto_transfer.csv
```

### 2. 執行完整 Pipeline

```bash
# 進入 BitoGuard 目錄
cd ARM-GraphSAGE/bitoguard

# Step 1: 特徵工程
python bitoguard_feature_pipeline.py \
    --config configs/config.yaml \
    --output results/features/user_features.csv

# Step 2: ARM 規則挖掘
python arm_mining.py \
    --config configs/config.yaml \
    --features results/features/user_features.csv \
    --output_rules results/features/arm_rules.pkl \
    --output_features results/features/arm_features.csv

# Step 3: Causal Forest
python causal_forest.py \
    --config configs/config.yaml \
    --features results/features/user_features.csv \
    --output results/features/cate_scores.csv

# Step 4: 特徵融合
python feature_fusion.py \
    --config configs/config.yaml \
    --tabular results/features/user_features.csv \
    --arm results/features/arm_features.csv \
    --cate results/features/cate_scores.csv \
    --output results/features/node_features.csv

# Step 5: 異質圖建構
python bitoguard_hetero_graph.py \
    --config configs/config.yaml \
    --node_features results/features/node_features.csv \
    --arm_rules results/features/arm_rules.pkl \
    --output results/graphs/graph.pt
```

### 3. 訓練 GNN 模型

```python
import torch
from torch_geometric.loader import NeighborLoader
from models.bitoguard_gnn import BitoGuardClassifier
from utils.utils import load_config, set_seed

# 載入配置
config = load_config('configs/config.yaml')
set_seed(config['seed'])

# 載入圖資料
hetero_data = torch.load('results/graphs/graph.pt')

# 初始化模型
node_feature_dims = {
    'user': hetero_data['user'].x.shape[1],
    'wallet': hetero_data['wallet'].x.shape[1]
}

model = BitoGuardClassifier(
    node_feature_dims=node_feature_dims,
    hidden_dim=config['gnn']['hidden_dim'],
    out_dim=config['gnn']['out_dim'],
    num_heads=config['gnn']['num_heads'],
    dropout=config['gnn']['dropout']
)

# 訓練 (詳見 train_bitoguard.py)
```

---

## 模組詳解

### Module 1: 特徵工程 Pipeline

**檔案**: `bitoguard_feature_pipeline.py`

**功能**: 從原始資料表提取 ~43 維特徵

**特徵群組**:
- Group A: 用戶基本資料 (age, career, KYC timing)
- Group B: TWD 存提款行為 (13 features)
- Group C: USDT 交易行為 (6 features)
- Group D: USDT 兌換行為 (4 features)

**輸出**: `user_features.csv`

---

### Module 2: ARM 規則挖掘

**檔案**: `arm_mining.py`

**功能**: 使用 FP-Growth 挖掘詐欺行為模式

**Behavioral Items**:
```python
- HIGH_WITHDRAW_RATIO  (提款比例 > 85%)
- NIGHT_TRANSACTIONS   (夜間交易 > 40%)
- MULTI_IP             (唯一IP > 5)
- BUY_ONLY             (只買不賣 > 90%)
- QUICK_KYC            (快速完成KYC < 1天)
...共 12 個 items
```

**輸出**:
- `arm_rules.pkl`: Association rules
- `arm_features.csv`: ARM 特徵 (hit_count, max_confidence, avg_lift, fraud_score)

**範例規則**:
```
{NIGHT_TRANSACTIONS, HIGH_WITHDRAW_RATIO} → fraud
  Support: 0.031, Confidence: 0.847, Lift: 6.2
```

---

### Module 3: Causal Forest

**檔案**: `causal_forest.py`

**功能**: 估計 CATE (Conditional Average Treatment Effect)

**定義**:
- Treatment (T): 近期 USDT 交易量 > 歷史均值 2 倍
- Outcome (Y): 詐欺標籤 或 異常分數
- Covariates (X): 基本特徵 (age, career, income_source...)

**輸出**: `cate_scores.csv`

**解讀 CATE**:
- CATE > 0: 交易增加確實導致風險上升 (主動詐欺)
- CATE ≈ 0: 交易與風險無因果關係
- CATE < 0: 交易增加反而降低風險 (正常業務)

---

### Module 4: 異質圖建構

**檔案**: `bitoguard_hetero_graph.py`

**功能**: 建立 PyG HeteroData

**圖結構**:
```
節點類型:
  - user   (用戶)
  - wallet (錢包地址)

邊類型:
  - (user, sends_to, wallet)         : 用戶發送到錢包
  - (wallet, receives_from, user)    : 錢包接收自用戶
  - (user, transacts_with, user)     : 用戶間交易

邊屬性:
  - amount (log1p 轉換)
  - ARM weight (規則置信度)
```

**輸出**: `graph.pt`

---

### Module 5: GNN 主模型

**檔案**: `models/bitoguard_gnn.py`

**核心模型**: `BitoGuardGNN`

**架構**:
1. **Input Projection**: 將不同節點類型投影到統一維度
2. **Layer 1**: HeteroConv + GATv2 (multi-head attention)
3. **Layer 2**: HeteroConv + GATv2 (single-head)
4. **CF Gating**: CATE 分數門控機制
5. **Output Projection**: 用戶 embedding

**創新點**:
- **ARM-weighted attention**: 規則置信度調整注意力權重
- **CF gating**: 因果效應門控，區分相關性與因果性

**Loss Function**: Focal Loss (處理類別不平衡)

---

## 配置文件

**檔案**: `configs/config.yaml`

主要配置項:
```yaml
# 資料路徑
data:
  raw_data_dir: "./Data"
  output_dir: "./results"

# ARM 參數
arm:
  min_support: 0.01
  min_confidence: 0.6
  min_lift: 2.0

# GNN 參數
gnn:
  hidden_dim: 128
  out_dim: 64
  num_heads: 4
  dropout: 0.3

# 訓練參數
training:
  epochs: 100
  batch_size: 1024
  learning_rate: 0.001
```

---

## 評估指標

| 指標 | 說明 | 目標值 |
|------|------|--------|
| **PR-AUC** | Precision-Recall AUC (主指標) | > 0.85 |
| **ROC-AUC** | ROC曲線面積 | > 0.92 |
| **Precision@100** | Top 100 預測的精確度 | > 0.80 |
| **Recall@5%FPR** | 5% 假陽率時的召回率 | > 0.75 |

---

## 與 Baseline GraphSAGE 的對比

### 架構差異

| 組件 | Baseline | BitoGuard |
|------|----------|-----------|
| 框架 | TensorFlow 1.x | PyTorch + PyG ✅ |
| 圖類型 | 同質圖 | 異質圖 ✅ |
| Aggregator | Mean/MaxPool | GATv2 (Attention) ✅ |
| 特徵 | 靜態 | 動態 (ARM + CATE) ✅ |
| 邊權重 | Uniform | ARM-guided ✅ |
| 因果推斷 | ❌ | Causal Forest ✅ |

### 預期性能提升

- **PR-AUC**: +15~20% (from ~0.70 to ~0.85)
- **False Positive Rate**: -30% (減少誤報)
- **可解釋性**: 從無到有 (SHAP + ARM rules)

---

## 專案結構

```
bitoguard/
├── configs/
│   └── config.yaml                    # 主配置文件
├── models/
│   ├── __init__.py
│   └── bitoguard_gnn.py               # GNN 模型定義
├── utils/
│   ├── __init__.py
│   └── utils.py                       # 工具函數
├── bitoguard_feature_pipeline.py      # Module 1
├── arm_mining.py                      # Module 2
├── causal_forest.py                   # Module 3
├── feature_fusion.py                  # 特徵融合
├── bitoguard_hetero_graph.py          # Module 4
├── README.md                          # 本文件
└── __init__.py
```

---

## 常見問題 (FAQ)

### Q1: 為什麼選擇 GATv2 而非原版 GraphSAGE?

**A**: 原版 GraphSAGE 的 Mean Aggregator 對所有鄰居給予相同權重，但在詐欺圖中：
- 一個騾子帳戶可能連接到數百個正常帳戶和少數詐欺帳戶
- Attention 機制能讓 GNN "專注" 於詐欺鄰居的信號
- GATv2 修復了原版 GAT 的靜態注意力問題

### Q2: ARM 規則有什麼用？

**A**: ARM 規則提供兩種價值：
1. **節點特徵**: 每個用戶命中的規則數量、置信度等作為額外特徵
2. **邊權重**: 兩個用戶若命中相同高置信規則，其邊的 message passing 權重提高

### Q3: Causal Forest 解決什麼問題？

**A**: 解決**混淆因素**問題：
- 高資產用戶本來就交易頻繁，不能直接用「交易量高」作為詐欺信號
- Causal Forest 估計純粹由行為改變帶來的風險增量
- 減少對「正常高交易用戶」的誤報

### Q4: 為什麼用異質圖？

**A**: 加密貨幣轉帳涉及兩種實體：
- **用戶**: 平台內註冊用戶
- **錢包**: 鏈上錢包地址 (可能是外部地址)

異質圖能更準確地建模這種結構，捕捉「用戶 → 錢包 → 用戶」的資金流向模式。

---

## 下一步開發

- [ ] Module 6: LightGBM Stacking 輸出頭
- [ ] Module 7: 可解釋性系統 (GNNExplainer + SHAP)
- [ ] 完整訓練腳本 with early stopping
- [ ] AWS SageMaker 部署腳本
- [ ] Demo Dashboard (React + D3.js)

---

## 參考文獻

1. Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. *NeurIPS*.
2. Brody, S., Alon, U., & Yahav, E. (2021). How attentive are graph attention networks? *ICLR*.
3. Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining association rules. *VLDB*.
4. Athey, S., & Wager, S. (2019). Estimating treatment effects with causal forests. *Annals of Statistics*.

---

## 授權

本專案為 BitoGuard Team 版權所有，僅供 AWS 詐欺偵測競賽使用。

---

**聯絡方式**: BitoGuard Team
**最後更新**: 2026-03-17
