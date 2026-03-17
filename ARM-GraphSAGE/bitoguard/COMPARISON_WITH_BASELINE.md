# BitoGuard vs Baseline GraphSAGE: 詳細對比

本文檔詳細說明 BitoGuard 相對於原始 GraphSAGE baseline 的改進與創新。

---

## 一、技術棧對比

### Baseline GraphSAGE (graphsage/)

```python
# 框架與依賴
- TensorFlow 1.x (舊版)
- 手動實作 aggregator 和 layers
- Numpy 處理資料
- 無現代 GNN 框架支援

# 核心檔案
graphsage/
├── models.py           # 基礎模型類別
├── aggregators.py      # Mean/MaxPool/LSTM aggregators
├── supervised_models.py # 監督式 GraphSAGE
└── layers.py           # 自定義層
```

### BitoGuard GraphSAGE-ARM-CF

```python
# 框架與依賴
- PyTorch 2.x (現代框架)
- PyTorch Geometric (專業 GNN 框架)
- mlxtend (ARM 規則挖掘)
- econml (因果推斷)
- LightGBM (Stacking)

# 核心檔案
bitoguard/
├── models/
│   └── bitoguard_gnn.py        # 異質圖 GNN + ARM + CF
├── bitoguard_feature_pipeline.py
├── arm_mining.py
├── causal_forest.py
├── bitoguard_hetero_graph.py
└── train_example.py
```

---

## 二、模型架構對比

### 2.1 圖結構

| 特性 | Baseline | BitoGuard |
|------|----------|-----------|
| **圖類型** | 同質圖 (homogeneous) | **異質圖 (heterogeneous)** |
| **節點類型** | 單一類型 (user) | **2 種類型 (user, wallet)** |
| **邊類型** | 單向或雙向 | **3 種類型**<br>• user→wallet<br>• wallet→user<br>• user→user |
| **實作方式** | Adjacency list (Numpy) | **HeteroData (PyG)** |

**程式碼對比**:

```python
# Baseline: 同質圖
adj = np.load('adj.npy')  # [num_nodes, num_nodes]
features = np.load('features.npy')  # [num_nodes, feature_dim]

# BitoGuard: 異質圖
from torch_geometric.data import HeteroData
data = HeteroData()
data['user'].x = user_features
data['wallet'].x = wallet_features
data['user', 'sends_to', 'wallet'].edge_index = edge_index
```

---

### 2.2 Aggregator (聚合器)

| Aggregator | Baseline | BitoGuard |
|------------|----------|-----------|
| **Mean** | ✅ 均勻權重 | ❌ 被 GATv2 取代 |
| **MaxPool** | ✅ MLP + Max | ❌ 被 GATv2 取代 |
| **LSTM** | ✅ 序列聚合 | ❌ 不適合詐欺偵測 |
| **GATv2** | ❌ 無 | ✅ **Attention-based**<br>• 動態學習鄰居權重<br>• 支援邊特徵<br>• 修正 GAT 靜態問題 |

**Baseline Aggregator**:

```python
# graphsage/aggregators.py
class MeanAggregator(Layer):
    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs
        neigh_means = tf.reduce_mean(neigh_vecs, axis=1)  # 均勻權重
        from_neighs = tf.matmul(neigh_means, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])
        output = tf.add_n([from_self, from_neighs])
        return self.act(output)
```

**BitoGuard Aggregator**:

```python
# bitoguard/models/bitoguard_gnn.py
from torch_geometric.nn import GATv2Conv

self.conv1 = HeteroConv({
    ('user', 'transacts_with', 'user'): GATv2Conv(
        hidden_dim, hidden_dim,
        heads=num_heads,           # 多頭注意力
        dropout=dropout,
        edge_dim=1                 # 支援邊特徵 (ARM weights)
    ),
    # ... 其他邊類型
}, aggr='sum')
```

**差異**:
- **Baseline**: 所有鄰居平等對待 → 無法區分詐欺/正常鄰居
- **BitoGuard**: Attention 機制 → 自動學習「誰重要」→ 提升詐欺信號

---

### 2.3 特徵處理

| 特徵來源 | Baseline | BitoGuard |
|---------|----------|-----------|
| **Tabular 特徵** | 外部給定 (固定) | ✅ 完整 feature engineering pipeline<br>• 43 維特徵<br>• 時間特徵、IP 特徵、行為模式 |
| **ARM 特徵** | ❌ 無 | ✅ **4 維 ARM 特徵**<br>• rule_hit_count<br>• max_confidence<br>• avg_lift<br>• fraud_score |
| **CATE 分數** | ❌ 無 | ✅ **1 維因果效應**<br>• Causal Forest 估計<br>• 區分相關性與因果性 |
| **總維度** | 未定義 | **~48 維** (43+4+1) |

**Baseline 特徵載入**:

```python
# graphsage/supervised_train.py
features = np.load(FLAGS.train_prefix + "-feats.npy")  # 外部提供
```

**BitoGuard 特徵工程**:

```python
# bitoguard/bitoguard_feature_pipeline.py
class BitoGuardFeatureEngineer:
    def extract_user_basic_features(self, user_info):
        # Group A: age, career, KYC timing...

    def extract_twd_transfer_features(self, twd_transfer):
        # Group B: 13 features (tx_count, withdraw_ratio, IP patterns...)

    def extract_usdt_trading_features(self, usdt_twd_trading):
        # Group C: 6 features (buy_ratio, market_ratio...)

    def extract_usdt_swap_features(self, usdt_swap):
        # Group D: 4 features (swap_count, interval...)
```

---

### 2.4 邊權重

| 邊權重 | Baseline | BitoGuard |
|--------|----------|-----------|
| **權重類型** | Uniform (1.0) | **ARM-guided**<br>• Confidence scores<br>• Lift scores |
| **注入方式** | N/A | **edge_attr** 傳入 GATv2Conv |
| **計算公式** | weight = 1.0 | weight = 1.0 + max_confidence × log(max_lift) |

**BitoGuard ARM 邊權重**:

```python
# arm_mining.py
def compute_arm_edge_weight(user_i, user_j, rules):
    shared_rules = get_shared_rule_hits(user_i, user_j, rules)
    if not shared_rules:
        return 1.0  # 預設權重
    max_conf = max(r.confidence for r in shared_rules)
    max_lift = max(r.lift for r in shared_rules)
    return 1.0 + max_conf * log(max_lift)
```

**影響**:
- 若兩個用戶同時命中高置信規則 (如 `{NIGHT_TRANSACTIONS, HIGH_WITHDRAW_RATIO}`)
- 他們之間的 message passing 權重提高 → 詐欺信號增強

---

## 三、訓練流程對比

### 3.1 Loss Function

| Loss | Baseline | BitoGuard |
|------|----------|-----------|
| **類型** | Softmax Cross-Entropy | **Focal Loss** |
| **處理不平衡** | ❌ 無 | ✅ α(1-p_t)^γ log(p_t)<br>• 專注於難樣本<br>• 降低易樣本權重 |
| **超參數** | 無 | α=0.25, γ=2.0 |

**Baseline Loss**:

```python
# graphsage/supervised_models.py
self.loss += tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        logits=self.node_preds,
        labels=self.placeholders['labels']
    )
)
```

**BitoGuard Focal Loss**:

```python
# bitoguard/models/bitoguard_gnn.py
class FocalLoss(nn.Module):
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()
```

**優勢**:
- 詐欺偵測通常有 **極端不平衡** (詐欺 < 5%)
- Focal Loss 能更好地學習少數類別

---

### 3.2 訓練策略

| 策略 | Baseline | BitoGuard |
|------|----------|-----------|
| **Mini-batch** | ✅ 固定鄰居採樣 | ✅ NeighborLoader<br>• 多跳採樣 [10, 5]<br>• 動態批次 |
| **Early Stopping** | ❌ 無 | ✅ Patience=10, min_delta=0.001 |
| **資料分割** | Random split | **Time-based split**<br>• 避免 data leakage<br>• 70% / 15% / 15% |
| **優化器** | Adam | Adam + L2 regularization |

---

## 四、創新點詳解

### 4.1 關聯規則挖掘 (ARM)

**Baseline**: ❌ 無

**BitoGuard**: ✅ FP-Growth 挖掘詐欺模式

**範例規則**:
```
{NIGHT_TRANSACTIONS, HIGH_WITHDRAW_RATIO} → fraud
  Support: 0.031, Confidence: 0.847, Lift: 6.2

{MULTI_IP, BUY_ONLY, FAST_SWAP} → fraud
  Support: 0.018, Confidence: 0.791, Lift: 5.8
```

**價值**:
1. **可解釋性**: 直接告訴調查人員「什麼行為組合導致高風險」
2. **特徵增強**: 每個用戶的 ARM 命中次數成為強特徵
3. **邊權重指導**: 共享規則的用戶間連結加強

---

### 4.2 因果推斷 (Causal Forest)

**Baseline**: ❌ 無

**BitoGuard**: ✅ CATE 估計

**問題**:
- 高資產用戶本來就交易頻繁 → 交易量與風險的相關性可能是**虛假的**
- ML 模型只學「相關性」，容易誤報

**解決方案 (Causal Forest)**:
```
Treatment T: 近期交易量 > 2 × 歷史均值
Outcome Y:   詐欺標籤
Covariates X: 基本特徵 (age, career, income_source...)

CATE = E[Y | T=1, X] - E[Y | T=0, X]  (在控制 X 後)
```

**解讀**:
- CATE > 0: 交易增加**確實導致**風險上升 → 主動詐欺
- CATE ≈ 0: 交易與風險無因果關係 → 正常用戶
- CATE < 0: 交易增加反而降低風險 → 業務擴張

**用途**:
1. **節點特徵**: CATE 分數作為額外特徵
2. **CF 門控**: 高 CATE 節點的 embedding 貢獻更大

```python
# CF-gated readout
gate = torch.sigmoid(self.cf_gate(cate_scores))
user_emb = user_emb * gate  # Element-wise gating
```

---

### 4.3 異質圖建模

**Baseline**: 只有 user→user 邊

**BitoGuard**: 3 種邊類型

```
user_A ──sends_to──> wallet_X ──receives_from──> user_B
   └─────────── transacts_with ──────────────────┘
```

**場景**:
- 詐欺集團 A 發送資金到外部錢包 X
- 錢包 X 轉移到騾子帳戶 B
- 異質圖能捕捉 A→X→B 的完整路徑

---

## 五、預期性能提升

### 5.1 量化指標

| 指標 | Baseline (估計) | BitoGuard (目標) | 提升 |
|------|----------------|------------------|------|
| **PR-AUC** | ~0.70 | **> 0.85** | +21% |
| **ROC-AUC** | ~0.85 | **> 0.92** | +8% |
| **Precision@100** | ~0.65 | **> 0.80** | +23% |
| **False Positive Rate** | ~10% | **< 7%** | -30% |

### 5.2 提升來源分解

| 組件 | PR-AUC 提升 | 說明 |
|------|-------------|------|
| **異質圖** | +5% | 捕捉 user-wallet-user 路徑 |
| **GATv2 Attention** | +7% | 專注詐欺鄰居信號 |
| **ARM 特徵** | +4% | 行為模式共現強特徵 |
| **CATE 分數** | +3% | 區分因果與相關性 |
| **Focal Loss** | +2% | 更好處理不平衡 |
| **總計** | **+21%** | 綜合效果 |

---

## 六、程式碼複雜度對比

### Baseline GraphSAGE

```
graphsage/
├── models.py              (~500 lines)
├── aggregators.py         (~450 lines)
├── supervised_models.py   (~130 lines)
└── supervised_train.py    (~250 lines)

Total: ~1,330 lines (TensorFlow 1.x)
```

### BitoGuard

```
bitoguard/
├── models/bitoguard_gnn.py       (~280 lines, PyTorch)
├── bitoguard_feature_pipeline.py (~380 lines)
├── arm_mining.py                 (~300 lines)
├── causal_forest.py              (~200 lines)
├── bitoguard_hetero_graph.py     (~350 lines)
└── train_example.py              (~200 lines)

Total: ~1,710 lines (PyTorch + PyG + ML pipelines)
```

**差異**:
- BitoGuard 程式碼更現代化 (PyTorch vs TensorFlow 1.x)
- 雖然稍長，但**模組化更好**，每個模組職責單一
- 利用 PyG 等現代框架，實際實作複雜度**更低**

---

## 七、可擴展性對比

| 維度 | Baseline | BitoGuard |
|------|----------|-----------|
| **新邊類型** | 需重寫 aggregator | ✅ 只需在 HeteroConv 加一行 |
| **新特徵來源** | 手動 concat | ✅ 模組化 pipeline |
| **大規模圖** | 全圖訓練 (OOM) | ✅ NeighborLoader mini-batch |
| **分散式訓練** | ❌ 困難 | ✅ PyG + PyTorch DDP |
| **部署** | TF 1.x SavedModel | ✅ TorchScript / ONNX |

---

## 八、總結

### Baseline GraphSAGE 的優勢
✅ 經典算法，論文引用高
✅ 適合學習 GNN 基礎概念

### Baseline GraphSAGE 的劣勢
❌ 框架過時 (TensorFlow 1.x)
❌ 只支援同質圖
❌ 無 attention 機制
❌ 無因果推斷
❌ 無可解釋性

### BitoGuard 的優勢
✅ 現代框架 (PyTorch + PyG)
✅ **異質圖** 建模
✅ **GATv2** attention-based aggregation
✅ **ARM** 關聯規則挖掘
✅ **Causal Forest** 因果推斷
✅ **Focal Loss** 處理不平衡
✅ **模組化設計**，易於擴展
✅ **端對端 pipeline**，從原始資料到預測

### BitoGuard 的挑戰
⚠️ 需要更多依賴 (econml, mlxtend, LightGBM)
⚠️ 訓練時間可能更長 (因為異質圖更複雜)
⚠️ 超參數調整空間更大

---

## 九、使用建議

**何時使用 Baseline GraphSAGE**:
- 學習 GNN 基礎
- 簡單的同質圖場景
- 資源受限環境

**何時使用 BitoGuard**:
- 生產環境的詐欺偵測
- 需要可解釋性
- 有異質實體 (user + wallet)
- 追求最高性能
- 需要因果推斷

---

**文檔版本**: 1.0
**最後更新**: 2026-03-17
**作者**: BitoGuard Team
