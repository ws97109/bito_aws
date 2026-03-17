# BitoGuard 快速開始指南

> 5 分鐘快速了解 BitoGuard 系統並開始使用

---

## 📋 前置條件

```bash
# 檢查 Python 版本
python --version  # 需要 >= 3.8

# 檢查 GPU (optional)
nvidia-smi
```

---

## 🚀 快速安裝

### 方法一：使用 conda (推薦)

```bash
# 1. 建立環境
conda create -n bitoguard python=3.9 -y
conda activate bitoguard

# 2. 安裝 PyTorch (選擇適合你的版本)
# CPU only:
pip install torch torchvision torchaudio

# GPU (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 安裝 PyTorch Geometric
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# 4. 安裝其他依賴
pip install pandas numpy scikit-learn pyyaml mlxtend lightgbm

# 5. (Optional) 安裝因果推斷包
pip install econml
```

### 方法二：使用 pip + requirements.txt

```bash
cd ARM-GraphSAGE/bitoguard
pip install -r requirements.txt  # (需要先建立 requirements.txt)
```

---

## 📁 資料準備

將你的資料放在 `Data/` 目錄下：

```
Bio_AWS_Workshop/
└── Data/
    ├── user_info.csv          ← 用戶基本資料
    ├── twd_transfer.csv       ← 台幣存提款記錄
    ├── usdt_swap.csv          ← USDT 兌換記錄
    ├── usdt_twd_trading.csv   ← USDT/TWD 交易記錄
    └── crypto_transfer.csv    ← 加密貨幣轉帳記錄
```

**資料格式要求**:
- 所有 CSV 必須包含 `user_id` 欄位
- `crypto_transfer` 必須包含: `user_id`, `relation_user_id`, `from_wallet`, `to_wallet`, `amount`
- 時間欄位建議格式: `YYYY-MM-DD HH:MM:SS`

---

## ⚡ 一鍵執行 Pipeline

```bash
cd ARM-GraphSAGE/bitoguard

# 執行完整 pipeline (特徵工程 → ARM → CF → 圖建構)
bash run_pipeline.sh
```

這個腳本會自動執行所有步驟，並在 `results/` 目錄下生成所有中間產物。

---

## 🎯 逐步執行 (手動模式)

如果你想逐步了解每個模組，可以手動執行：

### Step 1: 特徵工程

```bash
python bitoguard_feature_pipeline.py \
    --config configs/config.yaml \
    --output results/features/user_features.csv
```

**輸出**: `user_features.csv` (~43 維特徵)

### Step 2: ARM 規則挖掘

```bash
python arm_mining.py \
    --config configs/config.yaml \
    --features results/features/user_features.csv \
    --output_rules results/features/arm_rules.pkl \
    --output_features results/features/arm_features.csv
```

**輸出**:
- `arm_rules.pkl`: 關聯規則
- `arm_features.csv`: ARM 特徵 (4 維)

### Step 3: Causal Forest

```bash
python causal_forest.py \
    --config configs/config.yaml \
    --features results/features/user_features.csv \
    --output results/features/cate_scores.csv
```

**輸出**: `cate_scores.csv` (CATE 分數)

### Step 4: 特徵融合

```bash
python feature_fusion.py \
    --config configs/config.yaml \
    --tabular results/features/user_features.csv \
    --arm results/features/arm_features.csv \
    --cate results/features/cate_scores.csv \
    --output results/features/node_features.csv
```

**輸出**: `node_features.csv` (~48 維完整特徵)

### Step 5: 異質圖建構

```bash
python bitoguard_hetero_graph.py \
    --config configs/config.yaml \
    --node_features results/features/node_features.csv \
    --output results/graphs/graph.pt
```

**輸出**: `graph.pt` (PyG HeteroData)

### Step 6: 模型訓練

```bash
python train_example.py
```

**輸出**:
- `results/models/best_model.pt`: 最佳模型
- `results/features/gnn_embeddings.npy`: 用戶 embeddings

---

## 🔍 查看結果

### 檢查 ARM 規則

```python
from utils.utils import load_pickle

rules = load_pickle('results/features/arm_rules.pkl')
print(rules.head(10))  # 查看 Top 10 規則
```

### 檢查異質圖統計

```python
import torch

graph = torch.load('results/graphs/graph.pt')
print(f"User nodes: {graph['user'].num_nodes}")
print(f"Wallet nodes: {graph['wallet'].num_nodes}")
print(f"Edges: {graph.num_edges}")
```

### 視覺化 embeddings (Optional)

```python
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

embeddings = np.load('results/features/gnn_embeddings.npy')

# t-SNE 降維
tsne = TSNE(n_components=2, random_state=42)
emb_2d = tsne.fit_transform(embeddings)

# 繪圖
plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=5, alpha=0.5)
plt.title('User Embeddings (t-SNE)')
plt.savefig('embeddings_tsne.png')
```

---

## 🛠️ 自定義配置

編輯 `configs/config.yaml` 來調整參數：

```yaml
# 調整 GNN 模型大小
gnn:
  hidden_dim: 256      # 預設 128
  out_dim: 128         # 預設 64
  num_heads: 8         # 預設 4

# 調整訓練參數
training:
  epochs: 200          # 預設 100
  batch_size: 2048     # 預設 1024
  learning_rate: 0.0005 # 預設 0.001

# 調整 ARM 參數
arm:
  min_confidence: 0.7  # 預設 0.6
  min_lift: 3.0        # 預設 2.0
```

---

## 📊 評估模型

```python
from utils.utils import compute_metrics
import numpy as np

# 載入預測結果
y_true = ...  # 真實標籤
y_proba = ...  # 預測機率

# 計算指標
metrics = compute_metrics(
    y_true,
    (y_proba > 0.5).astype(int),  # 預測標籤
    y_proba
)

print(f"PR-AUC: {metrics['pr_auc']:.4f}")
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")
print(f"Precision@100: {metrics['precision_at_100']:.4f}")
```

---

## ❓ 常見問題

### Q1: CUDA out of memory 錯誤

**解決方案**:
- 降低 `batch_size` (config.yaml)
- 降低 `num_neighbors` 採樣數量
- 使用更小的 `hidden_dim`

### Q2: econml 安裝失敗

**解決方案**:
```bash
# 如果不需要 Causal Forest，可以跳過
# 或使用 conda 安裝
conda install -c conda-forge econml
```

### Q3: 找不到 crypto_transfer 欄位

**檢查**:
- 確認 CSV 檔案中有 `relation_user_id` 欄位
- 確認欄位名稱拼寫正確 (大小寫敏感)

### Q4: ARM 沒有發現任何規則

**原因**: `min_support` 或 `min_confidence` 設定太高

**解決方案**: 降低閾值
```yaml
arm:
  min_support: 0.005     # 從 0.01 降低到 0.005
  min_confidence: 0.5    # 從 0.6 降低到 0.5
```

---

## 📚 進階主題

### 分散式訓練 (多 GPU)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# 初始化 DDP
dist.init_process_group(backend='nccl')

# 包裝模型
model = DistributedDataParallel(model)
```

### 模型部署 (TorchScript)

```python
# 轉換為 TorchScript
model.eval()
scripted_model = torch.jit.script(model)
scripted_model.save('bitoguard_model.pt')

# 載入並推理
loaded_model = torch.jit.load('bitoguard_model.pt')
predictions = loaded_model(x_dict, edge_index_dict)
```

### 超參數調優 (Optuna)

```python
import optuna

def objective(trial):
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256)
    num_heads = trial.suggest_int('num_heads', 2, 8)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

    # 訓練模型
    model = BitoGuardClassifier(
        node_feature_dims=node_feature_dims,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        ...
    )

    # 返回驗證集 PR-AUC
    return val_pr_auc

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

---

## 🎓 學習路徑

**新手**:
1. 閱讀 [README.md](README.md)
2. 執行 `run_pipeline.sh`
3. 理解每個模組的輸入輸出

**進階**:
1. 閱讀 [COMPARISON_WITH_BASELINE.md](COMPARISON_WITH_BASELINE.md)
2. 自定義 ARM behavioral items
3. 實作新的 aggregator

**專家**:
1. 閱讀完整 spec ([BitoGuard_Model_Spec.md](BitoGuard_Model_Spec.md))
2. 實作 Module 6 (Stacking) 和 Module 7 (Explainability)
3. AWS SageMaker 部署

---

## 📞 獲取幫助

- **文檔**: 查看 `README.md` 和 `COMPARISON_WITH_BASELINE.md`
- **範例**: 參考 `train_example.py`
- **設定**: 檢查 `configs/config.yaml`

---

**祝你使用愉快！**

BitoGuard Team
2026-03-17
