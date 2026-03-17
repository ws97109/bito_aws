# BitoGuard 模型訓練完整指南

> 從零開始訓練 BitoGuard 模型的完整步驟

---

## 📋 訓練前檢查清單

### 1. 環境準備

- [ ] Python 3.8+ 已安裝
- [ ] CUDA 環境 (GPU 訓練，可選)
- [ ] 資料檔案已準備好

### 2. 資料檔案清單

確認以下檔案存在於 `Data/` 目錄：

```
Bio_AWS_Workshop/Data/
├── user_info.csv          ✓
├── twd_transfer.csv       ✓
├── usdt_swap.csv          ✓
├── usdt_twd_trading.csv   ✓
└── crypto_transfer.csv    ✓
```

---

## 🚀 完整啟動步驟

### Step 1: 環境安裝 (首次執行)

```bash
# 1.1 建立虛擬環境
cd /Users/tommy/Bio_AWS_Workshop/ARM-GraphSAGE/bitoguard
conda create -n bitoguard python=3.9 -y
conda activate bitoguard

# 1.2 安裝 PyTorch (根據你的系統選擇)
# 選項 A: CPU only
pip install torch torchvision torchaudio

# 選項 B: GPU (CUDA 11.8) - 如果有 NVIDIA GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 1.3 安裝 PyTorch Geometric
pip install torch-geometric

# 如果是 GPU 版本，還需要:
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# 1.4 安裝其他依賴
pip install pandas numpy scikit-learn pyyaml mlxtend lightgbm tqdm

# 1.5 (可選) 安裝因果推斷包
pip install econml

# 1.6 驗證安裝
python -c "import torch; import torch_geometric; print('✓ 安裝成功！')"
```

### Step 2: 檢查資料

```bash
# 2.1 檢查資料目錄
ls -lh ../../Data/

# 2.2 快速查看資料
python -c "
import pandas as pd
import os

data_dir = '../../Data'
files = ['user_info.csv', 'twd_transfer.csv', 'usdt_swap.csv',
         'usdt_twd_trading.csv', 'crypto_transfer.csv']

for f in files:
    path = os.path.join(data_dir, f)
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f'✓ {f}: {len(df)} rows, {len(df.columns)} columns')
    else:
        print(f'✗ {f}: 檔案不存在！')
"
```

**預期輸出**:
```
✓ user_info.csv: 63770 rows, 9 columns
✓ twd_transfer.csv: 195601 rows, 6 columns
✓ usdt_swap.csv: 53841 rows, 6 columns
✓ usdt_twd_trading.csv: 217634 rows, 9 columns
✓ crypto_transfer.csv: XXXXX rows, X columns
```

### Step 3: 執行完整 Pipeline (特徵工程 + 圖建構)

```bash
# 3.1 確保在正確目錄
cd /Users/tommy/Bio_AWS_Workshop/ARM-GraphSAGE/bitoguard

# 3.2 執行 Pipeline (這會花費 5-15 分鐘)
bash run_pipeline.sh
```

**Pipeline 會依序執行**:
1. ✓ 特徵工程 → `results/features/user_features.csv`
2. ✓ ARM 挖掘 → `results/features/arm_rules.pkl`, `arm_features.csv`
3. ✓ Causal Forest → `results/features/cate_scores.csv`
4. ✓ 特徵融合 → `results/features/node_features.csv`
5. ✓ 圖建構 → `results/graphs/graph.pt`

**如果 Pipeline 執行成功，你會看到**:
```
==================================================================================================
                              PIPELINE EXECUTION COMPLETE
==================================================================================================

Output files generated:
  Features:
    - results/features/user_features.csv
    - results/features/arm_features.csv
    - results/features/arm_rules.pkl
    - results/features/cate_scores.csv
    - results/features/node_features.csv

  Graphs:
    - results/graphs/graph.pt
```

### Step 4: 訓練模型 🎯

```bash
# 4.1 直接訓練 (使用預設配置)
python train_example.py
```

---

## ⚙️ 訓練參數配置

### 查看當前配置

打開 `configs/config.yaml`，找到 `training` 部分：

```yaml
# Training
training:
  epochs: 100              # ← 訓練輪數 (預設 100)
  batch_size: 1024         # ← 批次大小
  learning_rate: 0.001     # ← 學習率
  weight_decay: 0.0001     # ← L2 正則化
  num_neighbors: [10, 5]   # ← 鄰居採樣數量 (2-hop)

  # Loss function
  loss_type: "focal"       # ← Focal loss (處理不平衡)
  focal_alpha: 0.25
  focal_gamma: 2.0

  # Early stopping
  patience: 10             # ← 提前停止的耐心值
  min_delta: 0.001

  # Data split (time-based)
  train_ratio: 0.7         # ← 訓練集比例
  val_ratio: 0.15          # ← 驗證集比例
  test_ratio: 0.15         # ← 測試集比例
```

### 修改訓練輪數 (Epochs)

**方法 1: 修改配置文件** (推薦)

```bash
# 編輯 config.yaml
vi configs/config.yaml

# 找到 training.epochs，改成你想要的數字，例如:
epochs: 200  # 改成 200 輪
```

**方法 2: 直接修改訓練腳本**

編輯 `train_example.py`，找到:
```python
for epoch in range(config['training']['epochs']):
```

改成:
```python
for epoch in range(200):  # 直接寫死 200 輪
```

---

## 📊 訓練過程監控

### 訓練輸出範例

```
================================================================================
MODEL SUMMARY
================================================================================
BitoGuardClassifier(
  (gnn): BitoGuardGNN(...)
  (classifier): Linear(in_features=64, out_features=1, bias=True)
)
================================================================================
Total parameters: 234,567
Trainable parameters: 234,567
================================================================================

Starting training...

================================================================================
Epoch 1/100
  Train Loss: 0.5432
  Val PR-AUC: 0.7123
  Val ROC-AUC: 0.8234
  Val F1: 0.6543

Epoch 10/100
  Train Loss: 0.3214
  Val PR-AUC: 0.7856
  Val ROC-AUC: 0.8712
  Val F1: 0.7234

Epoch 20/100
  Train Loss: 0.2543
  Val PR-AUC: 0.8123
  Val ROC-AUC: 0.8923
  Val F1: 0.7567

...

Epoch 67/100

Early stopping at epoch 67
================================================================================

Training complete!
Best validation PR-AUC: 0.8567

Generating final embeddings...
Embeddings saved to: results/features/gnn_embeddings.npy
```

### 訓練時間估算

| 資料規模 | 設備 | 預估時間 (100 epochs) |
|---------|------|---------------------|
| ~60K users, ~200K edges | CPU | 2-4 小時 |
| ~60K users, ~200K edges | GPU (RTX 3080) | 20-40 分鐘 |
| ~60K users, ~500K edges | CPU | 4-8 小時 |
| ~60K users, ~500K edges | GPU (RTX 3080) | 40-80 分鐘 |

---

## 🛠️ 常見問題處理

### Q1: CUDA out of memory 錯誤

**錯誤訊息**:
```
RuntimeError: CUDA out of memory. Tried to allocate XXX MiB
```

**解決方案**:

```yaml
# 編輯 configs/config.yaml
training:
  batch_size: 512        # 從 1024 降到 512
  num_neighbors: [5, 3]  # 從 [10, 5] 降到 [5, 3]

gnn:
  hidden_dim: 64         # 從 128 降到 64
```

### Q2: econml 安裝失敗

**錯誤訊息**:
```
ERROR: Could not find a version that satisfies the requirement econml
```

**解決方案**:

```bash
# 方案 A: 使用 conda 安裝
conda install -c conda-forge econml

# 方案 B: 跳過 Causal Forest (註解掉 Step 3)
# 編輯 run_pipeline.sh，註解掉:
# python causal_forest.py ...

# 然後手動建立 dummy CATE scores
python -c "
import pandas as pd
user_features = pd.read_csv('results/features/user_features.csv')
cate_df = pd.DataFrame({
    'user_id': user_features['user_id'],
    'cate_score': 0.0  # Dummy scores
})
cate_df.to_csv('results/features/cate_scores.csv', index=False)
print('✓ Created dummy CATE scores')
"
```

### Q3: 找不到 crypto_transfer 檔案

**錯誤訊息**:
```
FileNotFoundError: [Errno 2] No such file or directory: '.../crypto_transfer.csv'
```

**解決方案**:

```bash
# 檢查檔案路徑
ls -l ../../Data/crypto_transfer.csv

# 如果檔案名稱不同，修改 configs/config.yaml:
data:
  crypto_transfer: "your_actual_filename.csv"
```

### Q4: ARM 沒有發現任何規則

**輸出**:
```
Generated 0 high-quality rules
```

**原因**: 閾值設定太高

**解決方案**:

```yaml
# 編輯 configs/config.yaml
arm:
  min_support: 0.005     # 從 0.01 降到 0.005
  min_confidence: 0.5    # 從 0.6 降到 0.5
  min_lift: 1.5          # 從 2.0 降到 1.5
```

### Q5: 訓練太慢

**加速方法**:

1. **使用 GPU**:
   ```yaml
   device: "cuda"  # 確保設定為 cuda
   ```

2. **減少 epochs**:
   ```yaml
   training:
     epochs: 50  # 從 100 降到 50
   ```

3. **增大 batch_size** (如果記憶體允許):
   ```yaml
   training:
     batch_size: 2048  # 從 1024 增加到 2048
   ```

4. **減少鄰居採樣**:
   ```yaml
   training:
     num_neighbors: [5, 3]  # 從 [10, 5] 降到 [5, 3]
   ```

---

## 📈 訓練後檢查結果

### 檢查生成的檔案

```bash
# 查看輸出檔案
ls -lh results/models/
ls -lh results/features/

# 應該看到:
# results/models/best_model.pt          ← 最佳模型
# results/features/gnn_embeddings.npy   ← User embeddings
```

### 載入並檢查模型

```python
import torch
import numpy as np

# 載入模型
from models.bitoguard_gnn import BitoGuardClassifier

# 載入圖資料
graph = torch.load('results/graphs/graph.pt')

# 初始化模型
node_feature_dims = {
    'user': graph['user'].x.shape[1],
    'wallet': graph['wallet'].x.shape[1]
}

model = BitoGuardClassifier(
    node_feature_dims=node_feature_dims,
    hidden_dim=128,
    out_dim=64,
    num_heads=4
)

# 載入訓練好的權重
model.load_state_dict(torch.load('results/models/best_model.pt'))
print("✓ 模型載入成功！")

# 載入 embeddings
embeddings = np.load('results/features/gnn_embeddings.npy')
print(f"✓ Embeddings shape: {embeddings.shape}")
```

### 視覺化結果 (可選)

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# t-SNE 降維
tsne = TSNE(n_components=2, random_state=42)
emb_2d = tsne.fit_transform(embeddings[:1000])  # 取前 1000 個用戶

# 繪圖
plt.figure(figsize=(10, 8))
plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=5, alpha=0.5)
plt.title('User Embeddings (t-SNE Visualization)')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.savefig('embeddings_tsne.png', dpi=300)
print("✓ 視覺化圖片已儲存: embeddings_tsne.png")
```

---

## 🎯 快速啟動檢查表

完整執行一遍的指令 (複製貼上即可):

```bash
# 1. 啟動環境
cd /Users/tommy/Bio_AWS_Workshop/ARM-GraphSAGE/bitoguard
conda activate bitoguard

# 2. 檢查資料
ls -l ../../Data/

# 3. 執行 Pipeline (第一次執行)
bash run_pipeline.sh

# 4. 訓練模型
python train_example.py

# 5. 檢查結果
ls -lh results/models/best_model.pt
ls -lh results/features/gnn_embeddings.npy
```

---

## 💡 進階訓練技巧

### 1. 使用 TensorBoard 監控

```python
# 在 train_example.py 中加入:
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/bitoguard_experiment_1')

# 在訓練循環中:
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Metrics/val_pr_auc', val_metrics['pr_auc'], epoch)
writer.close()

# 啟動 TensorBoard:
# tensorboard --logdir=runs
```

### 2. 超參數調優

```python
# 使用 Optuna 自動調參
import optuna

def objective(trial):
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    num_heads = trial.suggest_int('num_heads', 2, 8)

    # 訓練模型並返回驗證 PR-AUC
    val_pr_auc = train_and_evaluate(hidden_dim, lr, num_heads)
    return val_pr_auc

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print(f"Best hyperparameters: {study.best_params}")
```

### 3. 模型 Ensemble

```bash
# 訓練 5 個模型
for i in {1..5}; do
    python train_example.py --seed $i --output models/model_$i.pt
done

# 平均預測結果
python -c "
import torch
import numpy as np

models = [torch.load(f'results/models/model_{i}.pt') for i in range(1, 6)]
# 進行 ensemble prediction
"
```

---

## 📞 需要幫助?

如果遇到問題:

1. **檢查日誌**: 查看錯誤訊息
2. **查看文檔**: 閱讀 README.md 和 ARCHITECTURE.md
3. **調整配置**: 降低 batch_size 或 hidden_dim
4. **簡化流程**: 先跑小規模資料測試

---

**祝訓練順利！** 🚀

---

**文檔版本**: 1.0
**最後更新**: 2026-03-17
