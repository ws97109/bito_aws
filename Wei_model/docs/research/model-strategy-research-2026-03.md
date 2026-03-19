# 技術評估報告：加密貨幣黑名單用戶偵測 - 模型策略深度調研

## 報告資訊

- 調研日期：2026-03-17
- 調研目的：突破 F1=0.34、AUC-PR=0.30 瓶頸，針對 30:1 極端不平衡的加密貨幣人頭戶偵測任務

---

## 執行摘要

### 核心問題診斷

當前 F1=0.34 的瓶頸通常來自：
1. **決策邊界偏移**：30:1 不平衡使 `scale_pos_weight` 仍不足以讓模型學習少數類細緻特徵
2. **特徵空間多樣性不足**：Stacking 三個 GBDT 模型本質上在相似特徵空間集成
3. **假負例代價未建模**：缺乏非對稱代價函數設計
4. **缺乏未知異常模式覆蓋**：純監督方法只能學習已標注的模式

### 建議優先順序

| 優先級 | 策略 | 預期 F1 提升 | 實作難度 |
|--------|------|-------------|---------|
| P0 | 異常分數特徵（IF + HBOS + LOF） | +0.05~0.10 | 低 |
| P0 | Focal Loss 自訂 objective for LightGBM | +0.03~0.08 | 低 |
| P1 | Borderline-SMOTE + 閾值最佳化 | +0.03~0.06 | 低 |
| P1 | Cost-sensitive Ensemble 重新校準 | +0.02~0.05 | 中 |
| P2 | Pseudo-labeling 半監督 | +0.05~0.12 | 中 |
| P2 | EasyEnsembleClassifier | +0.03~0.07 | 中 |
| P3 | GNN（如有交易圖） | +0.10~0.20 | 高 |

---

## 主題一：非監督式 + 監督式結合策略

### PyOD 核心演算法

**Isolation Forest（最推薦）**
- 對高維度表格資料效果穩定，速度快（O(n log n)）
- 不假設資料分佈，對多重異常模式有覆蓋能力

```python
from pyod.models.iforest import IForest
iforest = IForest(n_estimators=200, contamination=0.033, random_state=42, n_jobs=-1)
iforest.fit(X_train_scaled)
scores = iforest.decision_scores_  # 越高越異常
```

**HBOS（最快）**
- 對每個特徵獨立建直方圖，O(n) 複雜度
- 與 Isolation Forest 異常模式互補

```python
from pyod.models.hbos import HBOS
hbos = HBOS(n_bins=20, contamination=0.033)
hbos.fit(X_train_scaled)
```

**LOF（局部密度異常）**
- 偵測行為上與周圍群體差異顯著的用戶
- `n_neighbors=20`，資料量大時用 CBLOF 替代

**Autoencoder（捕捉複雜非線性）**
- 只用正常樣本訓練，重建誤差高的為異常
- 架構建議：`[64, 32, 16, 32, 64]` 的對稱瓶頸結構

### 三種結合方式

**最推薦：異常分數作為特徵（Strategy A）**

```python
def add_anomaly_features_in_cv(X_train, X_val, contamination=0.033):
    """必須在 CV fold 內部訓練，避免資料洩漏"""
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    features_tr, features_val = {}, {}
    for name, model in [
        ('if', IForest(n_estimators=200, contamination=contamination, random_state=42)),
        ('hbos', HBOS(n_bins=20, contamination=contamination)),
        ('lof', LOF(n_neighbors=20, contamination=contamination))
    ]:
        model.fit(X_tr_sc)
        features_tr[f'{name}_score'] = model.decision_scores_
        features_val[f'{name}_score'] = model.decision_function(X_val_sc)

    return pd.DataFrame(features_tr), pd.DataFrame(features_val)
```

**Two-stage Detection（Strategy B）**：第一階段用非監督模型篩選候選集，第二階段監督分類。缺點：若第一階段漏掉正例則無法補救，通常不如 Strategy A。

**混合 Ensemble（Strategy C）**：加權融合監督機率（0.7）和非監督分數（0.3）。

---

## 主題二：極端不平衡資料進階處理技術

### Focal Loss 實作（LightGBM）

Focal Loss 核心思想：降低「容易分類樣本」的 loss 貢獻，讓模型專注於邊界難分樣本。

```python
def focal_loss_lgb(y_pred, dtrain, alpha=0.75, gamma=2.0):
    """
    alpha: 正例權重，30:1 場景建議 0.70~0.90
    gamma: focusing 強度，建議 1.0~3.0
    """
    y_true = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-y_pred))  # sigmoid
    pt = np.where(y_true == 1, p, 1 - p)
    a = np.where(y_true == 1, alpha, 1 - alpha)
    grad = a * (1 - pt)**gamma * (p - y_true)
    hess = a * (1 - pt)**gamma * p * (1 - p)
    return grad, hess

# 注意：使用 Focal Loss 時，predict() 輸出是 raw score，需要 sigmoid
model = lgb.train(params, dtrain,
    fobj=lambda y_pred, d: focal_loss_lgb(y_pred, d, alpha=0.75, gamma=2.0),
    ...
)
raw_score = model.predict(X_test)
probability = 1.0 / (1.0 + np.exp(-raw_score))
```

**與 scale_pos_weight 的差異**：`scale_pos_weight` 是線性放大正例梯度，Focal Loss 是非線性降權容易樣本，兩者不要同時使用。

### 採樣技術比較

| 方法 | 適合場景 | 建議 sampling_strategy | 注意事項 |
|------|----------|----------------------|---------|
| 標準 SMOTE | 低維度特徵 | 0.3~0.5 | 可能放大雜訊 |
| **Borderline-SMOTE** | **金融欺詐（推薦）** | **0.2~0.4** | **只在邊界生成** |
| ADASYN | 困難樣本集中區域 | 0.2~0.4 | 可能生成離群點 |
| SMOTE-ENN | 需要清理雜訊 | 0.3 | 計算較慢但最穩健 |

```python
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.combine import SMOTEENN

# 推薦：不要採樣到 1:1，0.3 即可
bsmote = BorderlineSMOTE(sampling_strategy=0.3, k_neighbors=5, random_state=42)
X_res, y_res = bsmote.fit_resample(X_train, y_train)
```

**關鍵**：SMOTE 必須放在 Pipeline 內，只能在 CV 的訓練 fold 上執行，絕對不能在整個訓練集上執行後再做 CV。

### 自適應閾值選擇

```python
from sklearn.metrics import precision_recall_curve
import numpy as np

def find_optimal_threshold_f1(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = np.where((precision + recall) > 0,
                  2 * precision * recall / (precision + recall), 0)
    best_idx = np.argmax(f1)
    return thresholds[min(best_idx, len(thresholds)-1)], f1[best_idx]

# 在 OOF 預測上搜尋，而非測試集
best_thresh, best_f1 = find_optimal_threshold_f1(y_train_oof, oof_predictions)
```

### BalancedBaggingClassifier：比 scale_pos_weight 更強

```python
from imblearn.ensemble import EasyEnsembleClassifier, BalancedBaggingClassifier

# EasyEnsemble：多輪欠採樣，每次選不同負例子集（最推薦）
ee = EasyEnsembleClassifier(n_estimators=10, random_state=42, n_jobs=-1)

# 為何優於 scale_pos_weight：
# scale_pos_weight 只放大梯度，不改變資料分佈
# EasyEnsemble 讓每個子分類器都看到均衡訓練集，集成後覆蓋所有負例
```

---

## 主題三：加密貨幣 AML 最新技術趨勢

### Graph-based 方法

**為何強大**：傳統表格方法只用用戶自身特徵，忽略交易網絡的關係資訊。人頭戶往往通過鏈式轉帳洗錢，呈現「peeling chain」或「fan-out」模式。

**GraphSAGE（工業界最常用）**

```python
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F

class AML_GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return F.log_softmax(self.conv2(x, edge_index), dim=1)

# 不平衡處理：在 GNN 的 loss 中加入 class_weight
class_weights = torch.tensor([1.0, 30.0])  # 對應 30:1 不平衡
criterion = torch.nn.NLLLoss(weight=class_weights)
```

**重要資料集**：
- **Elliptic Dataset**（最知名）：203,769 Bitcoin 交易節點
- **AMLSim**（IBM 開發的 AML 模擬器）
- **AML-World**（2023/2024，更大規模）

**2024-2025 最新進展**：
- Temporal Graph Networks (TGN)：處理動態交易圖，捕捉「分層交易」時序模式
- Heterogeneous Graph Networks（HGT）：多類型節點（用戶/地址/交易所）和邊（轉入/轉出/兌換）

**本場景建議**：有錢包地址或用戶交易關係時優先嘗試 GraphSAGE；只有統計特徵時可把 GraphSAGE 嵌入作為額外特徵。

### 時序特徵工程（高價值）

```python
def extract_temporal_features(df, user_col, time_col, amount_col):
    """人頭戶時序行為特徵"""
    features = {}
    for uid, g in df.groupby(user_col):
        g = g.sort_values(time_col)
        ts = pd.to_datetime(g[time_col])
        amounts = g[amount_col].values
        intervals = ts.diff().dt.total_seconds().dropna()

        features[uid] = {
            'interval_mean': intervals.mean() if len(intervals) > 0 else 0,
            'interval_cv': intervals.std() / (intervals.mean() + 1e-8),  # 變異係數
            'burst_ratio': (intervals < 60).mean() if len(intervals) > 0 else 0,
            'night_ratio': (ts.dt.hour < 6).mean(),      # 夜間交易比例
            'round_amount_ratio': (amounts % 1000 == 0).mean(),  # 整數金額
            'amount_autocorr': pd.Series(amounts).autocorr(lag=1) if len(amounts) > 1 else 0,
            'fund_concentration': herfindahl_index(amounts),  # 資金集中度
        }
    return pd.DataFrame(features).T
```

### 半監督學習：Pseudo-labeling

```python
def pseudo_label_iteration(model, X_labeled, y_labeled, X_unlabeled,
                            pos_threshold=0.85, neg_threshold=0.05, max_iter=3):
    """
    迭代式 pseudo-labeling
    正例閾值高（0.85+）：只有高置信度的異常才加入
    負例閾值低（0.05-）：只有高置信度的正常才加入
    """
    X_curr, y_curr = X_labeled.copy(), y_labeled.copy()
    X_unlabeled_curr = X_unlabeled.copy()

    for i in range(max_iter):
        model.fit(X_curr, y_curr)
        probs = model.predict_proba(X_unlabeled_curr)[:, 1]

        pos_mask = probs >= pos_threshold
        neg_mask = probs <= neg_threshold

        if not (pos_mask.any() or neg_mask.any()):
            break

        X_new = np.vstack([X_unlabeled_curr[pos_mask], X_unlabeled_curr[neg_mask]])
        y_new = np.concatenate([np.ones(pos_mask.sum()), np.zeros(neg_mask.sum())])

        X_curr = np.vstack([X_curr, X_new])
        y_curr = np.concatenate([y_curr, y_new])
        X_unlabeled_curr = X_unlabeled_curr[~(pos_mask | neg_mask)]

        print(f"Round {i+1}: +{pos_mask.sum()} pos, +{neg_mask.sum()} neg pseudo-labels")

    return model, X_curr, y_curr
```

---

## 主題四：競賽 Winning Solutions

### IEEE-CIS Fraud Detection（2019，29:1 不平衡）

**第一名（Chris Deotte）核心策略**：

1. **UID 重建**（最關鍵）：通過卡號和設備特徵組合構造用戶唯一識別符，再聚合同一用戶的統計特徵
2. **Frequency Encoding**：統計每個類別值在全量資料中的出現頻率，罕見值是異常信號
3. **Target Encoding（CV 內部執行）**
4. **時序感知切分**：按時間切分 CV fold，避免未來資訊洩漏

### AmEx Default Prediction（2022）

**關鍵技巧**：
- 對每個用戶的 13 個月時序資料計算多種聚合
- DART Boosting（LightGBM）：`lgb.LGBMClassifier(boosting_type='dart', drop_rate=0.1)`
- TabNet 作為集成中的多樣化 learner

### 通用競賽技巧

```python
# 1. Rank Averaging（比加權平均更穩健，適合不平衡）
def rank_average(predictions_list):
    from scipy.stats import rankdata
    ranked = [rankdata(pred) / len(pred) for pred in predictions_list]
    return np.mean(ranked, axis=0)

# 2. 組內排名特徵（隱性歸一化）
df[f'{col}_rank_in_group'] = df.groupby(group_col)[col].rank(pct=True)

# 3. 差分特徵（當前值 vs 歷史均值）
df[f'{col}_diff_hist'] = df[col] - df.groupby(uid_col)[col].transform(
    lambda x: x.expanding().mean().shift(1)
)
```

---

## 量化預期效果

| 方法組合 | 預期 F1(class=1) |
|---------|----------------|
| 當前基線（XGB+LGB+CatBoost Stacking） | 0.34 |
| + 異常分數特徵（IF+HBOS+LOF） | 0.38~0.42 |
| + Focal Loss（alpha=0.75, gamma=2.0） | 0.41~0.46 |
| + Borderline-SMOTE（strategy=0.3） | 0.43~0.49 |
| + 閾值最佳化 | 0.45~0.52 |
| + 時序特徵工程 | 0.48~0.56 |
| + Pseudo-labeling | 0.50~0.60 |

---

## 核心參考資料

**論文**
- Focal Loss：https://arxiv.org/abs/1708.02002
- PyOD：https://www.jmlr.org/papers/v20/19-011.html
- GraphSAGE：https://arxiv.org/abs/1706.03762
- GAT：https://arxiv.org/abs/1710.10903
- AML in Bitcoin (Elliptic)：https://arxiv.org/abs/1908.02591
- AML-World (NeurIPS 2024)：https://arxiv.org/abs/2306.16424
- xFraud：https://arxiv.org/abs/2011.12193

**GitHub**
- PyOD：https://github.com/yzhao062/pyod
- imbalanced-learn：https://github.com/scikit-learn-contrib/imbalanced-learn
- PyTorch Geometric：https://github.com/pyg-team/pytorch_geometric
- Elliptic Dataset：https://github.com/elliptic-co/elliptic-data-set
- AMLSim：https://github.com/IBM/AMLSim
- xFraud：https://github.com/eBay/xFraud
- SUOD（大規模非監督）：https://github.com/yzhao062/SUOD

**Kaggle**
- IEEE-CIS 第一名分析：https://www.kaggle.com/c/ieee-fraud-detection/discussion/111284
- AmEx Winning Solution：https://www.kaggle.com/competitions/amex-default-prediction/discussion/347637

**文件**
- imbalanced-learn 過採樣：https://imbalanced-learn.org/stable/over_sampling.html
- LightGBM 自訂目標函數：https://lightgbm.readthedocs.io/en/stable/Advanced-Topics.html
- sklearn 半監督學習：https://scikit-learn.org/stable/modules/semi_supervised.html
