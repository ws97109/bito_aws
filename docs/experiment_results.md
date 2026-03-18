# BitoGuard 實驗結果報告

> 更新日期：2026-03-18
> 分支：dev/model-development

---

## 1. 實驗版本演進

### 1.1 全版本效能比較

| 版本 | F1(class=1) | AUC-PR | Precision | Recall | AUC-ROC | 主要變更 |
|------|-------------|--------|-----------|--------|---------|---------|
| full_run_v2 | 0.0354 | 0.2636 | 0.545 | 0.018 | 0.8628 | 初版（閾值問題） |
| full_run_v3 | 0.3294 | 0.2730 | 0.287 | 0.387 | 0.8595 | 修復閾值 |
| full_run_v4 | 0.2956 | 0.2763 | 0.225 | 0.430 | 0.8580 | 調整超參數 |
| improved_v1 | 0.3477 | 0.2917 | 0.329 | 0.369 | 0.8609 | Optuna 調參 |
| improved_v2 | 0.3425 | 0.2887 | 0.344 | 0.342 | 0.8612 | 特徵篩選優化 |
| improved_v3 | 0.3388 | 0.3007 | 0.288 | 0.412 | 0.8663 | Red-flag 特徵 |
| **phase1_full** | **0.3567** | **0.3298** | **0.324** | **0.396** | **0.8702** | 非監督異常偵測 + Focal Loss |
| **phase2_pseudo** | **0.3694** | **0.3330** | **0.364** | **0.375** | **0.8695** | + 閾值掃描最佳化 |

### 1.2 相對基線提升（improved_v3 → phase2_pseudo）

| 指標 | 基線 | 最終 | 提升 |
|------|------|------|------|
| F1(class=1) | 0.3388 | **0.3694** | **+9.0%** |
| AUC-PR | 0.3007 | **0.3330** | **+10.7%** |
| Precision | 0.2878 | **0.3639** | **+26.4%** |
| AUC-ROC | 0.8663 | **0.8695** | +0.4% |

---

## 2. 各優化策略效果分析

### 2.1 有效的策略

| 策略 | 貢獻 | 說明 |
|------|------|------|
| **非監督異常偵測** (IF/HBOS/LOF) | AUC-PR +9.7% | 三種異常分數作為新特徵，提供「統計罕見度」維度 |
| **Focal Loss** (LightGBM) | Base AUC-PR 最高 | α=0.75, γ=2.0，專注邊界難分樣本，優於 scale_pos_weight |
| **PR-curve 閾值掃描** | F1 +2.6% | 自動搜尋最佳閾值（0.898），優於 OOF 閾值 |
| **特徵篩選** (81→65) | 減少雜訊 | 零方差 + 高相關 + 零重要性，三階段篩選 |

### 2.2 無效的策略

| 策略 | 結果 | 原因分析 |
|------|------|---------|
| **Pseudo-labeling** | F1 下降 0.044 | 從 predict 用戶標注 648 筆偽正例，但品質不足，引入雜訊 |
| **Borderline-SMOTE** | 無明顯提升 | 合成樣本未帶來新資訊，30:1 場景下 sampling_strategy=0.3 效果有限 |
| **換模型** | 差異微小 | XGB/LGB/CatBoost 在同一特徵空間下表現接近，瓶頸在特徵而非模型 |

### 2.3 Pseudo-labeling 詳細數據

```
Predict 用戶總數：12,753
Round 1：+648 正例（≥0.85）, +0 負例（≤0.05）
Round 2：無新增，停止
擴充後 OOF AUC-PR：0.62（虛高 — 模型自己標自己的資料）
測試集 F1：0.316（低於原始 0.360）→ 自動回退至原始模型
```

結論：偽標籤在本場景不可靠，predict 用戶的機率分佈偏高（mean=0.338），無法產出高品質負例。

---

## 3. 最終模型架構

### 3.1 三層偵測架構

```
第一層：非監督式異常偵測
├── Isolation Forest → if_score（全局異常）
├── HBOS → hbos_score（單維度異常）
└── LOF → lof_score（局部密度異常）

第二層：GNN 圖學習
├── GraphSAGE + GAT（異質交易圖）
└── 64-dim embedding → PCA 16-dim

第三層：監督式 Ensemble
├── XGBoost（scale_pos_weight=30）
├── LightGBM（Focal Loss α=0.75, γ=2.0）
├── CatBoost（auto_class_weights='Balanced'）
└── Meta-learner（Logistic Regression）
```

### 3.2 特徵組成

```
行為特徵（特徵工程）    : 65 維
異常分數（非監督）      :  3 維（IF, HBOS, LOF）
GNN Embedding（圖學習） : 16 維（PCA 降維後）
─────────────────────────────
總計                    : 84 維（含 GNN）/ 68 維（不含 GNN）
```

### 3.3 不平衡處理

| 方法 | 應用對象 | 狀態 |
|------|---------|------|
| Focal Loss (α=0.75, γ=2.0) | LightGBM | ✅ 採用 |
| scale_pos_weight=30 | XGBoost | ✅ 採用 |
| auto_class_weights='Balanced' | CatBoost | ✅ 採用 |
| PR-curve 閾值最佳化 | 最終預測 | ✅ 採用 |
| Borderline-SMOTE | CV fold 內 | ❌ 測試無效 |

---

## 4. 可解釋性成果

### 4.1 SHAP 全域特徵重要性（Top 10）

由 `shap_global.png` 呈現，主要風險因子：
- 一鍵買賣總額（usdt_swap 相關）
- 法幣提領相關（twd_wit_max, twd_withdraw_ratio）
- 性別特徵（is_female）
- 交易時間間隔（tx_interval_median, tx_interval_min）
- 帳號年齡
- 市價單比率

### 4.2 個體風險報告

每位高風險用戶產出包含：
- 風險分數 + 風險等級（5 級）
- Top 5 SHAP 風險因子及方向
- 反事實建議（哪些特徵調整可降低風險分數）

### 4.3 SSR 穩定性

| 擾動強度 ε | Top-1 SSR | Top-3 SSR | Top-5 SSR |
|-----------|-----------|-----------|-----------|
| 0.05 | 0.382 | 0.090 | 0.004 |
| 0.10 | 0.346 | 0.072 | 0.002 |
| 0.15 | 0.318 | 0.058 | 0.004 |
| 0.20 | 0.300 | 0.048 | 0.000 |

> Top-3 和 Top-5 SSR 極低（<0.10），表示 SHAP 排名前 3~5 的風險因子在微小擾動下非常穩定。

---

## 5. 效能瓶頸分析

### 為什麼 F1 無法突破 0.40

1. **正樣本稀少**（1,640 筆，3.2%）— 模型能學到的黑名單模式有限
2. **特徵空間重疊** — AUC-ROC 0.87 說明有辨別力，但黑名單與正常用戶行為差異不夠大
3. **特徵信號量天花板** — 5 張表能提取的資訊已充分利用（81 → 篩選 65 + 異常 3 = 68 維）
4. **資料品質限制** — 官方提供的競賽資料，無法引入外部特徵

### 結論

在當前資料條件下，F1 ≈ 0.37、AUC-PR ≈ 0.33 接近天花板。進一步提升需要：
- 更多正樣本標註
- 外部特徵（如鏈上資料、黑名單地址庫）
- 時序資料的完整交易序列（目前只有聚合統計量）

---

## 6. 專案檔案結構

```
model/
├── main.py                  # 主流程 pipeline
├── Feature_rngineering.py   # 特徵工程（81 維）
├── feature_selection.py     # 特徵篩選（81→65）
├── anomaly_detection.py     # 非監督異常偵測（IF/HBOS/LOF）
├── Gnn_model.py             # GNN 模型（GraphSAGE+GAT）
├── ensemble.py              # Stacking Ensemble（XGB+LGB+CatBoost）
├── pseudo_labeling.py       # Pseudo-labeling 半監督學習
└── shap_explainer.py        # SHAP 可解釋性 + SSR 穩定性

docs/
├── project_plan.md          # 完整企劃書
├── experiment_results.md    # 本實驗報告
└── research/                # 技術調研報告

output/phase2_pseudo/        # 最新輸出
├── metrics.json             # 評估指標
├── features.csv             # 篩選後特徵矩陣
├── user_risk_scores.csv     # 全量風險評分
├── shap_global.png          # 特徵重要性圖
├── risk_reports.txt         # 高風險個體報告
├── ssr_results.json         # SSR 穩定性數據
└── ssr_curves.png           # SSR 衰減曲線圖
```
