# LOO Toxicity 導入報告

> 實驗日期：2026-04-21
> 分支：`feat/model-improvements`
> 動機：參考 2026 DIGITIMES AI 黑客松第一名 BitoGuard（[gttthuang/Bito](https://github.com/gttthuang/Bito)），該團隊靠 LOO Toxicity 將 F1 從 0.36 推到 0.80。

---

## 1. LOO Toxicity 是什麼

**Leave-One-Out 毒性分數**：對每個使用者 i 在每個錢包 / IP 上，計算「除了自己以外的其他使用者裡，黑名單佔比」。

核心公式：
```
tox(wallet_w, user_i) = (bl_count_w - is_bl_i + S × global_rate) / (total_count_w - 1 + S)
```

- `S = 50` 平滑常數，穩定低樣本估計
- 減掉 `is_bl_i` 與 `-1` = Leave-One-Out，防止目標洩漏
- 對於 predict 集使用者（無 label）走 else 分支，不減自己

## 2. 為什麼能解決我們的 F1 瓶頸

我們之前卡在 F1=0.37、AUC-PR=0.33 有兩個根因：
1. 特徵都是**「使用者自己的行為」**，沒有關聯性
2. GNN 雖然是關聯模型，但在 1,640 正例下學不起來（訓練 loss 1.08）

LOO Toxicity 直接把**關聯資訊寫死成公式**，不用學習，GBDT 秒懂。

## 3. 新增特徵（14 維）

| 類別 | 特徵 | 含義 |
|------|------|------|
| Wallet 毒性 | `w_tox_max`, `w_tox_mean`, `w_tox_std` | 共用錢包的黑名單密度（極值/均值/標準差）|
| Wallet 毒性 | `toxic_w_ratio`, `toxic_w_count` | 超過門檻的錢包比例/數量 |
| IP 毒性 | `ip_tox_mean`, `ip_tox_max`, `toxic_ip_count` | 共用 IP 的黑名單密度 |
| 直接關聯 | `relation_blacklist_ratio`, `relation_blacklist_count` | 內轉對象中黑名單比例 |
| 二階鄰居 | `neighbor_tox_mean`, `neighbor_tox_max` | 錢包共用的「鄰居的鄰居」毒性 |
| 二階鄰居 | `toxic_neighbor_count`, `n_neighbors` | 毒性鄰居數量 / 鄰居總數 |

## 4. Smoke Test：LOO 特徵 vs 標籤的區別能力

跑完 `build_toxicity_features` 後，比較 status=0 與 status=1 的均值：

| 特徵 | status=0（正常） | status=1（黑名單）| 倍數 |
|------|----------------|-----------------|------|
| toxic_w_ratio | 0.103 | **0.258** | 2.5× |
| toxic_w_count | 0.341 | **0.958** | 2.8× |
| toxic_ip_count | 0.002 | **0.099** | **55×** |
| relation_blacklist_count | 0.003 | **0.023** | 9× |
| w_tox_max | 0.039 | **0.061** | 1.6× |
| neighbor_tox_max | 0.122 | **0.154** | 1.3× |

`toxic_ip_count` 在黑名單身上是正常使用者的 **55 倍**——這已經是業界最強訊號之一。

## 5. 改動摘要（最小化、可回滾）

**唯一的程式碼改動**：
- `Feature_engineering.py` 新增 `build_toxicity_features` 函式
- `Feature_engineering.py` 的 `build_all_features` 新增 `train_label` optional 參數
- `main.py` 在三處 `build_all_features` 呼叫加上 `train_label=train_label_df`
- predict 端的特徵工程改成合併 train+predict 後再跑（因 LOO 需要共享圖）

**沒有動**：`ensemble.py`、`Gnn_model.py`、`anomaly_detection.py`、`shap_explainer.py`。

## 6. 實測結果（Test set，10,204 筆，328 正例）

### 6.1 Baseline vs + LOO

| 配置 | F1 | AUC-PR | AUC-ROC | Precision | Recall |
|------|-----|--------|---------|-----------|--------|
| Baseline (no GNN, no LOO) | 0.3746 | 0.3262 | 0.8641 | 0.3552 | 0.3963 |
| **Baseline + LOO (no GNN)** | **0.8277** | **0.8600** | **0.9778** | **0.9280** | **0.7470** |
| **Δ** | **+121%** | **+164%** | **+13%** | **+161%** | **+88%** |

**完全壓倒性突破**。每個指標都顯著提升，而且：
- 5-fold CV 每 fold 的 AUC-PR 都在 0.82~0.89（穩定）
- Stacking OOF AUC-PR = 0.8557（泛化良好）
- Precision 0.928 意味只有 7% 誤報率

### 6.2 + LOO 後多閾值策略

| 策略 | 閾值 | F1 | F2 | Precision | Recall |
|------|------|-----|-----|-----------|--------|
| max_f1（預設） | 0.9784 | **0.8277** | 0.7773 | 0.928 | 0.747 |
| max_f2（Recall 優先） | 0.8670 | 0.7694 | **0.8043** | 0.718 | 0.829 |
| min_cost（10:1）| 0.8601 | 0.7615 | 0.8025 | 0.702 | **0.832** |

### 6.3 Base Learner 單獨效能對照

| 模型 | Before LOO (5-fold avg) | After LOO (5-fold avg) |
|------|-------------------------|------------------------|
| XGBoost | ~0.22 | **0.8431** |
| LightGBM (Focal) | ~0.24 | **0.8537** |
| CatBoost | ~0.23 | **0.8419** |
| Stacking OOF | ~0.29 | **0.8557** |

### 6.4 SHAP 特徵重要性（Top 10）

| 排名 | 特徵 | SHAP % | 類型 |
|------|------|--------|------|
| 1 | **w_tox_max** | **25.4%** | 🟢 LOO |
| 2 | **w_tox_mean** | **9.2%** | 🟢 LOO |
| 3 | **neighbor_tox_mean** | **8.5%** | 🟢 LOO |
| 4 | tx_interval_median | 3.0% | 行為 |
| 5 | twd_net_flow | 2.9% | 行為 |
| 6 | weekend_tx_ratio | 2.7% | 行為 |
| 7 | if_score | 2.4% | 異常 |
| 8 | career_freq | 2.3% | 用戶 |
| 9 | account_age_days | 2.2% | 用戶 |
| 10 | swap_sum | 2.1% | 行為 |

**LOO 特徵合計約 48% 的重要性**，確認了「關聯比行為強」的假說。

### 6.5 Predict 提交檔案（12,753 人）

| Submission | 閾值 | 判黑名單數 | 佔比 |
|-----------|------|-----------|------|
| submission.csv（max_f1）| 0.9784 | 71 | 0.56% |
| submission_max_f2.csv | 0.8670 | 244 | 1.91% |
| submission_min_cost.csv | 0.8601 | 252 | 1.98% |
| submission_top200.csv | — | 200 | 1.57% |
| submission_top500.csv | — | 500 | 3.92% |
| submission_top1000.csv | — | 1,000 | 7.84% |

`max_f1` 版本最保守（0.56%）因為 Precision 0.928 需要極高閾值。如果比賽實際正例率接近 train 的 3.21%，`top500` 或 `max_f2` 可能更貼近答案。建議先提交 `max_f1` 看真實分數，再用 `top500` / `max_f2` 當備援。

---

## 7. 致謝

LOO 公式與實作思路完全來自 2026 DIGITIMES AI 黑客松第一名團隊 BitoGuard（[gttthuang/Bito](https://github.com/gttthuang/Bito)）。本實作為忠實移植。
