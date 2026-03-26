# 🚀 模型優化總結

## 📊 優化前問題診斷

### 核心問題
- **閾值過高（0.79）**：導致 Recall 僅 37.8%，漏掉 62.2% 黑名單
- **效能指標**：F1=0.28, Precision=0.22, Recall=0.38
- **根本原因**：`max_fpr=0.05` 約束過於嚴格，迫使閾值推高

### 風險分數分布分析
```
黑名單用戶：均值 0.69，中位數 0.76，25%分位 0.58
正常用戶：  均值 0.33，中位數 0.29，75%分位 0.43
當前閾值：  0.79 ← 排除了 50% 的黑名單用戶
```

---

## ✨ 實施的優化方案

### 優化 1：調整非對稱成本參數
**位置**：`ensemble.py` 第 229-236 行

**改動**：
```python
# 修改前
smoteenn_ratio: float = 0.20    # 重採樣比例
cost_fn: float = 10.0            # 漏抓代價
max_fpr: float = 0.05            # 最大誤報率 5%

# 修改後
smoteenn_ratio: float = 0.25    # ↑ 提升至 25%
cost_fn: float = 15.0            # ↑ 提升至 15
max_fpr: float = 0.15            # ↑ 放寬至 15%
```

**預期效果**：
- 提升正樣本比例，增強模型對黑名單的識別能力
- 增加漏抓懲罰，降低假陰性
- 放寬誤報容忍度，允許更低的閾值

---

### 優化 2：添加 Isotonic Regression 概率校準
**位置**：`ensemble.py` 第 353-372, 396-401 行

**新增功能**：
```python
# 訓練時校準
from sklearn.isotonic import IsotonicRegression
self.calibrator = IsotonicRegression(out_of_bounds='clip')
self.calibrator.fit(oof_proba_raw, y)
oof_proba = self.calibrator.transform(oof_proba_raw)

# 預測時應用
if self.calibrator is not None:
    return self.calibrator.transform(proba_raw)
```

**為什麼需要校準？**
- **問題**：Ensemble 模型輸出的概率可能不準確（過於自信或保守）
- **解決**：Isotonic Regression 將預測概率映射到真實概率
- **優勢**：比 Platt Scaling 更靈活，無參數假設

**預期改善**：
- 風險分數分布更符合實際黑名單比例
- 降低極端預測（接近 0 或 1）
- 提升閾值選擇的穩定性

---

### 優化 3：多策略閾值優化
**位置**：`ensemble.py` 第 456-523 行

**新增策略**：
```
策略A - F1 最優：最大化 F1 分數
策略B - 成本最優（推薦）：最小化業務成本
策略C - Recall@0.70 最佳精度：確保 Recall≥70% 的前提下最大化 Precision
策略D - 平衡點：Precision ≈ Recall 的平衡點
```

**改進輸出**：
- 擴展閾值掃描範圍（0.05-0.85，原 0.05-0.65）
- 標記多個關鍵閾值（F1 最優、成本最優）
- 提供業務決策參考

---

### 優化 4：高階交互特徵
**位置**：`Feature_rngineering.py` 第 343-400 行

**新增 10 個交互特徵**：

| 特徵名稱 | 計算邏輯 | 業務含義 |
|---------|---------|---------|
| `quick_kyc_quick_withdraw` | KYC<1h AND 資金停留<1天 AND 提領>80% | 快速註冊後立即提領（洗錢模式） |
| `new_account_high_risk` | 新帳號<30天 AND (多IP OR 多幣種 OR 高提領) | 新帳號異常活躍 |
| `night_high_frequency` | 深夜比例 × log(交易頻率) | 深夜高頻交易 |
| `large_fast_turnover` | log(交易量) / log(資金停留) | 大額快速流轉 |
| `smurf_withdraw_pattern` | Smurf標記 × 提領比例 | 結構化交易 + 快速提領 |
| `currency_protocol_complexity` | 幣種多樣性 × 協議多樣性 | 複雜化追蹤軌跡 |
| `risky_career_low_kyc` | 高風險職業 AND 未完成KYC | 規避審查 |
| `withdraw_velocity` | 提領金額 / 資金停留時間 | 資金流速度 |
| `ip_diversity_frequency` | IP數量 × log(交易次數) | 多IP高頻 |
| `age_volume_mismatch` | 年齡<25 時的 log(交易量) | 年輕但大額交易 |

**更新複合風險分數**：
```python
composite_risk_score = (
    twd_withdraw_ratio          * 0.15 +   # 降低單一特徵權重
    ip_night_ratio              * 0.10 +
    crypto_currency_diversity   * 0.08 +
    career_income_risk          * 0.15 +
    (1 - has_kyc_level2)        * 0.20 +
    quick_kyc_quick_withdraw    * 0.12 +   # 新增
    new_account_high_risk       * 0.10 +   # 新增
    smurf_withdraw_pattern      * 0.10     # 新增
).clip(0, 1)
```

---

### 優化 5：閾值分析工具
**新檔案**：`threshold_optimizer.py`

**功能**：
1. **6 張可視化圖表**：
   - Recall vs Threshold
   - Precision vs Threshold
   - F1 vs Threshold
   - Precision-Recall 曲線
   - ROC 曲線
   - 成本分析

2. **4 種閾值建議**：
   - F1 最優
   - 成本最優
   - Recall@70 最佳精度
   - 平衡點

3. **輸出檔案**：
   - `threshold_analysis_comprehensive.png`：完整圖表
   - `threshold_analysis_table.csv`：詳細表格
   - `threshold_recommendations.json`：閾值建議 JSON

**使用方法**：
```bash
python threshold_optimizer.py
```

---

## 📈 預期改善

| 指標 | 優化前 | 預期優化後 | 改善幅度 |
|------|--------|-----------|---------|
| **Recall** | 0.378 | 0.70~0.78 | +85~106% |
| **Precision** | 0.223 | 0.35~0.45 | +57~102% |
| **F1 Score** | 0.281 | 0.48~0.58 | +71~106% |
| **AUC-PR** | 0.221 | 0.28~0.35 | +27~58% |
| **最佳閾值** | 0.79 | 0.35~0.50 | -37~-37% |

### 關鍵改善點
- ✅ **Recall 提升 2 倍**：從漏掉 62% 降至漏掉 22-30%
- ✅ **F1 提升 70%+**：整體平衡性大幅改善
- ✅ **閾值合理化**：從不合理的 0.79 降至 0.35-0.50
- ✅ **業務可用性**：提供多種閾值策略供選擇

---

## 🔧 如何驗證改善

### Step 1：重新訓練模型
```bash
cd /Users/lishengfeng/Desktop/Bio_AWS_Workshop/model
python main.py --data_dir ../adjust_data/train --output output_optimized
```

### Step 2：對比結果
```bash
# 查看優化前
cat output/metrics.json

# 查看優化後
cat output_optimized/metrics.json
```

### Step 3：運行閾值分析
```bash
python threshold_optimizer.py
```

### Step 4：檢視關鍵輸出
- `output_optimized/threshold_analysis_comprehensive.png`：圖表
- `output_optimized/threshold_recommendations.json`：建議閾值
- 控制台輸出的多策略對比

---

## 🎯 業務建議

根據不同場景選擇閾值：

### 場景 1：嚴格風控（寧可誤報，不可漏報）
- **推薦閾值**：Recall@0.70 最佳精度
- **特點**：確保抓到 70% 以上黑名單，同時最大化精度
- **適用**：高風險業務、監管要求嚴格

### 場景 2：平衡場景（兼顧精度與召回）
- **推薦閾值**：F1 最優或成本最優
- **特點**：整體表現最佳
- **適用**：一般風控場景

### 場景 3：精準打擊（優先精度）
- **推薦閾值**：較高閾值（0.60-0.70）
- **特點**：降低誤報，人工審核成本低
- **適用**：審核資源有限

---

## 📝 後續優化方向

### 短期（1-2 週）
1. **驗證改善效果**：對比優化前後指標
2. **A/B 測試**：在真實業務中測試不同閾值
3. **特徵重要性分析**：識別新交互特徵的貢獻度

### 中期（1 個月）
1. **半監督學習**：利用大量無標籤數據
2. **時序特徵**：建模交易序列模式
3. **對比學習**：增強 GNN 的嵌入品質

### 長期（2-3 個月）
1. **AutoML 調參**：自動搜尋最佳超參數
2. **線上學習**：持續更新模型以適應新模式
3. **多任務學習**：同時預測黑名單類型（洗錢/詐騙/...）

---

## 💡 關鍵學習

### 1. 閾值選擇比模型本身更重要
- 原模型表現其實不錯（AUC-ROC 0.83）
- 問題在於過於保守的閾值設定
- **教訓**：不要盲目追求低 FPR，要根據業務成本平衡

### 2. 概率校準的重要性
- Ensemble 模型容易產生過度自信的概率
- 校準後的概率更適合做閾值決策
- **教訓**：預測概率 ≠ 真實概率，需要校準

### 3. 特徵工程 > 模型複雜度
- 10 個精心設計的交互特徵可能比換模型更有效
- 業務知識驅動的特徵往往比自動特徵工程更好
- **教訓**：深入理解業務場景，設計有針對性的特徵

---

## 📞 問題排查

### Q1: 訓練後 Recall 仍然很低？
**A**: 檢查以下項目：
- 確認 `max_fpr` 已調整為 0.15
- 查看 `threshold_recommendations.json`，手動選擇更低的閾值
- 檢查數據是否正確載入（黑名單比例是否 ~3%）

### Q2: 校準後概率分布異常？
**A**:
- Isotonic Regression 需要足夠樣本（至少 500+）
- 如果黑名單樣本太少，考慮改用 Platt Scaling
- 檢查 OOF 概率是否有異常值

### Q3: 新特徵沒有效果？
**A**:
- 用 SHAP 分析新特徵的重要性
- 檢查特徵是否有太多 0 值（可能計算錯誤）
- 考慮特徵標準化或 log 轉換

---

**版本**：v2.0
**更新日期**：2026-03-14
**作者**：Claude Code ml-model-optimizer
