# 🎯 黑名單特徵學習器 - 使用指南

## 📖 核心概念

與傳統的二分類模型不同，**黑名單學習器**採用「**學習黑名單長什麼樣子**」的策略：

### 傳統方法 vs 黑名單學習器

| 比較項目 | 傳統二分類 | 黑名單學習器 |
|---------|-----------|------------|
| **訓練數據** | 需要大量黑名單+正常用戶 | **只需要黑名單樣本** |
| **學習目標** | 區分黑名單與正常用戶 | **學習黑名單的核心特徵模式** |
| **判斷依據** | 模型預測概率 | **與已知黑名單的相似度** |
| **可解釋性** | 較弱（黑盒） | **強（顯示與哪種黑名單類型相似）** |
| **新型黑名單** | 難以識別 | **可識別（只要與已知黑名單相似）** |

---

## 🔬 技術原理

黑名單學習器結合了 **5 種學習策略**：

### 1️⃣ **K-Means 聚類** - 發現黑名單子群組
```
目的：黑名單不是單一類型，可能有多種模式
方法：將黑名單分為 k 個子群（如：洗錢型、詐騙型、機器人型）
輸出：每個新用戶與各子群的相似度
```

**示例**：
```
群組 0: 快速註冊+快速提領（洗錢型）- 50 個黑名單
群組 1: 多 IP + 深夜交易（機器人型）- 30 個黑名單
群組 2: 大額 + Smurf 交易（結構化洗錢）- 40 個黑名單
群組 3: 多幣種 + 快速流轉（混淆追蹤）- 35 個黑名單
群組 4: 其他異常模式 - 25 個黑名單
```

---

### 2️⃣ **One-Class SVM** - 學習黑名單的邊界
```
目的：定義「正常的黑名單」應該在哪個範圍內
方法：只用黑名單訓練，學習它們的分布邊界
輸出：新用戶是否在「黑名單空間」內
```

**視覺化**：
```
    │
    │   ╭─────────╮
    │   │ 黑名單  │ ← SVM 學習這個邊界
    │   │  空間   │
    │   ╰─────────╯
    │  ○ ← 新用戶在邊界內 → 高相似度
    │              × ← 新用戶在邊界外 → 低相似度
    └──────────────
```

---

### 3️⃣ **Isolation Forest** - 異常檢測視角
```
目的：黑名單是「異常」，檢測新用戶的異常程度
方法：構建隨機決策樹，異常點更容易被孤立
輸出：新用戶的異常分數
```

**原理**：正常用戶聚集在一起，黑名單通常在外圍（異常）

---

### 4️⃣ **K-NN 最近鄰** - 相似度計算
```
目的：直接計算與最近 k 個黑名單的距離
方法：找出最相似的 k 個已知黑名單
輸出：平均距離（越近 = 越相似）
```

**應用**：可以回答「這個用戶和哪些已知黑名單最像？」

---

### 5️⃣ **綜合相似度分數** - 多策略融合
```python
綜合分數 = (
    SVM 分數       * 0.30 +
    異常分數       * 0.25 +
    KNN 相似度     * 0.25 +
    聚類中心相似度  * 0.20
)
```

**優勢**：結合多個視角，更穩健

---

## 🚀 使用方法

### **方法 1：快速測試**
```bash
cd /Users/lishengfeng/Desktop/Bio_AWS_Workshop/model
python test_blacklist_learner.py
```

**輸出**：
- `output/blacklist_space.png` - 黑名單特徵空間可視化
- `output/blacklist_learner_results.csv` - 詳細預測結果
- 控制台輸出：效能指標、子群組分析、高風險用戶解釋

---

### **方法 2：整合到現有流程**

在 `main.py` 中添加：

```python
from blacklist_learner import BlacklistLearner, demo_blacklist_learner

# ... 現有訓練流程 ...

# 在評估之後添加
print("\n[Step 7] 黑名單特徵學習器")
learner, bl_result = demo_blacklist_learner(
    X_tr, y_tr, X_te, y_te,
    feature_names=feature_names,
    output_dir=output_dir
)
```

---

### **方法 3：獨立使用**

```python
from blacklist_learner import BlacklistLearner

# 1. 訓練（只用黑名單）
learner = BlacklistLearner(
    n_clusters=5,         # 黑名單子群數量
    contamination=0.05,   # 異常檢測容忍度
    n_neighbors=10,       # K-NN 鄰居數
)
learner.fit(X_all, y_all)

# 2. 預測新用戶
result = learner.predict_similarity(X_new)
# result['combined_score'] - 綜合相似度 [0, 1]
# result['closest_cluster'] - 最相似的黑名單群組

# 3. 判定是否為黑名單
y_pred = learner.predict(X_new, threshold=0.5)

# 4. 解釋為什麼判定為黑名單
explanation = learner.explain_user(user_idx, X_new)
print(f"相似度: {explanation['combined_score']:.3f}")
print(f"最相似群組: 群組 {explanation['closest_cluster']}")
print(f"最相似的黑名單: {explanation['similar_blacklist_indices']}")
```

---

## 📊 預期效果

### **優勢場景**

1. **黑名單樣本稀少** ✅
   - 只需要少量黑名單即可學習
   - 不依賴大量正常用戶標註

2. **新型態黑名單** ✅
   - 可以識別「與已知黑名單相似」的新用戶
   - 即使是全新的洗錢手法，只要行為模式相似就能檢測

3. **可解釋性要求高** ✅
   - 明確指出「與哪種黑名單類型相似」
   - 可以追溯到具體的已知黑名單案例

4. **不平衡數據** ✅
   - 不受正負樣本比例影響
   - 黑名單比例 0.1% 也能有效學習

### **性能對比**

| 指標 | 傳統 Ensemble | 黑名單學習器 | 說明 |
|------|--------------|------------|------|
| **AUC-ROC** | 0.83 | 0.75~0.80 | 略低，但更穩健 |
| **可解釋性** | ⭐⭐ | ⭐⭐⭐⭐⭐ | 可追溯到具體案例 |
| **新型黑名單** | ⭐⭐ | ⭐⭐⭐⭐⭐ | 泛化能力強 |
| **訓練速度** | 慢 | 快（只學習黑名單） |
| **數據需求** | 高 | 低（只需黑名單） |

---

## 🎨 可視化示例

### **黑名單特徵空間**

```
         PCA Component 2
              │
        群組1 ●│
              │  群組2 ●
              │      ╲
              │       ● 群組3
     ─────────┼───────────── PCA Component 1
         群組4│●
              │   群組5 ●
              │
```

- **紅色 X**：群組中心
- **彩色點**：已知黑名單（訓練集）
- **藍色圓圈**：測試集正常用戶
- **橙色三角**：測試集黑名單

**解讀**：
- 測試集黑名單靠近已知黑名單 → 模型有效
- 測試集正常用戶遠離黑名單群組 → 區分度高

---

## 💼 業務應用

### **場景 1：人工審核輔助**
```
系統檢測到高風險用戶 → 顯示相似度報告 → 人工審核

報告內容：
- 綜合相似度：0.87（極高）
- 最相似群組：群組 2（多 IP + 深夜交易型）
- 最相似的已知黑名單：用戶 #12345, #23456, #34567
- 關鍵特徵：ip_unique_count=15, ip_night_ratio=0.85
```

**優勢**：審核人員可以參考歷史案例，提升效率

---

### **場景 2：新型黑名單偵測**
```
傳統模型無法識別的新型洗錢手法
    ↓
黑名單學習器檢測到「與群組 3 相似度 0.75」
    ↓
雖然手法新穎，但行為模式與已知案例類似
    ↓
成功攔截
```

---

### **場景 3：黑名單演化追蹤**
```
定期分析黑名單子群組的變化：

2026-Q1: 5 個群組，主要是快速提領型
2026-Q2: 出現新群組（混合幣種型），需加強監控
2026-Q3: 群組 2 消失，可能手法已被破解
```

**應用**：動態調整風控策略

---

## 🔍 進階功能

### **1. 閾值調整**

不同業務場景使用不同閾值：

```python
# 嚴格模式（寧可誤報）
y_pred_strict = learner.predict(X_new, threshold=0.3)

# 平衡模式
y_pred_balanced = learner.predict(X_new, threshold=0.5)

# 保守模式（只抓最明顯的）
y_pred_conservative = learner.predict(X_new, threshold=0.7)
```

---

### **2. 子群組特徵分析**

找出每個黑名單群組的典型特徵：

```python
for i in range(learner.n_clusters):
    cluster_mask = (learner.cluster_labels == i)
    cluster_features = learner.blacklist_features[cluster_mask].mean(axis=0)

    # 找出該群組最顯著的特徵
    top_idx = np.abs(cluster_features).argsort()[-5:][::-1]
    print(f"群組 {i} 的典型特徵：")
    for idx in top_idx:
        print(f"  {feature_names[idx]}: {cluster_features[idx]:.3f}")
```

---

### **3. 相似黑名單查詢**

給定一個新用戶，找出最相似的已知黑名單：

```python
explanation = learner.explain_user(new_user_idx, X_new)

print("最相似的已知黑名單：")
for idx, dist in zip(
    explanation['similar_blacklist_indices'],
    explanation['similar_blacklist_distances']
):
    print(f"  用戶 #{training_user_ids[idx]}, 距離={dist:.3f}")
```

**應用**：審核時可以參考歷史案例

---

## 🛠️ 故障排除

### **Q1: 聚類數量應該設多少？**

**A**：根據黑名單樣本數量：
- 50-100 個黑名單 → `n_clusters=3`
- 100-500 個黑名單 → `n_clusters=5`
- 500+ 個黑名單 → `n_clusters=8-10`

**驗證**：查看 Silhouette Score（> 0.3 為佳）

---

### **Q2: 相似度分數偏低怎麼辦？**

**A**：可能原因：
1. **特徵標準化問題** → 確認已使用 `StandardScaler`
2. **黑名單樣本太少** → 至少需要 20+ 個黑名單
3. **特徵選擇不當** → 移除噪聲特徵

**解決**：
```python
# 增加 KNN 權重（更關注最近鄰）
combined_score = (
    svm_score * 0.20 +
    iso_score * 0.20 +
    knn_similarity * 0.40 +  # ← 提高
    cluster_similarity * 0.20
)
```

---

### **Q3: 如何結合傳統 Ensemble 模型？**

**A**：兩種策略：

**策略 1：投票融合**
```python
# 傳統模型預測
ensemble_pred = ensemble.predict_proba(X_test)

# 黑名單學習器預測
bl_pred = learner.predict_similarity(X_test)['combined_score']

# 加權融合
final_score = ensemble_pred * 0.6 + bl_pred * 0.4
```

**策略 2：串聯使用**
```python
# 第一階段：Ensemble 粗篩（閾值較低）
candidates = ensemble_pred > 0.3

# 第二階段：黑名單學習器精篩
final_pred = learner.predict(X_test[candidates], threshold=0.5)
```

---

## 📈 性能優化建議

### **1. 特徵降維**

如果特徵數 > 100，建議先 PCA 降維：
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)  # 保留 95% 方差
X_reduced = pca.fit_transform(X_scaled)
```

---

### **2. 增量更新**

當有新的黑名單案例時：
```python
# 重新訓練（加入新黑名單）
X_all_updated = np.vstack([X_all, X_new_blacklist])
y_all_updated = np.hstack([y_all, np.ones(len(X_new_blacklist))])

learner.fit(X_all_updated, y_all_updated)
```

---

### **3. 批次預測**

大量用戶時，使用批次處理：
```python
batch_size = 1000
results = []

for i in range(0, len(X_huge), batch_size):
    batch = X_huge[i:i+batch_size]
    result = learner.predict_similarity(batch)
    results.append(result)
```

---

## 🎓 延伸閱讀

### **相關論文**

1. **One-Class SVM**
   - Tax, D. M., & Duin, R. P. (2004). Support vector data description. Machine learning.

2. **Isolation Forest**
   - Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. ICDM.

3. **Prototypical Networks**
   - Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical networks for few-shot learning. NeurIPS.

### **進階技術**

- **Siamese Network**：學習成對樣本的相似度
- **Triplet Loss**：拉近黑名單內部距離，推遠與正常用戶距離
- **Self-Supervised Learning**：利用無標籤數據預訓練特徵提取器

---

## 📞 技術支援

如有問題，檢查以下項目：

1. ✅ 黑名單樣本數量 ≥ 20
2. ✅ 特徵已標準化
3. ✅ 無缺失值或無窮值
4. ✅ `sklearn`, `matplotlib` 已安裝

**執行測試**：
```bash
python test_blacklist_learner.py
```

---

**版本**：v1.0
**更新日期**：2026-03-19
**作者**：Claude Code - Bio AWS Workshop
