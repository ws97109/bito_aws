# 儀表板架構建議

## Overview（總覽）
這頁給評審快速了解全局，是第一印象分數最高的地方：

- 模型效能 KPI 卡片：AUC-ROC (0.859)、F1 (0.362)、Precision、Recall
- 風險分數分佈直方圖：全部用戶的 risk_score 分佈，標出 threshold 切線
- 混淆矩陣：TP / FP / FN / TN 的數量與比例
- 公平性儀表盤：性別、年齡、職業、收入的 PASS/WARNING/FAIL 狀態燈號 + disparate impact ratio gauge

## 黑名單（Predicted Blacklist）

- 高風險排行榜（Top-N 表格）：user_id、risk_score、risk_level
- SHAP 特徵重要性：全局 Top 10 bar chart（你有 shap_all_features.csv）
- 個案 SHAP Waterfall：可下拉選擇某個 user，顯示其 waterfall plot（你已有 waterfall_predict/ 裡的圖）
- GNN 子圖：選定用戶的 1-hop 或 2-hop 鄰居網路圖，節點顏色用 risk_score 上色，邊類型用不同顏色區分（sends / funds / transfers）

---
## 按下按鈕切換(FP/FN)  

### FP — 白的被預測成黑的
- FP 清單表格：user_id、risk_score（你有 237 筆）
- FP 的 SHAP 分析：這群人共同的高 SHAP 特徵是什麼？哪些特徵讓模型誤判？→ 可以做一個「FP 群體 vs 真黑名單」的特徵對比 bar chart
- GNN 子圖：這些 FP 用戶在圖上的結構 — 他們是不是跟黑名單用戶共享錢包？這能解釋為什麼被誤判
- 建議行動：標示這些人可能的誤判原因（如：與黑名單共享同一 wallet address）

### FN — 黑的被預測成白的
- FN 清單表格：user_id、risk_score（你有 203 筆）
- FN 的 SHAP 分析：這群人的風險特徵為何不明顯？哪些特徵值讓他們「躲過」模型？
- GNN 子圖：這些漏網之魚在圖上是否是孤立節點（邊少），導致 GNN 沒抓到？
- 與 TP 對比：FN vs TP 的特徵分佈比較，找出模型盲點