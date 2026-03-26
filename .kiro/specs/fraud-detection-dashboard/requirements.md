# Requirements Document

## Introduction

BitoGuard 詐騙偵測前端 Dashboard 是一個互動式視覺化介面，供風控分析師監控加密貨幣交易網路中的詐騙節點。系統整合 PC-GNN / GraphSAGE 模型的輸出結果，提供節點統計概覽、確定詐騙戶篩選，以及以圖形方式呈現選定節點的完整交易關聯網路。

## Glossary

- **Dashboard**: 整體前端頁面，包含左側統計區塊與右側互動區域
- **Node（節點）**: 代表一個用戶帳戶，對應 `user_id`
- **Fraud_Node（詐騙節點）**: 模型預測或已標記為黑名單（status=1）的節點
- **Risk_Score（風險分數）**: 模型輸出的 [0, 1] 機率值，代表該節點為詐騙的可能性
- **Transaction_Graph（交易網路圖）**: 以節點（用戶）與邊（交易關係）呈現的互動式圖形
- **Subgraph（子圖）**: 以選定節點為中心，包含其 N-hop 鄰居的局部圖
- **Relation_Type（關係類型）**: 邊的類型，包含 R1（共用 IP）、R2（加密貨幣內轉）、R3（共用錢包）
- **Stats_Panel（統計面板）**: 左側顯示整體統計資訊的區塊
- **Node_Selector（節點選擇器）**: 右側上方的下拉式選單，用於篩選確定詐騙戶
- **Graph_Viewer（圖形檢視器）**: 右側主要區域，顯示選定節點的交易網路圖
- **API_Server（API 伺服器）**: 後端服務，提供節點資料與圖形資料的 REST API

---

## Requirements

### Requirement 1：整體佈局與頁面結構

**User Story:** 作為風控分析師，我想要一個清晰的雙欄式 Dashboard 佈局，以便同時查看統計摘要與互動式圖形。

#### Acceptance Criteria

1. THE Dashboard SHALL 以雙欄式佈局呈現，左側為 Stats_Panel（寬度佔 25%），右側為主要操作區域（寬度佔 75%）
2. THE Dashboard SHALL 在視窗寬度小於 1024px 時，自動切換為單欄垂直堆疊佈局
3. THE Dashboard SHALL 顯示頁面標題「BitoGuard 詐騙偵測儀表板」於頁面頂部
4. WHEN 頁面載入完成，THE Dashboard SHALL 在 3 秒內完成初始資料渲染

---

### Requirement 2：左側統計面板（Stats_Panel）

**User Story:** 作為風控分析師，我想要在左側看到關鍵統計數字，以便快速掌握整體詐騙風險狀況。

#### Acceptance Criteria

1. THE Stats_Panel SHALL 顯示以下統計卡片：總節點數、詐騙節點數、正常節點數、詐騙比例（%）
2. THE Stats_Panel SHALL 顯示風險分布圖，以長條圖呈現 Risk_Score 在 [0, 0.2)、[0.2, 0.4)、[0.4, 0.6)、[0.6, 0.8)、[0.8, 1.0] 五個區間的節點數量
3. THE Stats_Panel SHALL 顯示關係類型統計，列出 R1、R2、R3 各類型的邊數量
4. WHEN 統計資料載入失敗，THE Stats_Panel SHALL 顯示錯誤提示訊息並提供重試按鈕
5. THE Stats_Panel SHALL 每 60 秒自動刷新一次統計數據

---

### Requirement 3：節點選擇器（Node_Selector）

**User Story:** 作為風控分析師，我想要透過下拉式選單篩選並選取確定詐騙戶，以便聚焦分析特定高風險節點。

#### Acceptance Criteria

1. THE Node_Selector SHALL 以下拉式選單呈現，預設顯示「請選擇詐騙節點」提示文字
2. THE Node_Selector SHALL 僅列出 Risk_Score 大於等於 0.5 且 status=1 的 Fraud_Node
3. WHEN 使用者在 Node_Selector 輸入關鍵字，THE Node_Selector SHALL 在 300ms 內過濾並顯示符合 user_id 的選項
4. THE Node_Selector SHALL 在每個選項中顯示 user_id 與對應的 Risk_Score（格式：`user_id | 風險分數: 0.XX`）
5. WHEN 使用者選取一個 Fraud_Node，THE Node_Selector SHALL 觸發 Graph_Viewer 載入該節點的 Subgraph
6. IF 可選節點清單為空，THEN THE Node_Selector SHALL 顯示「目前無確定詐騙節點」提示

---

### Requirement 4：交易網路圖（Graph_Viewer）

**User Story:** 作為風控分析師，我想要在選取詐騙節點後，看到該節點的完整交易關聯網路圖，以便分析共犯結構與資金流向。

#### Acceptance Criteria

1. WHEN 使用者選取一個 Fraud_Node，THE Graph_Viewer SHALL 在 5 秒內渲染出以該節點為中心的 2-hop Subgraph
2. THE Graph_Viewer SHALL 以不同顏色區分節點類型：紅色代表 Fraud_Node（status=1）、橘色代表高風險節點（Risk_Score ≥ 0.5）、藍色代表正常節點（Risk_Score < 0.5）
3. THE Graph_Viewer SHALL 以不同線條樣式區分 Relation_Type：實線代表 R2（內轉）、虛線代表 R1（共用 IP）、點線代表 R3（共用錢包）
4. THE Graph_Viewer SHALL 支援滑鼠滾輪縮放（縮放範圍 0.1x 至 5x）與拖曳平移操作
5. WHEN 使用者點擊圖中任一節點，THE Graph_Viewer SHALL 在側邊資訊欄顯示該節點的 user_id、Risk_Score、status 與 top-3 SHAP 特徵
6. THE Graph_Viewer SHALL 在圖形右上角提供「重置視角」按鈕，點擊後恢復至初始縮放與位置
7. IF Subgraph 節點數超過 200 個，THEN THE Graph_Viewer SHALL 顯示警告訊息並僅渲染 1-hop 鄰居
8. WHEN Subgraph 資料載入失敗，THE Graph_Viewer SHALL 顯示錯誤訊息並提供重試按鈕

---

### Requirement 5：節點詳細資訊面板

**User Story:** 作為風控分析師，我想要點擊圖中節點後看到詳細的風險資訊，以便了解該節點被判定為高風險的原因。

#### Acceptance Criteria

1. WHEN 使用者點擊 Graph_Viewer 中的節點，THE Dashboard SHALL 顯示節點詳細資訊面板，包含 user_id、Risk_Score、status、帳戶年齡（account_age_days）
2. THE Dashboard SHALL 在節點詳細資訊面板中顯示 top-3 SHAP 特徵名稱與對應貢獻值（正值以紅色標示，負值以綠色標示）
3. THE Dashboard SHALL 在節點詳細資訊面板中顯示該節點的直接鄰居數量，依 Relation_Type 分類列出
4. WHEN 使用者點擊節點詳細資訊面板中的「設為中心節點」按鈕，THE Graph_Viewer SHALL 重新以該節點為中心渲染 Subgraph
5. IF 節點的 SHAP 資料不存在，THEN THE Dashboard SHALL 在 SHAP 區塊顯示「SHAP 資料不可用」提示

---

### Requirement 6：後端 API 介面

**User Story:** 作為前端開發者，我想要有明確定義的 REST API，以便前端能正確取得節點與圖形資料。

#### Acceptance Criteria

1. THE API_Server SHALL 提供 `GET /api/stats` 端點，回傳總節點數、詐騙節點數、正常節點數、詐騙比例、風險分布直方圖資料、各關係類型邊數
2. THE API_Server SHALL 提供 `GET /api/fraud-nodes` 端點，回傳所有 status=1 節點的 user_id 與 Risk_Score 清單，依 Risk_Score 降序排列
3. THE API_Server SHALL 提供 `GET /api/subgraph/{user_id}?hops=2` 端點，回傳以指定 user_id 為中心的 N-hop Subgraph，包含節點清單（含 user_id、Risk_Score、status）與邊清單（含 source、target、relation_type）
4. THE API_Server SHALL 提供 `GET /api/node/{user_id}` 端點，回傳指定節點的詳細資訊，包含 user_id、Risk_Score、status、account_age_days、top-3 SHAP 特徵
5. IF 請求的 user_id 不存在，THEN THE API_Server SHALL 回傳 HTTP 404 狀態碼與錯誤訊息
6. THE API_Server SHALL 在所有 API 回應中包含 `Content-Type: application/json` 標頭
7. WHEN API 請求處理時間超過 10 秒，THE API_Server SHALL 回傳 HTTP 504 狀態碼

---

### Requirement 7：資料載入與狀態管理

**User Story:** 作為使用者，我想要在資料載入期間看到明確的載入狀態，以便了解系統目前的處理進度。

#### Acceptance Criteria

1. WHILE 資料正在載入，THE Dashboard SHALL 在對應區塊顯示載入動畫（spinner）
2. THE Dashboard SHALL 在頁面初始載入時，同時發送 Stats_Panel 與 Fraud_Node 清單的 API 請求（並行請求）
3. IF 任一 API 請求回傳非 2xx 狀態碼，THEN THE Dashboard SHALL 顯示對應的錯誤訊息，且不影響其他區塊的正常顯示
4. THE Dashboard SHALL 快取 Subgraph 資料，WHEN 使用者重複選取同一節點，THE Dashboard SHALL 直接使用快取資料而不重新發送 API 請求
