# Implementation Plan: BitoGuard 詐騙偵測儀表板

## Overview

以 React 18 + TypeScript 實作純前端詐騙偵測儀表板，使用 mock data 模擬後端 API，透過 react-force-graph-2d 呈現交易網路圖，recharts 呈現統計圖表，Tailwind CSS 處理樣式，React Context + useReducer 管理全域狀態。

## Tasks

- [x] 1. 建立專案結構與核心型別定義
  - 在 `frontend/src/` 下建立 `api/`、`mock/`、`components/`、`context/`、`hooks/`、`types/` 目錄結構
  - 在 `types/index.ts` 定義所有共用型別：`StatsResponse`、`FraudNode`、`SubgraphNode`、`SubgraphEdge`、`SubgraphResponse`、`NodeDetailResponse`、`ShapFeature`、`DashboardState`
  - 安裝依賴：`react-force-graph-2d`、`recharts`、`fast-check`（dev）、`vitest`（dev）、`@testing-library/react`（dev）
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [x] 2. 實作 Mock 資料層
  - [x] 2.1 建立 `mock/statsData.ts`，包含約 500 節點的統計摘要、五個風險區間分布、R1/R2/R3 邊數量
    - _Requirements: 2.1, 2.2, 2.3, 6.1_
  - [x] 2.2 建立 `mock/fraudNodesData.ts`，包含約 50 個 status=1 節點，依 risk_score 降序排列
    - _Requirements: 3.2, 6.2_
  - [x] 2.3 建立 `mock/subgraphData.ts`，包含多個節點的 2-hop subgraph 資料（節點含 user_id、risk_score、status；邊含 source、target、relation_type）
    - _Requirements: 4.1, 6.3_
  - [x] 2.4 建立 `mock/nodeData.ts`，包含各節點的詳細資訊（含 SHAP top-3 特徵、鄰居數量）
    - _Requirements: 5.1, 5.2, 5.3, 6.4_

- [x] 3. 實作 API Service 層
  - [x] 3.1 建立 `api/client.ts`，定義基礎 fetch 函數與錯誤處理（含 404、504 處理）
    - _Requirements: 6.5, 6.6, 6.7_
  - [x] 3.2 建立 `api/statsApi.ts`，實作 `getStats()`，目前回傳 mock 資料，預留 HTTP 替換位置
    - _Requirements: 6.1_
  - [x] 3.3 建立 `api/fraudNodesApi.ts`，實作 `getFraudNodes()`，回傳降序排列的詐騙節點清單
    - _Requirements: 6.2_
  - [x] 3.4 建立 `api/subgraphApi.ts`，實作 `getSubgraph(userId, hops)`，支援 hops 參數
    - _Requirements: 6.3_
  - [x] 3.5 建立 `api/nodeApi.ts`，實作 `getNodeDetail(userId)`
    - _Requirements: 6.4_

- [x] 4. 實作全域狀態管理（DashboardContext）
  - [x] 4.1 建立 `context/DashboardContext.tsx`，定義 `DashboardState`、actions 與 reducer
    - 包含 `stats`、`fraudNodes`、`selectedUserId`、`subgraph`、`selectedNode`、`subgraphCache`、`loading`、`error` 狀態
    - _Requirements: 7.1, 7.2, 7.3, 7.4_
  - [x] 4.2 在 reducer 中實作 subgraph 快取邏輯：選取已快取的 user_id 時直接使用快取，不觸發 API 呼叫
    - _Requirements: 7.4_
  - [ ]* 4.3 為 reducer 撰寫屬性測試
    - **Property 12: Subgraph 快取避免重複請求**
    - **Validates: Requirements 7.4**

- [x] 5. 實作 Custom Hooks
  - [x] 5.1 建立 `hooks/useStats.ts`，呼叫 `getStats()`，管理 loading/error 狀態，實作 60 秒自動刷新
    - _Requirements: 2.5, 7.1_
  - [x] 5.2 建立 `hooks/useFraudNodes.ts`，呼叫 `getFraudNodes()`，管理 loading/error 狀態
    - _Requirements: 3.2, 7.1_
  - [x] 5.3 建立 `hooks/useSubgraph.ts`，呼叫 `getSubgraph()`，整合快取邏輯，處理節點數 > 200 自動降級為 1-hop
    - _Requirements: 4.7, 7.4_

- [x] 6. 實作通用元件
  - [x] 6.1 建立 `components/common/Spinner.tsx`，顯示載入動畫
    - _Requirements: 7.1_
  - [x] 6.2 建立 `components/common/ErrorMessage.tsx`，顯示錯誤訊息與可選的重試按鈕
    - _Requirements: 2.4, 4.8, 7.3_

- [x] 7. 實作統計面板元件
  - [x] 7.1 建立 `components/stats/StatCard.tsx`，顯示單一統計數值（標題 + 數值）
    - _Requirements: 2.1_
  - [x] 7.2 建立 `components/stats/RiskBarChart.tsx`，使用 recharts `BarChart` 呈現五個風險區間分布
    - _Requirements: 2.2_
  - [x] 7.3 建立 `components/stats/RelationStats.tsx`，列出 R1/R2/R3 邊數量
    - _Requirements: 2.3_
  - [x] 7.4 建立 `components/stats/StatsPanel.tsx`，組合 StatCard × 4、RiskBarChart、RelationStats，處理 loading/error 狀態
    - _Requirements: 2.1, 2.2, 2.3, 2.4_
  - [ ]* 7.5 為 StatsPanel 撰寫屬性測試
    - **Property 1: StatsPanel 正確渲染所有統計資訊**
    - **Validates: Requirements 2.1, 2.2, 2.3**

- [x] 8. 實作節點選擇器元件
  - [x] 8.1 建立 `components/graph/NodeSelector.tsx`，實作下拉選單、300ms debounce 搜尋過濾、空清單提示
    - 選項格式：`{user_id} | 風險分數: {risk_score}`（兩位小數）
    - _Requirements: 3.1, 3.3, 3.4, 3.6_
  - [x] 8.2 實作 `getFilteredNodes(nodes, keyword)` 純函數，過濾 `status=1 && risk_score >= 0.5` 且 user_id 含關鍵字的節點
    - _Requirements: 3.2, 3.3_
  - [ ]* 8.3 為節點過濾邏輯撰寫屬性測試
    - **Property 2: NodeSelector 過濾邏輯正確性**
    - **Validates: Requirements 3.2, 3.3**
  - [ ]* 8.4 為選項格式化撰寫屬性測試
    - **Property 3: NodeSelector 選項格式正確性**
    - **Validates: Requirements 3.4**

- [x] 9. 實作圖形視覺化核心函數與元件
  - [x] 9.1 實作 `getNodeColor(node: SubgraphNode): string` 純函數，依 status/risk_score 回傳顏色
    - status=1 → `#ef4444`（紅）；risk_score ≥ 0.5 && status ≠ 1 → `#f97316`（橘）；其他 → `#3b82f6`（藍）
    - _Requirements: 4.2_
  - [ ]* 9.2 為節點顏色映射撰寫屬性測試
    - **Property 4: 節點顏色映射正確性**
    - **Validates: Requirements 4.2**
  - [x] 9.3 實作 `getLinkDash(edge: SubgraphEdge): number[]` 純函數，依 relation_type 回傳 linkLineDash 陣列
    - R2 → `[]`（實線）；R1 → `[4, 2]`（虛線）；R3 → `[1, 2]`（點線）
    - _Requirements: 4.3_
  - [ ]* 9.4 為邊樣式映射撰寫屬性測試
    - **Property 5: 邊樣式映射正確性**
    - **Validates: Requirements 4.3**
  - [x] 9.5 建立 `components/graph/GraphViewer.tsx`，使用 `react-force-graph-2d` 渲染 subgraph
    - 套用 `getNodeColor`、`getLinkDash`；設定 `minZoom=0.1`、`maxZoom=5`；實作 `onNodeClick` 更新 selectedNode；提供「重置視角」按鈕；節點數 > 200 顯示警告
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8_
  - [ ]* 9.6 為節點點擊行為撰寫單元測試
    - **Property 6: 節點點擊觸發詳細資訊載入**
    - **Validates: Requirements 4.5**

- [x] 10. 實作節點詳細資訊面板
  - [x] 10.1 實作 `getShapColor(contribution: number): string` 純函數，正值 → 紅色、負值 → 綠色、零值 → 灰色
    - _Requirements: 5.2_
  - [ ]* 10.2 為 SHAP 顏色映射撰寫屬性測試
    - **Property 8: SHAP 貢獻值顏色映射正確性**
    - **Validates: Requirements 5.2**
  - [x] 10.3 建立 `components/graph/NodeDetailPanel.tsx`，顯示 user_id、risk_score、status、account_age_days、SHAP top-3（含顏色）、鄰居數量（R1/R2/R3）、「設為中心節點」按鈕
    - SHAP 資料不存在時顯示「SHAP 資料不可用」
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
  - [ ]* 10.4 為 NodeDetailPanel 撰寫屬性測試
    - **Property 7: NodeDetailPanel 正確渲染節點資訊**
    - **Validates: Requirements 5.1, 5.3**
  - [ ]* 10.5 為「設為中心節點」行為撰寫屬性測試
    - **Property 9: 設為中心節點更新選取狀態**
    - **Validates: Requirements 5.4**

- [x] 11. 實作主佈局與整合
  - [x] 11.1 建立 `components/layout/Dashboard.tsx`，組合雙欄佈局（左 25% StatsPanel / 右 75% 主區域）
    - 右側依序排列：NodeSelector、GraphViewer、NodeDetailPanel
    - 小於 1024px 切換為單欄；頁面頂部顯示標題「BitoGuard 詐騙偵測儀表板」
    - _Requirements: 1.1, 1.2, 1.3_
  - [x] 11.2 在 `App.tsx` 中掛載 `DashboardProvider` 與 `Dashboard`，頁面初始載入時並行發送 stats 與 fraud-nodes 請求
    - _Requirements: 1.4, 7.2_
  - [ ]* 11.3 為 API 錯誤隔離撰寫屬性測試
    - **Property 11: API 錯誤隔離不影響其他區塊**
    - **Validates: Requirements 7.3**
  - [ ]* 11.4 為 Loading 狀態撰寫屬性測試
    - **Property 13: Loading 狀態顯示 Spinner**
    - **Validates: Requirements 7.1**

- [x] 12. Checkpoint — 確認所有測試通過
  - 確認所有測試通過，若有問題請向使用者確認。

- [x] 13. 驗證 Fraud Nodes 排序與快取行為
  - [x] 13.1 確認 `getFraudNodes()` 回傳的陣列已依 risk_score 降序排列
    - _Requirements: 6.2_
  - [ ]* 13.2 為 Fraud Nodes 排序撰寫屬性測試
    - **Property 10: Fraud Nodes API 回傳降序排列**
    - **Validates: Requirements 6.2**
  - [ ]* 13.3 為 Subgraph 快取撰寫整合測試
    - **Property 12: Subgraph 快取避免重複請求**
    - **Validates: Requirements 7.4**

- [x] 14. Final Checkpoint — 確認所有測試通過
  - 確認所有測試通過，若有問題請向使用者確認。

## Notes

- 標記 `*` 的子任務為選填，可跳過以加速 MVP 開發
- 每個任務均對應需求文件中的具體驗收條件，確保可追溯性
- API Service 層（`api/`）為未來替換真實後端的唯一修改點
- 屬性測試使用 `fast-check`，每個屬性至少執行 100 次迭代
- 單元測試使用 `Vitest` + `@testing-library/react`
