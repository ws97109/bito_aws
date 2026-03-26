# Design Document: BitoGuard 詐騙偵測儀表板

## Overview

BitoGuard 詐騙偵測儀表板是一個純前端 React + TypeScript 應用，供風控分析師視覺化監控加密貨幣交易網路中的詐騙節點。系統整合 PC-GNN / GraphSAGE 模型輸出，以雙欄式佈局呈現統計摘要與互動式交易網路圖。

目前階段為純前端開發，使用 mock data 模擬後端回應，所有 API 呼叫集中於 service 層，未來可直接替換為真實 HTTP 請求。

### 技術選型

- **框架**: React 18 + TypeScript
- **圖形視覺化**: `react-force-graph-2d`（基於 D3-force，支援 2D 互動式圖形，API 簡潔）
- **圖表**: `recharts`（統計長條圖）
- **樣式**: Tailwind CSS
- **狀態管理**: React Context + useReducer（輕量，無需 Redux）
- **Mock 資料**: 靜態 JSON 模組，透過 service 層注入

---

## Architecture

```
src/
├── api/                    # API service 層（未來替換為真實 HTTP）
│   ├── client.ts           # axios/fetch 基礎設定
│   ├── statsApi.ts         # GET /api/stats
│   ├── fraudNodesApi.ts    # GET /api/fraud-nodes
│   ├── subgraphApi.ts      # GET /api/subgraph/{user_id}
│   └── nodeApi.ts          # GET /api/node/{user_id}
├── mock/                   # Mock 資料（模擬後端回應）
│   ├── statsData.ts
│   ├── fraudNodesData.ts
│   ├── subgraphData.ts
│   └── nodeData.ts
├── components/
│   ├── layout/
│   │   └── Dashboard.tsx   # 雙欄佈局容器
│   ├── stats/
│   │   ├── StatsPanel.tsx  # 左側統計面板
│   │   ├── StatCard.tsx    # 統計卡片
│   │   ├── RiskBarChart.tsx # 風險分布長條圖
│   │   └── RelationStats.tsx # R1/R2/R3 統計
│   ├── graph/
│   │   ├── NodeSelector.tsx # 下拉式節點選擇器
│   │   ├── GraphViewer.tsx  # 交易網路圖
│   │   └── NodeDetailPanel.tsx # 節點詳細資訊
│   └── common/
│       ├── Spinner.tsx
│       └── ErrorMessage.tsx
├── context/
│   └── DashboardContext.tsx # 全域狀態管理
├── hooks/
│   ├── useStats.ts
│   ├── useFraudNodes.ts
│   └── useSubgraph.ts
└── types/
    └── index.ts            # 共用型別定義
```

### 資料流

```
Mock Data → API Service Layer → Custom Hooks → Context → Components
```

未來替換路徑：只需修改 `api/` 層，將 mock import 替換為 fetch/axios 呼叫。

---

## Components and Interfaces

### Dashboard（雙欄佈局）

```tsx
// 左 25% / 右 75%，< 1024px 切換為單欄
<div className="flex flex-col lg:flex-row">
  <StatsPanel className="w-full lg:w-1/4" />
  <main className="w-full lg:w-3/4">
    <NodeSelector />
    <GraphViewer />
    <NodeDetailPanel />
  </main>
</div>
```

### NodeSelector

- 使用 `react-select` 或原生 `<select>` + 搜尋 input
- 300ms debounce 過濾
- 選項格式：`{user_id} | 風險分數: {Risk_Score}`
- 僅顯示 `status=1 && Risk_Score >= 0.5` 的節點

### GraphViewer

- 使用 `react-force-graph-2d`
- 節點顏色：紅色（status=1）、橘色（Risk_Score ≥ 0.5）、藍色（其他）
- 邊樣式：透過 `linkLineDash` 區分 R1/R2/R3
- 縮放範圍：0.1x ~ 5x（`minZoom` / `maxZoom` props）
- 點擊節點觸發 `onNodeClick` → 更新 Context 中的 `selectedNode`

### NodeDetailPanel

- 顯示 `user_id`、`Risk_Score`、`status`、`account_age_days`
- SHAP top-3：正值紅色、負值綠色
- 「設為中心節點」按鈕 → 重新 fetch subgraph

---

## Data Models

### API 回應型別

```typescript
// GET /api/stats
interface StatsResponse {
  total_nodes: number;
  fraud_nodes: number;
  normal_nodes: number;
  fraud_ratio: number;           // 0~1
  risk_distribution: {
    range: string;               // "[0, 0.2)"
    count: number;
  }[];
  relation_counts: {
    r1: number;
    r2: number;
    r3: number;
  };
}

// GET /api/fraud-nodes
interface FraudNode {
  user_id: number;
  risk_score: number;
}
type FraudNodesResponse = FraudNode[];

// GET /api/subgraph/{user_id}?hops=2
interface SubgraphNode {
  user_id: number;
  risk_score: number;
  status: 0 | 1;
}
interface SubgraphEdge {
  source: number;
  target: number;
  relation_type: 'R1' | 'R2' | 'R3';
}
interface SubgraphResponse {
  nodes: SubgraphNode[];
  edges: SubgraphEdge[];
}

// GET /api/node/{user_id}
interface ShapFeature {
  feature_name: string;
  contribution: number;          // 正值 = 增加風險，負值 = 降低風險
}
interface NodeDetailResponse {
  user_id: number;
  risk_score: number;
  status: 0 | 1;
  account_age_days: number;
  shap_features: ShapFeature[];  // top-3，可為空陣列
  neighbor_counts: {
    r1: number;
    r2: number;
    r3: number;
  };
}
```

### 全域狀態（DashboardContext）

```typescript
interface DashboardState {
  stats: StatsResponse | null;
  fraudNodes: FraudNode[];
  selectedUserId: number | null;
  subgraph: SubgraphResponse | null;
  selectedNode: NodeDetailResponse | null;
  subgraphCache: Map<number, SubgraphResponse>;  // 快取
  loading: {
    stats: boolean;
    fraudNodes: boolean;
    subgraph: boolean;
    nodeDetail: boolean;
  };
  error: {
    stats: string | null;
    fraudNodes: string | null;
    subgraph: string | null;
    nodeDetail: string | null;
  };
}
```

### Mock 資料結構

Mock 資料模擬真實 API 回應格式，包含：
- 約 500 個節點（其中 ~50 個 status=1）
- 節點間的 R1/R2/R3 邊關係
- 每個節點的 SHAP top-3 特徵（基於 `user_features.csv` 欄位名稱）

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*


### Property 1: StatsPanel 正確渲染所有統計資訊

*For any* valid `StatsResponse`，渲染後的 StatsPanel 應包含：總節點數、詐騙節點數、正常節點數、詐騙比例，以及五個風險區間的長條圖資料，以及 R1/R2/R3 的邊數量。

**Validates: Requirements 2.1, 2.2, 2.3**

### Property 2: NodeSelector 過濾邏輯正確性

*For any* 節點清單，NodeSelector 中顯示的選項應只包含 `status=1 && risk_score >= 0.5` 的節點；且當輸入關鍵字時，結果應只包含 `user_id` 字串中含有該關鍵字的節點。

**Validates: Requirements 3.2, 3.3**

### Property 3: NodeSelector 選項格式正確性

*For any* FraudNode，其在選單中的顯示文字應符合格式 `{user_id} | 風險分數: {risk_score}`，且 risk_score 格式化為兩位小數。

**Validates: Requirements 3.4**

### Property 4: 節點顏色映射正確性

*For any* SubgraphNode，顏色映射函數應回傳：`status=1` → 紅色、`risk_score >= 0.5 && status != 1` → 橘色、`risk_score < 0.5` → 藍色，且三個條件互斥。

**Validates: Requirements 4.2**

### Property 5: 邊樣式映射正確性

*For any* SubgraphEdge，樣式映射函數應回傳：`R2` → 實線（空陣列）、`R1` → 虛線、`R3` → 點線，且每種 relation_type 對應唯一的線條樣式。

**Validates: Requirements 4.3**

### Property 6: 節點點擊觸發詳細資訊載入

*For any* 圖中節點，點擊後應觸發 `GET /api/node/{user_id}` 請求，且 selectedNode 狀態應更新為對應節點的詳細資訊。

**Validates: Requirements 4.5**

### Property 7: NodeDetailPanel 正確渲染節點資訊

*For any* valid `NodeDetailResponse`，渲染後的 NodeDetailPanel 應包含 `user_id`、`risk_score`、`status`、`account_age_days`，以及依 R1/R2/R3 分類的鄰居數量。

**Validates: Requirements 5.1, 5.3**

### Property 8: SHAP 貢獻值顏色映射正確性

*For any* SHAP 特徵貢獻值，顏色映射函數應回傳：正值 → 紅色、負值 → 綠色，且零值有明確的預設顏色。

**Validates: Requirements 5.2**

### Property 9: 設為中心節點更新選取狀態

*For any* 節點詳細資訊面板中的節點，點擊「設為中心節點」後，`selectedUserId` 狀態應更新為該節點的 `user_id`，並觸發 subgraph 重新載入。

**Validates: Requirements 5.4**

### Property 10: Fraud Nodes API 回傳降序排列

*For any* fraud nodes 清單，回傳的陣列應按 `risk_score` 降序排列（即 `list[i].risk_score >= list[i+1].risk_score` 對所有有效索引成立）。

**Validates: Requirements 6.2**

### Property 11: API 錯誤隔離不影響其他區塊

*For any* 單一 API 請求失敗的情況，其他 API 請求的 loading/data 狀態應不受影響，對應區塊應正常顯示。

**Validates: Requirements 7.3**

### Property 12: Subgraph 快取避免重複請求

*For any* user_id，若已載入過其 subgraph，重複選取同一節點時，API 呼叫次數應仍為 1（使用快取資料）。

**Validates: Requirements 7.4**

### Property 13: Loading 狀態顯示 Spinner

*For any* API 請求進行中的狀態（`loading.{key} = true`），對應區塊應顯示 spinner 元件，且在請求完成後 spinner 應消失。

**Validates: Requirements 7.1**

---

## Error Handling

| 情境 | 處理方式 |
|------|---------|
| Stats API 失敗 | StatsPanel 顯示錯誤訊息 + 重試按鈕，不影響右側 |
| Fraud Nodes API 失敗 | NodeSelector 顯示錯誤訊息，圖形區域不受影響 |
| Subgraph API 失敗 | GraphViewer 顯示錯誤訊息 + 重試按鈕 |
| Node Detail API 失敗 | NodeDetailPanel 顯示錯誤訊息 |
| user_id 不存在（404） | 顯示「找不到該節點」錯誤訊息 |
| 網路逾時 | 顯示「請求逾時，請稍後再試」錯誤訊息 |
| Subgraph 節點數 > 200 | 顯示警告，自動降級為 1-hop |
| SHAP 資料不存在 | 顯示「SHAP 資料不可用」提示 |

所有錯誤訊息透過 `DashboardContext` 的 `error` 狀態管理，各區塊獨立顯示，互不干擾。

---

## Testing Strategy

### 雙軌測試方法

**單元測試（Unit Tests）**：驗證具體範例、邊界條件與錯誤處理
- 工具：`Vitest` + `@testing-library/react`
- 重點：
  - 初始渲染狀態（標題、預設提示文字）
  - 錯誤狀態 UI（錯誤訊息、重試按鈕）
  - 邊界條件（空清單、SHAP 不存在、節點數 > 200）
  - 並行 API 請求（初始載入）
  - 60 秒自動刷新（使用 `vi.useFakeTimers()`）

**屬性測試（Property-Based Tests）**：驗證對所有輸入都成立的通用屬性
- 工具：`fast-check`
- 最少 100 次迭代
- 每個屬性測試對應設計文件中的一個 Property

### 屬性測試配置

```typescript
// 每個屬性測試的標籤格式
// Feature: fraud-detection-dashboard, Property {N}: {property_text}

import fc from 'fast-check';

// 範例：Property 4 - 節點顏色映射
// Feature: fraud-detection-dashboard, Property 4: 節點顏色映射正確性
test('node color mapping is correct for all nodes', () => {
  fc.assert(
    fc.property(
      fc.record({
        user_id: fc.integer({ min: 1 }),
        risk_score: fc.float({ min: 0, max: 1 }),
        status: fc.constantFrom(0, 1) as fc.Arbitrary<0 | 1>,
      }),
      (node) => {
        const color = getNodeColor(node);
        if (node.status === 1) return color === '#ef4444'; // red
        if (node.risk_score >= 0.5) return color === '#f97316'; // orange
        return color === '#3b82f6'; // blue
      }
    ),
    { numRuns: 100 }
  );
});
```

### 測試覆蓋目標

| 層級 | 工具 | 目標 |
|------|------|------|
| 純函數（映射、格式化、過濾） | fast-check | Properties 1-13 |
| 元件渲染 | @testing-library/react | 具體範例與邊界條件 |
| 狀態管理 | Vitest | Context reducer 邏輯 |
| API Service 層 | Vitest + msw | Mock API 回應格式驗證 |
