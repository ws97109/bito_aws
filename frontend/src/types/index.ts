// GET /api/stats
export interface StatsResponse {
  total_nodes: number;
  fraud_nodes: number;
  normal_nodes: number;
  fraud_ratio: number; // 0~1
  risk_distribution: {
    range: string; // e.g. "[0, 0.2)"
    count: number;
  }[];
  relation_counts: {
    r1: number;
    r2: number;
    r3: number;
  };
}

// GET /api/fraud-nodes
export interface FraudNode {
  user_id: number;
  risk_score: number;
}

export type FraudNodesResponse = FraudNode[];

// GET /api/subgraph/{user_id}?hops=2
export interface SubgraphNode {
  user_id: number;
  risk_score: number;
  status: 0 | 1;
  node_type?: 'user' | 'wallet';
  node_label?: string; // original string ID (e.g. 'user_820487', 'wallet_abc123')
}

export interface SubgraphEdge {
  source: number;
  target: number;
  relation_type: 'R1' | 'R2' | 'R3';
}

export interface SubgraphResponse {
  nodes: SubgraphNode[];
  edges: SubgraphEdge[];
}

// GET /api/node/{user_id}
export interface ShapFeature {
  feature_name: string;
  contribution: number; // positive = increases risk, negative = decreases risk
}

export interface NodeDetailResponse {
  user_id: number;
  risk_score: number;
  status: 0 | 1;
  account_age_days: number;
  shap_features: ShapFeature[]; // top-3, may be empty
  neighbor_counts: {
    r1: number;
    r2: number;
    r3: number;
  };
}

// ── SHAP Waterfall ────────────────────────────────────────────────────────────
// GET /api/shap/waterfall?mode=fp|fn[&user_id=xxx]
// Backend reads shap_values CSV, filters for the target group, returns top-N.
export interface ShapWaterfallFeature {
  feature_name: string;   // human-readable column name
  feature_value: string;  // actual value for this prediction (e.g. "2174", "高")
  contribution: number;   // SHAP value in log-odds space
}

export interface ShapWaterfallResponse {
  mode: 'fp' | 'fn';
  user_id: number | null;  // null → group average
  base_value: number;      // E[f(x)] — model expected output
  features: ShapWaterfallFeature[];
}

// ── FP/FN node types ──────────────────────────────────────────────────────────
export interface FpFnNode {
  user_id: number;
  risk_score: number;
  actual_status: 0 | 1;
  predicted_status: 0 | 1;
}

export type DashboardMode = 'fraud' | 'fp-fn';
export type FpFnMode = 'fp' | 'fn';

// Global dashboard state (DashboardContext)
export interface DashboardState {
  stats: StatsResponse | null;
  fraudNodes: FraudNode[];
  selectedUserId: number | null;
  subgraph: SubgraphResponse | null;
  selectedNode: NodeDetailResponse | null;
  subgraphCache: Map<number, SubgraphResponse>;
  dashboardMode: DashboardMode;
  fpFnMode: FpFnMode;
  fpNodes: FpFnNode[];
  fnNodes: FpFnNode[];
  shapWaterfall: ShapWaterfallResponse | null;
  loading: {
    stats: boolean;
    fraudNodes: boolean;
    subgraph: boolean;
    nodeDetail: boolean;
    fpFnNodes: boolean;
    shapWaterfall: boolean;
  };
  error: {
    stats: string | null;
    fraudNodes: string | null;
    subgraph: string | null;
    nodeDetail: string | null;
    fpFnNodes: string | null;
    shapWaterfall: string | null;
  };
}
