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
  // R1=轉出錢包(wallet_funds_user) R2=帳戶轉帳戶(user_transfers_user)
  // R3=轉入錢包(user_sends_wallet)  R4=共用錢包(derived: 2 users share same wallet)
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

export interface NeighborPeer {
  peer_id: number;
  node_type: 'user' | 'wallet';
  node_label?: string;
  risk_score: number;
  status: 0 | 1;
  tx_count: number; // number of edges between this node and center
}

export interface NodeDetailResponse {
  user_id: number;
  risk_score: number;
  status: 0 | 1;
  account_age_days: number;
  shap_features: ShapFeature[];
  neighbor_counts: { r1: number; r2: number; r3: number }; // unique peer counts
  neighbor_details: {
    r1: NeighborPeer[];     // 錢包→帳戶 (wallets funding this user)
    r2_out: NeighborPeer[]; // 帳戶→帳戶 outgoing (this user → peer)
    r2_in: NeighborPeer[];  // 帳戶→帳戶 incoming (peer → this user)
    r3: NeighborPeer[];     // 帳戶→錢包 (wallets this user sent to)
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

// ── Predict node types ───────────────────────────────────────────────────────
export interface PredictNode {
  user_id: number;
  risk_score: number;
  is_blacklist: 0 | 1;
  shap_features: ShapFeature[];  // top-10 SHAP from predict_detail.csv
}

export type DashboardMode = 'fraud' | 'fp-fn' | 'predict' | 'features';
export type FpFnMode = 'fp' | 'fn';

// Global dashboard state (DashboardContext)
export interface DashboardState {
  stats: StatsResponse | null;
  fraudNodes: FraudNode[];
  selectedUserId: number | null;
  selectedWalletId: string | null;
  subgraph: SubgraphResponse | null;
  selectedNode: NodeDetailResponse | null;
  subgraphCache: Map<number, SubgraphResponse>;
  dashboardMode: DashboardMode;
  fpFnMode: FpFnMode;
  fpNodes: FpFnNode[];
  fnNodes: FpFnNode[];
  predictNodes: PredictNode[];
  shapWaterfall: ShapWaterfallResponse | null;
  loading: {
    stats: boolean;
    fraudNodes: boolean;
    subgraph: boolean;
    nodeDetail: boolean;
    fpFnNodes: boolean;
    predictNodes: boolean;
    shapWaterfall: boolean;
  };
  error: {
    stats: string | null;
    fraudNodes: string | null;
    subgraph: string | null;
    nodeDetail: string | null;
    fpFnNodes: string | null;
    predictNodes: string | null;
    shapWaterfall: string | null;
  };
}
