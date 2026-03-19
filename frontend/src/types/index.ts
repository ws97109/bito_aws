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

// Global dashboard state (DashboardContext)
export interface DashboardState {
  stats: StatsResponse | null;
  fraudNodes: FraudNode[];
  selectedUserId: number | null;
  subgraph: SubgraphResponse | null;
  selectedNode: NodeDetailResponse | null;
  subgraphCache: Map<number, SubgraphResponse>;
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
