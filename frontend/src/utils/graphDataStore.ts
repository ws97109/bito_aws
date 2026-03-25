/**
 * graphDataStore – loads and indexes all CSV data from /output/.
 *
 * CSV files (served by Vite plugin at /output/<file>.csv):
 *   gnn_node_list.csv   → node_id, node_type, risk_score, label
 *   gnn_edge_list.csv   → source, target, source_raw, target_raw, edge_type
 *   blacklist_analysis.csv → user_id, risk_score  (fraud detection / blacklist)
 *   black_to_white.csv  → user_id, risk_score  (FP: predicted fraud, actually normal)
 *   white_to_black.csv  → user_id, risk_score  (FN: predicted normal, actually fraud)
 *   features.csv        → user_id + 60+ feature columns (real feature values per user)
 *
 * Edge-type → relation mapping:
 *   user_transfers_user → R2 (加密貨幣內轉)
 *   user_sends_wallet   → R3 (共用錢包)
 *   wallet_funds_user   → R1 (共用錢包反向 / R1)
 */

import { parseCsvRecords } from './csvParser';
import type { StatsResponse, FraudNode, FpFnNode, SubgraphResponse, SubgraphNode, SubgraphEdge, NodeDetailResponse, ShapWaterfallResponse, ShapWaterfallFeature } from '../types/index';

// ── Internal types ────────────────────────────────────────────────────────────

interface NodeRecord {
  nodeId: string;         // 'user_4'
  nodeType: 'user' | 'wallet';
  riskScore: number;
  label: number;          // 0.0 | 1.0
  numericId: number;      // for user: parsed integer; for wallet: negative index
}

interface EdgeRecord {
  source: string;
  target: string;
  sourceRaw: string;
  targetRaw: string;
  edgeType: 'user_sends_wallet' | 'user_transfers_user' | 'wallet_funds_user';
}

type RelationType = 'R1' | 'R2' | 'R3';

const EDGE_TO_RELATION: Record<string, RelationType> = {
  wallet_funds_user:   'R1', // 錢包→帳戶 (轉出錢包)
  user_transfers_user: 'R2', // 帳戶→帳戶
  user_sends_wallet:   'R3', // 帳戶→錢包 (轉入錢包)
};

// ── Module-level cache ────────────────────────────────────────────────────────

let nodeMap: Map<string, NodeRecord> | null = null;       // nodeId → NodeRecord
let userIdMap: Map<number, NodeRecord> | null = null;     // numericId → NodeRecord (users only)
let adjMap: Map<string, EdgeRecord[]> | null = null;      // nodeId → edges (both directions)
let allEdges: EdgeRecord[] | null = null;

let blacklistCache: FraudNode[] | null = null;
let fpCache: FpFnNode[] | null = null;
let fnCache: FpFnNode[] | null = null;

// features.csv: user_id → Record<feature_name, raw_string_value>
let featuresMap: Map<number, Record<string, string>> | null = null;

// ── Loaders ───────────────────────────────────────────────────────────────────

async function fetchCsv(path: string): Promise<string> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to fetch ${path}: ${res.status}`);
  return res.text();
}

async function loadGraphData(): Promise<void> {
  if (nodeMap !== null) return; // already loaded

  const [nodeText, edgeText] = await Promise.all([
    fetchCsv('/output/gnn_node_list.csv'),
    fetchCsv('/output/gnn_edge_list.csv'),
  ]);

  // Build node map
  const nm = new Map<string, NodeRecord>();
  const um = new Map<number, NodeRecord>();
  let walletIndex = -1;

  for (const row of parseCsvRecords(nodeText)) {
    const nodeId = row['node_id'] ?? '';
    const nodeType = row['node_type'] === 'wallet' ? 'wallet' : 'user';
    const riskScore = parseFloat(row['risk_score'] ?? '0');
    const label = parseFloat(row['label'] ?? '0');
    let numericId: number;

    if (nodeType === 'user') {
      numericId = parseInt(nodeId.replace('user_', ''), 10);
    } else {
      numericId = walletIndex--;
    }

    const record: NodeRecord = { nodeId, nodeType, riskScore, label, numericId };
    nm.set(nodeId, record);
    if (nodeType === 'user') um.set(numericId, record);
  }

  // Build edge list and adjacency map
  const edges: EdgeRecord[] = [];
  const adj = new Map<string, EdgeRecord[]>();

  const addAdj = (key: string, edge: EdgeRecord) => {
    const list = adj.get(key);
    if (list) list.push(edge);
    else adj.set(key, [edge]);
  };

  for (const row of parseCsvRecords(edgeText)) {
    const edge: EdgeRecord = {
      source: row['source'] ?? '',
      target: row['target'] ?? '',
      sourceRaw: row['source_raw'] ?? '',
      targetRaw: row['target_raw'] ?? '',
      edgeType: (row['edge_type'] ?? 'user_transfers_user') as EdgeRecord['edgeType'],
    };
    edges.push(edge);
    addAdj(edge.source, edge);
    addAdj(edge.target, edge);

    // register wallet nodes that only appear in edge list
    if (edge.source.startsWith('wallet_') && !nm.has(edge.source)) {
      const wRec: NodeRecord = { nodeId: edge.source, nodeType: 'wallet', riskScore: 0, label: 0, numericId: walletIndex-- };
      nm.set(edge.source, wRec);
    }
    if (edge.target.startsWith('wallet_') && !nm.has(edge.target)) {
      const wRec: NodeRecord = { nodeId: edge.target, nodeType: 'wallet', riskScore: 0, label: 0, numericId: walletIndex-- };
      nm.set(edge.target, wRec);
    }
  }

  nodeMap = nm;
  userIdMap = um;
  allEdges = edges;
  adjMap = adj;
}

async function loadFeaturesData(): Promise<void> {
  if (featuresMap !== null) return;
  const text = await fetchCsv('/output/features.csv');
  const records = parseCsvRecords(text);
  const fm = new Map<number, Record<string, string>>();
  for (const row of records) {
    const uid = parseInt(row['user_id'] ?? '', 10);
    if (!isNaN(uid)) fm.set(uid, row);
  }
  featuresMap = fm;
}

// Priority features with weights for contribution approximation
const SHAP_FEATURE_CONFIG: Array<{ key: string; label: string; weight: number }> = [
  { key: 'composite_risk_score',  label: '綜合風險分數',       weight:  0.50 },
  { key: 'ip_unique_count',       label: '共用 IP 節點數',     weight:  0.001 },
  { key: 'crypto_wit_count',      label: '加密提幣次數',       weight:  0.02 },
  { key: 'twd_wit_sum',           label: '台幣提款總額',       weight:  0.000001 },
  { key: 'twd_wit_count',         label: '台幣提款次數',       weight:  0.015 },
  { key: 'account_age_days',      label: '帳戶年齡（天）',     weight: -0.0005 },
  { key: 'is_high_risk_career',   label: '高風險職業',         weight:  0.18 },
  { key: 'crypto_dep_sum',        label: '加密入金總額',       weight:  0.000001 },
  { key: 'twd_smurf_flag',        label: '台幣分散提款標記',   weight:  0.30 },
  { key: 'withdraw_velocity',     label: '提款速度',           weight:  0.04 },
];

export async function getShapForUser(
  mode: 'fp' | 'fn' | 'blacklist',
  userId?: number,
): Promise<ShapWaterfallResponse> {
  await loadFeaturesData();

  const shapMode: 'fp' | 'fn' = mode === 'fn' ? 'fn' : 'fp';

  if (userId != null && featuresMap !== null) {
    const row = featuresMap.get(userId);
    if (row) {
      const features: ShapWaterfallFeature[] = SHAP_FEATURE_CONFIG.map(cfg => {
        const raw = row[cfg.key] ?? '0';
        const numVal = parseFloat(raw) || 0;
        const contribution = parseFloat((numVal * cfg.weight).toFixed(3));
        return {
          feature_name: cfg.label,
          feature_value: raw,
          contribution,
        };
      });
      return {
        mode: shapMode,
        user_id: userId,
        base_value: -2.885,
        features,
      };
    }
  }

  // Fall back to mock data
  const { mockFpShap, mockFnShap } = await import('../mock/shapData');
  return shapMode === 'fp' ? mockFpShap : mockFnShap;
}

// ── Public API ────────────────────────────────────────────────────────────────

export async function getComputedStats(): Promise<StatsResponse> {
  await loadGraphData();
  const nm = nodeMap!;
  const edges = allEdges!;

  let fraudNodes = 0;
  let totalNodes = 0;
  const riskBuckets = [0, 0, 0, 0, 0]; // [0,0.2) [0.2,0.4) [0.4,0.6) [0.6,0.8) [0.8,1.0]

  for (const rec of nm.values()) {
    if (rec.nodeType !== 'user') continue;
    totalNodes++;
    if (rec.label >= 1.0) fraudNodes++;
    const idx = Math.min(Math.floor(rec.riskScore / 0.2), 4);
    riskBuckets[idx]++;
  }

  const relation_counts = { r1: 0, r2: 0, r3: 0 };
  for (const edge of edges) {
    const rel = EDGE_TO_RELATION[edge.edgeType];
    if (rel === 'R1') relation_counts.r1++;
    else if (rel === 'R2') relation_counts.r2++;
    else if (rel === 'R3') relation_counts.r3++;
  }

  return {
    total_nodes: totalNodes,
    fraud_nodes: fraudNodes,
    normal_nodes: totalNodes - fraudNodes,
    fraud_ratio: totalNodes > 0 ? fraudNodes / totalNodes : 0,
    risk_distribution: [
      { range: '[0, 0.2)',  count: riskBuckets[0] },
      { range: '[0.2, 0.4)', count: riskBuckets[1] },
      { range: '[0.4, 0.6)', count: riskBuckets[2] },
      { range: '[0.6, 0.8)', count: riskBuckets[3] },
      { range: '[0.8, 1.0]', count: riskBuckets[4] },
    ],
    relation_counts,
  };
}

export async function getBlacklistNodes(): Promise<FraudNode[]> {
  if (blacklistCache) return blacklistCache;

  // Load graph data so gnn_node_list is available
  await loadGraphData();

  const text = await fetchCsv('/output/blacklist_analysis.csv');
  const blacklistMap = new Map<number, number>(); // user_id → risk_score
  for (const r of parseCsvRecords(text)) {
    const uid = parseInt(r['user_id'] ?? '0', 10);
    const score = parseFloat(r['risk_score'] ?? '0');
    if (!isNaN(uid)) blacklistMap.set(uid, score);
  }

  // Merge with gnn_node_list fraud nodes (label=1) so graph nodes are always searchable
  for (const rec of nodeMap!.values()) {
    if (rec.nodeType === 'user' && rec.label >= 1.0 && !blacklistMap.has(rec.numericId)) {
      blacklistMap.set(rec.numericId, rec.riskScore);
    }
  }

  blacklistCache = Array.from(blacklistMap.entries())
    .map(([user_id, risk_score]) => ({ user_id, risk_score }))
    .sort((a, b) => b.risk_score - a.risk_score);
  return blacklistCache;
}

export async function getFpFnData(): Promise<{ fp: FpFnNode[]; fn: FpFnNode[] }> {
  if (fpCache && fnCache) return { fp: fpCache, fn: fnCache };

  const [fpText, fnText] = await Promise.all([
    fetchCsv('/output/black_to_white.csv'),
    fetchCsv('/output/white_to_black.csv'),
  ]);

  fpCache = parseCsvRecords(fpText)
    .map(r => ({
      user_id: parseInt(r['user_id'] ?? '0', 10),
      risk_score: parseFloat(r['risk_score'] ?? '0'),
      actual_status: 0 as const,
      predicted_status: 1 as const,
    }))
    .filter(n => !isNaN(n.user_id))
    .sort((a, b) => b.risk_score - a.risk_score);

  fnCache = parseCsvRecords(fnText)
    .map(r => ({
      user_id: parseInt(r['user_id'] ?? '0', 10),
      risk_score: parseFloat(r['risk_score'] ?? '0'),
      actual_status: 1 as const,
      predicted_status: 0 as const,
    }))
    .filter(n => !isNaN(n.user_id))
    .sort((a, b) => b.risk_score - a.risk_score);

  return { fp: fpCache, fn: fnCache };
}

export async function getComputedSubgraph(userId: number, hops: number = 2): Promise<SubgraphResponse> {
  await loadGraphData();
  const nm = nodeMap!;
  const adj = adjMap!;

  const centerKey = `user_${userId}`;
  if (!nm.has(centerKey)) {
    return { nodes: [], edges: [] };
  }

  // BFS to collect nodes and edges within `hops`
  const visitedNodes = new Set<string>();
  const visitedEdges = new Set<string>();
  const resultEdges: SubgraphEdge[] = [];

  let frontier = new Set<string>([centerKey]);
  visitedNodes.add(centerKey);

  for (let hop = 0; hop < hops; hop++) {
    const nextFrontier = new Set<string>();
    for (const nodeId of frontier) {
      const edges = adj.get(nodeId) ?? [];
      for (const edge of edges) {
        const edgeKey = `${edge.source}|${edge.target}|${edge.edgeType}`;
        if (visitedEdges.has(edgeKey)) continue;
        visitedEdges.add(edgeKey);

        const neighbor = edge.source === nodeId ? edge.target : edge.source;

        // Convert to numeric IDs for SubgraphEdge
        const srcRec = nm.get(edge.source);
        const tgtRec = nm.get(edge.target);
        if (!srcRec || !tgtRec) continue;

        resultEdges.push({
          source: srcRec.numericId,
          target: tgtRec.numericId,
          relation_type: EDGE_TO_RELATION[edge.edgeType] ?? 'R2',
        });

        if (!visitedNodes.has(neighbor)) {
          visitedNodes.add(neighbor);
          nextFrontier.add(neighbor);
        }
      }
    }
    frontier = nextFrontier;
    if (frontier.size === 0) break;
  }

  const resultNodes: SubgraphNode[] = [];
  for (const nodeId of visitedNodes) {
    const rec = nm.get(nodeId);
    if (!rec) continue;
    resultNodes.push({
      user_id: rec.numericId,
      risk_score: rec.riskScore,
      status: rec.label >= 1.0 ? 1 : 0,
      node_type: rec.nodeType,
      node_label: rec.nodeId,
    });
  }

  return { nodes: resultNodes, edges: resultEdges };
}

/**
 * Synchronous check — returns true once graph data is loaded and the user
 * exists in the GNN node list. Safe to call before loadGraphData() resolves
 * (returns false while still loading).
 */
export function hasGraphData(userId: number): boolean {
  if (!nodeMap) return false;
  return nodeMap.has(`user_${userId}`);
}

export async function getComputedNodeDetail(userId: number): Promise<NodeDetailResponse | null> {
  await loadGraphData();
  const rec = userIdMap!.get(userId);
  if (!rec) return null;

  const adj = adjMap!;
  const edges = adj.get(rec.nodeId) ?? [];

  const neighbor_counts = { r1: 0, r2: 0, r3: 0 };
  for (const edge of edges) {
    const rel = EDGE_TO_RELATION[edge.edgeType];
    if (rel === 'R1') neighbor_counts.r1++;
    else if (rel === 'R2') neighbor_counts.r2++;
    else if (rel === 'R3') neighbor_counts.r3++;
  }

  return {
    user_id: userId,
    risk_score: rec.riskScore,
    status: rec.label >= 1.0 ? 1 : 0,
    account_age_days: 0,
    shap_features: [],
    neighbor_counts,
  };
}
