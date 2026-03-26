/**
 * graphDataStore – loads and indexes all CSV data from /output/.
 *
 * CSV files (served by Vite plugin at /output/<file>.csv):
 *   gnn_node_list.csv   → node_id, node_type, risk_score, label
 *   gnn_edge_list.csv   → source, target, source_raw, target_raw, edge_type
 *   blacklist_analysis.csv → user_id, risk_score  (confirmed blacklist, 黑名單 sheet, 328 rows)
 *   black_to_white.csv  → user_id, risk_score  (FP: 白預測成黑, 286 rows)
 *   white_to_black.csv  → user_id, risk_score  (FN: 黑預測成白, 193 rows)
 *   features.csv        → user_id + feature columns (from predict_detail.csv, 12,753 rows)
 *   shap_values.csv     → user_id + SHAP columns, top-10 non-null per user (63,770 rows)
 *
 * Edge-type → relation mapping:
 *   user_transfers_user → R2 (加密貨幣內轉)
 *   user_sends_wallet   → R3 (共用錢包)
 *   wallet_funds_user   → R1 (共用錢包反向 / R1)
 */

import { parseCsvRecords } from './csvParser';
import type { StatsResponse, FraudNode, FpFnNode, PredictNode, SubgraphResponse, SubgraphNode, SubgraphEdge, NodeDetailResponse, NeighborPeer, ShapWaterfallResponse, ShapWaterfallFeature, ShapFeature } from '../types/index';

// ── 特徵英文 → 中文對照表 ──────────────────────────────────────────────────────
const FEATURE_NAME_ZH: Record<string, string> = {
  // 用戶基本特徵
  kyc_speed_sec:           'KYC 完成速度（秒）',
  account_age_days:        '帳號年齡（天）',
  age:                     '年齡',
  is_female:               '是否女性',
  is_high_risk_career:     '高風險職業',
  is_high_risk_income:     '高風險收入來源',
  career_income_risk:      '職業×收入組合風險',
  career_freq:             '職業頻率',
  is_app_user:             'APP 用戶',
  reg_hour:                '註冊時間（時）',
  reg_is_night:            '深夜註冊',
  reg_is_weekend:          '週末註冊',
  has_kyc_level2:          '已完成 KYC2',
  kyc_gap_days:            'KYC 間隔（天）',
  reg_to_kyc1_days:        '註冊到 KYC1（天）',
  // 法幣行為
  twd_dep_count:           '法幣入金次數',
  twd_dep_sum:             '法幣入金總額',
  twd_dep_mean:            '法幣入金均值',
  twd_dep_std:             '法幣入金標準差',
  twd_dep_max:             '法幣入金最大值',
  twd_wit_count:           '法幣提領次數',
  twd_wit_sum:             '法幣提領總額',
  twd_wit_mean:            '法幣提領均值',
  twd_wit_std:             '法幣提領標準差',
  twd_wit_max:             '法幣提領最大值',
  twd_net_flow:            '法幣淨流入',
  twd_withdraw_ratio:      '提領/入金比',
  twd_smurf_flag:          '結構化交易旗標',
  twd_wit_ip_ratio:        '提領 IP 覆蓋率',
  // 虛擬貨幣行為
  crypto_dep_count:        '加密入金次數',
  crypto_dep_sum:          '加密入金總額',
  crypto_dep_mean:         '加密入金均值',
  crypto_dep_max:          '加密入金最大值',
  crypto_wit_count:        '加密提領次數',
  crypto_wit_sum:          '加密提領總額',
  crypto_wit_mean:         '加密提領均值',
  crypto_wit_max:          '加密提領最大值',
  crypto_currency_diversity:  '使用幣種數',
  crypto_protocol_diversity:  '使用鏈協定數',
  crypto_wallet_hash_nunique: '錢包地址數',
  crypto_internal_count:      '內轉次數',
  crypto_internal_peer_count: '內轉對象數',
  crypto_external_wit_count:  '鏈上提領次數',
  crypto_wit_ip_ratio:        '加密提領 IP 覆蓋率',
  // 交易行為
  trading_count:           '掛單成交次數',
  trading_sum:             '掛單成交總額',
  trading_mean:            '掛單成交均值',
  trading_max:             '掛單成交最大值',
  trading_buy_ratio:       '買單比率',
  trading_market_order_ratio: '市價單比率',
  swap_count:              '一鍵買賣次數',
  swap_sum:                '一鍵買賣總額',
  total_trading_volume:    '總交易量',
  // IP 特徵
  ip_unique_count:         '唯一 IP 數',
  ip_total_count:          '總 IP 使用次數',
  ip_night_ratio:          '深夜操作 IP 比率',
  ip_max_shared:           'IP 最大共用人數',
  // 資金停留
  fund_stay_sec:           '資金停留時間（秒）',
  // 圖特徵
  pagerank_score:          'PageRank 分數',
  graph_in_degree:         '圖入度',
  graph_out_degree:        '圖出度',
  connected_component_size: '連通分量大小',
  betweenness_centrality:  '介數中心性',
  // 跨表整合
  total_tx_count:          '總交易筆數',
  first_to_last_tx_days:   '首末交易間隔（天）',
  weekend_tx_ratio:        '週末交易比率',
  velocity_ratio_7d_vs_30d: '7天/30天交易加速比',
  // 紅旗特徵
  dep_to_first_wit_hours:  '入金到首次提領（時）',
  twd_to_crypto_out_ratio: '法幣入/幣出比',
  tx_amount_cv:            '交易金額變異係數',
  rapid_kyc_then_trade:    'KYC 後 48h 交易',
  crypto_out_in_ratio:     '加密出/入比',
  same_day_in_out_count:   '同天入出金天數',
  // 時序特徵
  tx_interval_mean:        '交易間隔均值（秒）',
  tx_interval_std:         '交易間隔標準差',
  tx_interval_min:         '交易最短間隔（秒）',
  tx_interval_median:      '交易間隔中位數',
  tx_burst_count:          '交易爆發次數',
  amount_p90_p10_ratio:    '金額 P90/P10 比',
  active_days:             '活躍天數',
  active_day_ratio:        '活躍天數比',
  // 異常偵測分數
  if_score:                '孤立森林分數',
  hbos_score:              'HBOS 分數',
  lof_score:               'LOF 分數',
  // 複合分數
  composite_risk_score:    '複合風險分數',
  // GNN 嵌入
  gnn_emb_0:  'GNN 嵌入 0',  gnn_emb_1:  'GNN 嵌入 1',
  gnn_emb_2:  'GNN 嵌入 2',  gnn_emb_3:  'GNN 嵌入 3',
  gnn_emb_4:  'GNN 嵌入 4',  gnn_emb_5:  'GNN 嵌入 5',
  gnn_emb_6:  'GNN 嵌入 6',  gnn_emb_7:  'GNN 嵌入 7',
  gnn_emb_8:  'GNN 嵌入 8',  gnn_emb_9:  'GNN 嵌入 9',
  gnn_emb_10: 'GNN 嵌入 10', gnn_emb_11: 'GNN 嵌入 11',
  gnn_emb_12: 'GNN 嵌入 12', gnn_emb_13: 'GNN 嵌入 13',
  gnn_emb_14: 'GNN 嵌入 14', gnn_emb_15: 'GNN 嵌入 15',
};

/** 將英文特徵名轉為中文，找不到則回傳原名 */
function zhFeatureName(eng: string): string {
  return FEATURE_NAME_ZH[eng] ?? eng;
}

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
let predictCache: PredictNode[] | null = null;

// features.csv: user_id → Record<feature_name, raw_string_value>
let featuresMap: Map<number, Record<string, string>> | null = null;

// shap_values.csv: user_id → Record<feature_name, shap_float_string> (empty string = null/not in top-10)
let shapMap: Map<number, Record<string, string>> | null = null;

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
    const riskScore = parseFloat(row['risk_score'] || '0');
    const label = parseFloat(row['label'] || '0');
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
  try {
    const text = await fetchCsv('/output/features.csv');
    const records = parseCsvRecords(text);
    const fm = new Map<number, Record<string, string>>();
    for (const row of records) {
      const uid = parseInt(row['user_id'] ?? '', 10);
      if (!isNaN(uid)) fm.set(uid, row);
    }
    featuresMap = fm;
  } catch {
    // features.csv is optional; SHAP waterfall will omit raw feature values
    featuresMap = new Map();
  }
}

async function loadShapData(): Promise<void> {
  if (shapMap !== null) return;
  const text = await fetchCsv('/output/shap_values_all_top10.csv');
  const records = parseCsvRecords(text);
  const sm = new Map<number, Record<string, string>>();
  for (const row of records) {
    const uid = parseInt(row['user_id'] ?? '', 10);
    if (!isNaN(uid)) sm.set(uid, row);
  }
  shapMap = sm;
}

export async function getShapForUser(
  mode: 'fp' | 'fn' | 'blacklist',
  userId?: number,
): Promise<ShapWaterfallResponse> {
  await Promise.all([loadFeaturesData(), loadShapData()]);

  const shapMode: 'fp' | 'fn' = mode === 'fn' ? 'fn' : 'fp';

  if (userId != null && shapMap !== null) {
    const shapRow = shapMap.get(userId);
    if (shapRow) {
      const featureRow = featuresMap?.get(userId) ?? {};

      // Collect non-empty SHAP columns (these are the top-10 features for this user)
      const features: ShapWaterfallFeature[] = [];
      for (const [colName, shapVal] of Object.entries(shapRow)) {
        if (colName === 'user_id' || shapVal === '' || shapVal == null) continue;
        const contribution = parseFloat(shapVal);
        if (isNaN(contribution)) continue;
        const featureValue = featureRow[colName] ?? '';
        features.push({
          feature_name: zhFeatureName(colName),
          contribution,
          feature_value: featureValue,
        });
      }

      // Sort by absolute SHAP value descending, take top 10
      features.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
      const top10 = features.slice(0, 10);

      return {
        mode: shapMode,
        user_id: userId,
        base_value: -2.885,
        features: top10,
      };
    }
  }

  // Fall back: compute group average from real SHAP data
  if (shapMap !== null && shapMap.size > 0) {
    // Get relevant user IDs for this mode (FP or FN)
    const relevantIds = new Set<number>();
    if (shapMode === 'fp' && fpCache) {
      fpCache.forEach(n => relevantIds.add(n.user_id));
    } else if (shapMode === 'fn' && fnCache) {
      fnCache.forEach(n => relevantIds.add(n.user_id));
    }

    // Aggregate SHAP values across group
    const shapSums = new Map<string, { sum: number; count: number }>();
    for (const [uid, row] of shapMap.entries()) {
      if (relevantIds.size > 0 && !relevantIds.has(uid)) continue;
      for (const [col, val] of Object.entries(row)) {
        if (col === 'user_id' || val === '' || val == null) continue;
        const num = parseFloat(val);
        if (isNaN(num)) continue;
        const entry = shapSums.get(col) ?? { sum: 0, count: 0 };
        entry.sum += num;
        entry.count++;
        shapSums.set(col, entry);
      }
    }

    const avgFeatures: ShapWaterfallFeature[] = [];
    for (const [col, { sum, count }] of shapSums.entries()) {
      if (count === 0) continue;
      avgFeatures.push({
        feature_name: zhFeatureName(col),
        contribution: parseFloat((sum / count).toFixed(4)),
        feature_value: '群體平均',
      });
    }
    avgFeatures.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));

    return {
      mode: shapMode,
      user_id: null,
      base_value: -2.885,
      features: avgFeatures.slice(0, 10),
    };
  }

  // Last resort: mock
  const { mockFpShap, mockFnShap } = await import('../mock/shapData');
  return shapMode === 'fp' ? mockFpShap : mockFnShap;
}

// ── Public API ────────────────────────────────────────────────────────────────

export async function getComputedStats(): Promise<StatsResponse> {
  await loadGraphData();
  const nm = nodeMap!;
  const edges = allEdges!;

  const FRAUD_THRESHOLD = 0.8415;

  let fraudNodes = 0;
  let totalNodes = 0;
  const riskBuckets = [0, 0, 0, 0, 0];
  // [0, 0.2)  [0.2, 0.4)  [0.4, 0.6)  [0.6, THRESHOLD)  [THRESHOLD, 1.0]

  for (const rec of nm.values()) {
    if (rec.nodeType !== 'user') continue;
    totalNodes++;
    if (rec.label >= 1.0) fraudNodes++;
    const s = rec.riskScore;
    let idx: number;
    if      (s < 0.2)              idx = 0;
    else if (s < 0.4)              idx = 1;
    else if (s < 0.6)              idx = 2;
    else if (s < FRAUD_THRESHOLD)  idx = 3;
    else                           idx = 4;
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
      { range: '[0, 0.2)',        count: riskBuckets[0] },
      { range: '[0.2, 0.4)',      count: riskBuckets[1] },
      { range: '[0.4, 0.6)',      count: riskBuckets[2] },
      { range: '[0.6, 0.8415)',   count: riskBuckets[3] },
      { range: '[0.8415, 1.0]',   count: riskBuckets[4] },
    ],
    relation_counts,
  };
}

export async function getBlacklistNodes(): Promise<FraudNode[]> {
  if (blacklistCache) return blacklistCache;

  // 來源：all_user_risk_scores.csv，排除 true_label 為空白的預測目標用戶
  const text = await fetchCsv('/output/all_user_risk_scores.csv');
  const result: FraudNode[] = [];
  for (const r of parseCsvRecords(text)) {
    const label = r['true_label']?.trim();
    if (!label || label === '') continue; // 跳過 true_label 空白
    const uid = parseInt(r['user_id'] ?? '', 10);
    const score = parseFloat(r['risk_score'] ?? '0');
    if (isNaN(uid)) continue;
    result.push({ user_id: uid, risk_score: score });
  }

  blacklistCache = result.sort((a, b) => b.risk_score - a.risk_score);
  return blacklistCache;
}

export async function getFpFnData(): Promise<{ fp: FpFnNode[]; fn: FpFnNode[] }> {
  if (fpCache && fnCache) return { fp: fpCache, fn: fnCache };

  // 來源：all_user_risk_scores.csv
  // 欄位：user_id, true_label (0=白/1=黑/空=預測目標), risk_score, predicted_blacklist, risk_level, data_source
  const text = await fetchCsv('/output/all_user_risk_scores.csv');
  const records = parseCsvRecords(text);

  const fp: FpFnNode[] = [];
  const fn: FpFnNode[] = [];

  const THRESHOLD = 0.8415;
  for (const r of records) {
    const label = r['true_label']?.trim();
    if (label !== '0' && label !== '1') continue; // 跳過預測目標（空白）

    const user_id    = parseInt(r['user_id'] ?? '', 10);
    const risk_score = parseFloat(r['risk_score'] ?? '0');
    const actual     = parseInt(label, 10) as 0 | 1;
    const pred       = risk_score >= THRESHOLD ? 1 : 0;

    if (isNaN(user_id)) continue;

    if (actual === 0 && pred === 1) {
      // FP：實際正常，預測為詐騙
      fp.push({ user_id, risk_score, actual_status: 0, predicted_status: 1 });
    } else if (actual === 1 && pred === 0) {
      // FN：實際詐騙，預測為正常
      fn.push({ user_id, risk_score, actual_status: 1, predicted_status: 0 });
    }
  }

  fpCache = fp.sort((a, b) => b.risk_score - a.risk_score);
  fnCache = fn.sort((a, b) => a.risk_score - b.risk_score); // FN 按風險分數由低到高，方便找邊界案例

  return { fp: fpCache, fn: fnCache };
}

export async function getPredictData(): Promise<PredictNode[]> {
  if (predictCache) return predictCache;

  const text = await fetchCsv('/output/predict_detail.csv');
  const records = parseCsvRecords(text);

  // predict_detail.csv 欄位：user_id, risk_score, status（1=黑名單/0=正常）
  const metaCols = new Set(['user_id', 'risk_score', 'status']);
  const firstRow = records[0];
  const featureCols = firstRow ? Object.keys(firstRow).filter(k => !metaCols.has(k)) : [];

  predictCache = records.map(r => {
    const uid = parseInt(r['user_id'] ?? '0', 10);
    const riskScore = parseFloat(r['risk_score'] ?? '0');
    const isBlacklist = parseInt(r['status'] ?? '0', 10) as 0 | 1;

    // Extract non-null SHAP features (top-10 have values, rest are empty)
    const shapFeatures: ShapFeature[] = [];
    for (const col of featureCols) {
      const val = r[col];
      if (val != null && val !== '') {
        shapFeatures.push({
          feature_name: zhFeatureName(col),
          contribution: parseFloat(val),
        });
      }
    }
    // Sort by |contribution| descending
    shapFeatures.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));

    return { user_id: uid, risk_score: riskScore, is_blacklist: isBlacklist, shap_features: shapFeatures };
  })
    .filter(n => !isNaN(n.user_id))
    .sort((a, b) => b.risk_score - a.risk_score);

  return predictCache;
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

  // Build node set, filtering low-risk non-center users
  const keptNodeIds = new Set<number>();
  const resultNodes: SubgraphNode[] = [];
  for (const nodeId of visitedNodes) {
    const rec = nm.get(nodeId);
    if (!rec) continue;
    // Keep: center user, all wallets, users with risk >= 0.4
    if (rec.nodeType === 'user' && rec.numericId !== userId && rec.riskScore < 0.4) continue;
    keptNodeIds.add(rec.numericId);
    resultNodes.push({
      user_id: rec.numericId,
      risk_score: rec.riskScore,
      status: rec.label >= 1.0 ? 1 : 0,
      node_type: rec.nodeType,
      node_label: rec.nodeId,
    });
  }

  // Only keep edges where both endpoints are in the kept node set
  const filteredEdges = resultEdges.filter(
    e => keptNodeIds.has(e.source) && keptNodeIds.has(e.target),
  );

  // Remove orphan wallets (wallets with no remaining edges)
  const connectedIds = new Set<number>();
  for (const e of filteredEdges) {
    connectedIds.add(e.source);
    connectedIds.add(e.target);
  }
  // Center user is always kept even if isolated
  connectedIds.add(userId);
  const finalNodes = resultNodes.filter(n => connectedIds.has(n.user_id));

  return { nodes: finalNodes, edges: filteredEdges };
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
  await Promise.all([loadGraphData(), loadShapData()]);
  const nm = nodeMap!;
  const rec = userIdMap!.get(userId);
  if (!rec) return null;

  const adj = adjMap!;
  const edges = adj.get(rec.nodeId) ?? [];

  // Build neighbor_details: group by relation type and direction
  // Use Map<peerId, {rec, txCount}> per bucket
  const r1Map   = new Map<number, { rec: NodeRecord; count: number }>();
  const r2OutMap = new Map<number, { rec: NodeRecord; count: number }>();
  const r2InMap  = new Map<number, { rec: NodeRecord; count: number }>();
  const r3Map   = new Map<number, { rec: NodeRecord; count: number }>();

  for (const edge of edges) {
    const neighborKey = edge.source === rec.nodeId ? edge.target : edge.source;
    const neighborRec = nm.get(neighborKey);
    if (!neighborRec) continue;
    const peerId = neighborRec.numericId;

    if (edge.edgeType === 'wallet_funds_user') {
      // R1: wallet → this user (wallet is source)
      const e = r1Map.get(peerId) ?? { rec: neighborRec, count: 0 };
      r1Map.set(peerId, { rec: neighborRec, count: e.count + 1 });
    } else if (edge.edgeType === 'user_transfers_user') {
      // R2: check direction
      if (edge.source === rec.nodeId) {
        // outgoing: this user → peer
        const e = r2OutMap.get(peerId) ?? { rec: neighborRec, count: 0 };
        r2OutMap.set(peerId, { rec: neighborRec, count: e.count + 1 });
      } else {
        // incoming: peer → this user
        const e = r2InMap.get(peerId) ?? { rec: neighborRec, count: 0 };
        r2InMap.set(peerId, { rec: neighborRec, count: e.count + 1 });
      }
    } else if (edge.edgeType === 'user_sends_wallet') {
      // R3: this user → wallet (wallet is target)
      const e = r3Map.get(peerId) ?? { rec: neighborRec, count: 0 };
      r3Map.set(peerId, { rec: neighborRec, count: e.count + 1 });
    }
  }

  const toNeighborPeer = ({ rec: r, count }: { rec: NodeRecord; count: number }): NeighborPeer => ({
    peer_id: r.numericId,
    node_type: r.nodeType,
    node_label: r.nodeId,
    risk_score: r.riskScore,
    status: r.label >= 1.0 ? 1 : 0,
    tx_count: count,
  });

  const r1Peers    = Array.from(r1Map.values()).map(toNeighborPeer).sort((a, b) => b.risk_score - a.risk_score);
  const r2OutPeers = Array.from(r2OutMap.values()).map(toNeighborPeer).sort((a, b) => b.risk_score - a.risk_score);
  const r2InPeers  = Array.from(r2InMap.values()).map(toNeighborPeer).sort((a, b) => b.risk_score - a.risk_score);
  const r3Peers    = Array.from(r3Map.values()).map(toNeighborPeer).sort((a, b) => b.risk_score - a.risk_score);

  // Unique R2 peer count (union of in+out)
  const r2UniqueIds = new Set([...r2OutMap.keys(), ...r2InMap.keys()]);

  const neighbor_counts = {
    r1: r1Map.size,
    r2: r2UniqueIds.size,
    r3: r3Map.size,
  };

  // Load SHAP features for this user
  const shapFeatures: ShapFeature[] = [];
  if (shapMap) {
    const shapRow = shapMap.get(userId);
    if (shapRow) {
      for (const [colName, shapVal] of Object.entries(shapRow)) {
        if (colName === 'user_id' || shapVal === '' || shapVal == null) continue;
        const contribution = parseFloat(shapVal);
        if (isNaN(contribution)) continue;
        shapFeatures.push({ feature_name: zhFeatureName(colName), contribution });
      }
      shapFeatures.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
    }
  }

  return {
    user_id: userId,
    risk_score: rec.riskScore,
    status: rec.label >= 1.0 ? 1 : 0,
    account_age_days: 0,
    shap_features: shapFeatures.slice(0, 10),
    neighbor_counts,
    neighbor_details: { r1: r1Peers, r2_out: r2OutPeers, r2_in: r2InPeers, r3: r3Peers },
  };
}

export interface ConfusionMatrixData {
  tp: number; fp: number; tn: number; fn: number;
  total: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  specificity: number;
  threshold: number;
}

export async function getConfusionMatrix(): Promise<ConfusionMatrixData> {
  // 來源：all_user_risk_scores.csv（含完整真實標籤）
  // 只計算 true_label 有值的列（0=白、1=黑），空白為預測目標不納入
  const { fp: fpNodes, fn: fnNodes } = await getFpFnData();
  const fpCount = fpNodes.length;
  const fnCount = fnNodes.length;

  // 從 all_user_risk_scores.csv 直接算全部 TP / TN
  const text = await fetchCsv('/output/all_user_risk_scores.csv');
  const THRESHOLD = 0.8415;
  let tp = 0, tn = 0;
  for (const r of parseCsvRecords(text)) {
    const label = r['true_label']?.trim();
    if (label !== '0' && label !== '1') continue;
    const actual    = parseInt(label, 10);
    const predicted = parseFloat(r['risk_score'] ?? '0') >= THRESHOLD ? 1 : 0;
    if (actual === 1 && predicted === 1) tp++;
    else if (actual === 0 && predicted === 0) tn++;
  }

  const fp    = fpCount;
  const fn    = fnCount;
  const total = tp + fp + tn + fn;
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall    = tp + fn > 0 ? tp / (tp + fn) : 0;
  return {
    tp, fp, tn, fn, total,
    accuracy:    total > 0 ? (tp + tn) / total : 0,
    precision,
    recall,
    f1:          precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0,
    specificity: tn + fp > 0 ? tn / (tn + fp) : 0,
    threshold:   0.8415,
  };
}

export interface ShapTop20Entry {
  rank: number;
  feature: string;   // 英文欄位名
  label: string;     // 中文名稱
  shap: number;      // 平均 |SHAP| 值
  pct: string;       // 佔比 e.g. "8.45%"
  freq: number;      // 頻率次數
  cumPct: string;    // 累積%
}

let shapTop20AllCache: ShapTop20Entry[] | null = null;
let shapTop20BlacklistCache: ShapTop20Entry[] | null = null;

function parseShapTop20(text: string): ShapTop20Entry[] {
  const records = parseCsvRecords(text);
  return records.map(r => ({
    rank:    parseInt(r['排名'] ?? '0', 10),
    feature: r['feature'] ?? '',
    label:   r['中文名稱'] ?? r['feature'] ?? '',
    shap:    parseFloat(r['SHAP值'] ?? '0'),
    pct:     r['佔比'] ?? '',
    freq:    parseInt(r['頻率次數'] ?? '0', 10),
    cumPct:  r['累積%'] ?? '',
  })).filter(e => e.rank > 0);
}

export async function getShapTop20AllUsers(): Promise<ShapTop20Entry[]> {
  if (shapTop20AllCache) return shapTop20AllCache;
  const text = await fetchCsv('/output/shap_top20_all_users.csv');
  shapTop20AllCache = parseShapTop20(text);
  return shapTop20AllCache;
}

export async function getShapTop20Blacklist(): Promise<ShapTop20Entry[]> {
  if (shapTop20BlacklistCache) return shapTop20BlacklistCache;

  const text = await fetchCsv('/output/shap_values_blacklist.csv');
  const records = parseCsvRecords(text);

  // 計算每個特徵的 mean|SHAP| 與正值頻率次數
  const stats = new Map<string, { sumAbs: number; posCount: number; count: number }>();
  for (const row of records) {
    for (const [col, val] of Object.entries(row)) {
      if (col === 'user_id' || val === '' || val == null) continue;
      const num = parseFloat(val as string);
      if (isNaN(num)) continue;
      const s = stats.get(col) ?? { sumAbs: 0, posCount: 0, count: 0 };
      s.sumAbs += Math.abs(num);
      if (num > 0) s.posCount += 1;
      s.count += 1;
      stats.set(col, s);
    }
  }

  // 計算佔比分母（全特徵 mean|SHAP| 總和）
  let totalMeanAbs = 0;
  const featureStats = Array.from(stats.entries()).map(([col, { sumAbs, posCount, count }]) => {
    const meanAbs = sumAbs / count;
    totalMeanAbs += meanAbs;
    return { col, meanAbs, posCount };
  });

  // 排序取前 20
  featureStats.sort((a, b) => b.meanAbs - a.meanAbs);
  const top20 = featureStats.slice(0, 20);

  // 組裝 ShapTop20Entry
  let cumSum = 0;
  shapTop20BlacklistCache = top20.map(({ col, meanAbs, posCount }, i) => {
    const pctVal = totalMeanAbs > 0 ? (meanAbs / totalMeanAbs) * 100 : 0;
    cumSum += pctVal;
    return {
      rank:    i + 1,
      feature: col,
      label:   zhFeatureName(col),
      shap:    parseFloat(meanAbs.toFixed(6)),
      pct:     pctVal.toFixed(2) + '%',
      freq:    posCount,
      cumPct:  cumSum.toFixed(2) + '%',
    };
  });

  return shapTop20BlacklistCache;
}

export interface FeatureImportanceEntry {
  feature_name: string;       // 英文欄位名
  label: string;              // 中文名稱
  frequency: number;          // 出現在 top-10 的用戶人數
  avg_abs_shap: number;       // 平均 |SHAP| (只計非空值)
  avg_shap: number;           // 平均 SHAP (正=拉高風險, 負=拉低)
}

/** 從 shap_values.csv 彙總每個特徵的重要性統計 */
export async function getFeatureImportanceSummary(): Promise<FeatureImportanceEntry[]> {
  await loadShapData();
  if (!shapMap || shapMap.size === 0) return [];

  const stats = new Map<string, { sumAbs: number; sum: number; count: number }>();

  for (const row of shapMap.values()) {
    for (const [col, val] of Object.entries(row)) {
      if (col === 'user_id' || val === '' || val == null) continue;
      const num = parseFloat(val);
      if (isNaN(num)) continue;
      const entry = stats.get(col) ?? { sumAbs: 0, sum: 0, count: 0 };
      entry.sumAbs += Math.abs(num);
      entry.sum += num;
      entry.count += 1;
      stats.set(col, entry);
    }
  }

  return Array.from(stats.entries())
    .map(([col, { sumAbs, sum, count }]) => ({
      feature_name: col,
      label: zhFeatureName(col),
      frequency: count,
      avg_abs_shap: parseFloat((sumAbs / count).toFixed(5)),
      avg_shap: parseFloat((sum / count).toFixed(5)),
    }))
    .sort((a, b) => b.avg_abs_shap - a.avg_abs_shap);
}
