import type { NodeDetailResponse } from '../types/index';

function makeDetail(
  userId: number,
  riskScore: number,
  status: 0 | 1,
  ageDays: number,
  shapTop3: [string, number][],
  r1: number,
  r2: number,
  r3: number,
): NodeDetailResponse {
  return {
    user_id: userId,
    risk_score: riskScore,
    status,
    account_age_days: ageDays,
    shap_features: shapTop3.map(([feature_name, contribution]) => ({ feature_name, contribution })),
    neighbor_counts: { r1, r2, r3 },
    neighbor_details: { r1: [], r2_out: [], r2_in: [], r3: [] },
  };
}

// Fraud nodes (status=1)
const fraudEntries: NodeDetailResponse[] = [
  makeDetail(1042, 0.987, 1, 120, [['transfer_frequency', 0.42], ['crypto_transfer_count', 0.31], ['unique_counterparties', 0.18]], 8, 5, 3),
  makeDetail(1187, 0.981, 1, 85,  [['avg_transfer_amount', 0.38], ['usdt_swap_count', 0.29], ['transfer_frequency', 0.21]], 6, 4, 2),
  makeDetail(1356, 0.975, 1, 210, [['crypto_transfer_count', 0.45], ['transfer_frequency', 0.27], ['twd_transfer_count', 0.15]], 9, 3, 4),
  makeDetail(1093, 0.968, 1, 55,  [['unique_counterparties', 0.41], ['avg_transfer_amount', 0.33], ['usdt_swap_count', 0.19]], 7, 6, 2),
  makeDetail(1274, 0.962, 1, 340, [['transfer_frequency', 0.39], ['crypto_transfer_count', 0.28], ['account_age_days', -0.12]], 5, 4, 3),
  makeDetail(1511, 0.955, 1, 72,  [['usdt_swap_count', 0.44], ['avg_transfer_amount', 0.30], ['unique_counterparties', 0.17]], 8, 2, 5),
  makeDetail(1128, 0.948, 1, 180, [['crypto_transfer_count', 0.37], ['transfer_frequency', 0.26], ['twd_transfer_count', 0.20]], 6, 5, 3),
  makeDetail(1463, 0.941, 1, 95,  [['avg_transfer_amount', 0.43], ['usdt_swap_count', 0.25], ['transfer_frequency', 0.18]], 7, 3, 4),
  makeDetail(1319, 0.934, 1, 430, [['transfer_frequency', 0.36], ['unique_counterparties', 0.29], ['crypto_transfer_count', 0.16]], 5, 6, 2),
  makeDetail(1076, 0.927, 1, 63,  [['usdt_swap_count', 0.40], ['avg_transfer_amount', 0.32], ['twd_transfer_count', 0.14]], 9, 4, 3),
  makeDetail(1582, 0.919, 1, 155, [['crypto_transfer_count', 0.38], ['transfer_frequency', 0.27], ['unique_counterparties', 0.19]], 6, 3, 5),
  makeDetail(1234, 0.912, 1, 280, [['transfer_frequency', 0.41], ['usdt_swap_count', 0.24], ['avg_transfer_amount', 0.17]], 8, 5, 2),
  makeDetail(1407, 0.905, 1, 110, [['unique_counterparties', 0.39], ['crypto_transfer_count', 0.30], ['twd_transfer_count', 0.16]], 5, 4, 4),
  makeDetail(1155, 0.897, 1, 520, [['avg_transfer_amount', 0.35], ['transfer_frequency', 0.28], ['usdt_swap_count', 0.20]], 7, 6, 3),
  makeDetail(1638, 0.889, 1, 88,  [['crypto_transfer_count', 0.42], ['unique_counterparties', 0.26], ['account_age_days', -0.15]], 6, 3, 2),
  makeDetail(1291, 0.882, 1, 195, [['transfer_frequency', 0.37], ['avg_transfer_amount', 0.31], ['twd_transfer_count', 0.18]], 8, 4, 5),
  makeDetail(1174, 0.874, 1, 145, [['usdt_swap_count', 0.43], ['crypto_transfer_count', 0.25], ['unique_counterparties', 0.16]], 5, 5, 3),
  makeDetail(1523, 0.867, 1, 370, [['avg_transfer_amount', 0.36], ['transfer_frequency', 0.29], ['usdt_swap_count', 0.17]], 7, 3, 4),
  makeDetail(1389, 0.859, 1, 78,  [['transfer_frequency', 0.40], ['unique_counterparties', 0.27], ['crypto_transfer_count', 0.15]], 6, 6, 2),
  makeDetail(1061, 0.851, 1, 240, [['crypto_transfer_count', 0.38], ['avg_transfer_amount', 0.30], ['twd_transfer_count', 0.19]], 9, 4, 3),
  makeDetail(1445, 0.843, 1, 165, [['unique_counterparties', 0.41], ['usdt_swap_count', 0.24], ['transfer_frequency', 0.18]], 5, 3, 5),
  makeDetail(1217, 0.836, 1, 310, [['transfer_frequency', 0.35], ['crypto_transfer_count', 0.28], ['avg_transfer_amount', 0.20]], 8, 5, 2),
  makeDetail(1702, 0.828, 1, 92,  [['avg_transfer_amount', 0.39], ['twd_transfer_count', 0.26], ['unique_counterparties', 0.16]], 6, 4, 4),
  makeDetail(1336, 0.820, 1, 460, [['usdt_swap_count', 0.37], ['transfer_frequency', 0.29], ['crypto_transfer_count', 0.17]], 7, 3, 3),
  makeDetail(1108, 0.812, 1, 130, [['crypto_transfer_count', 0.40], ['unique_counterparties', 0.25], ['avg_transfer_amount', 0.18]], 5, 6, 2),
  makeDetail(1567, 0.804, 1, 200, [['transfer_frequency', 0.36], ['usdt_swap_count', 0.28], ['twd_transfer_count', 0.19]], 8, 4, 5),
  makeDetail(1253, 0.796, 1, 115, [['avg_transfer_amount', 0.38], ['crypto_transfer_count', 0.27], ['unique_counterparties', 0.16]], 6, 3, 3),
  makeDetail(1481, 0.788, 1, 285, [['unique_counterparties', 0.35], ['transfer_frequency', 0.30], ['usdt_swap_count', 0.18]], 7, 5, 4),
  makeDetail(1144, 0.780, 1, 175, [['crypto_transfer_count', 0.39], ['avg_transfer_amount', 0.24], ['twd_transfer_count', 0.17]], 5, 4, 2),
  makeDetail(1625, 0.772, 1, 395, [['transfer_frequency', 0.37], ['unique_counterparties', 0.27], ['usdt_swap_count', 0.16]], 8, 3, 5),
  makeDetail(1372, 0.764, 1, 105, [['avg_transfer_amount', 0.40], ['crypto_transfer_count', 0.26], ['transfer_frequency', 0.18]], 6, 6, 3),
  makeDetail(1199, 0.756, 1, 255, [['usdt_swap_count', 0.36], ['twd_transfer_count', 0.29], ['unique_counterparties', 0.17]], 7, 4, 4),
  makeDetail(1543, 0.748, 1, 140, [['transfer_frequency', 0.38], ['avg_transfer_amount', 0.25], ['crypto_transfer_count', 0.19]], 5, 3, 2),
  makeDetail(1087, 0.740, 1, 320, [['crypto_transfer_count', 0.35], ['unique_counterparties', 0.28], ['usdt_swap_count', 0.18]], 8, 5, 3),
  makeDetail(1416, 0.732, 1, 185, [['avg_transfer_amount', 0.37], ['transfer_frequency', 0.26], ['twd_transfer_count', 0.16]], 6, 4, 5),
  makeDetail(1268, 0.724, 1, 445, [['unique_counterparties', 0.39], ['crypto_transfer_count', 0.24], ['avg_transfer_amount', 0.17]], 7, 3, 3),
  makeDetail(1731, 0.716, 1, 98,  [['transfer_frequency', 0.36], ['usdt_swap_count', 0.27], ['unique_counterparties', 0.19]], 5, 6, 2),
  makeDetail(1163, 0.708, 1, 230, [['crypto_transfer_count', 0.38], ['avg_transfer_amount', 0.25], ['twd_transfer_count', 0.18]], 8, 4, 4),
  makeDetail(1494, 0.700, 1, 160, [['avg_transfer_amount', 0.35], ['transfer_frequency', 0.28], ['usdt_swap_count', 0.17]], 6, 3, 3),
  makeDetail(1327, 0.692, 1, 380, [['usdt_swap_count', 0.37], ['unique_counterparties', 0.26], ['crypto_transfer_count', 0.16]], 7, 5, 5),
  makeDetail(1112, 0.684, 1, 125, [['transfer_frequency', 0.39], ['avg_transfer_amount', 0.24], ['twd_transfer_count', 0.18]], 5, 4, 2),
  makeDetail(1658, 0.676, 1, 295, [['crypto_transfer_count', 0.36], ['unique_counterparties', 0.27], ['usdt_swap_count', 0.17]], 8, 3, 4),
  makeDetail(1245, 0.668, 1, 170, [['avg_transfer_amount', 0.38], ['transfer_frequency', 0.25], ['crypto_transfer_count', 0.19]], 6, 6, 3),
  makeDetail(1503, 0.660, 1, 415, [['unique_counterparties', 0.35], ['usdt_swap_count', 0.28], ['twd_transfer_count', 0.16]], 7, 4, 5),
  makeDetail(1381, 0.652, 1, 135, [['transfer_frequency', 0.37], ['crypto_transfer_count', 0.26], ['avg_transfer_amount', 0.18]], 5, 3, 2),
  makeDetail(1136, 0.644, 1, 265, [['avg_transfer_amount', 0.39], ['unique_counterparties', 0.24], ['usdt_swap_count', 0.17]], 8, 5, 3),
  makeDetail(1592, 0.636, 1, 190, [['crypto_transfer_count', 0.36], ['transfer_frequency', 0.27], ['twd_transfer_count', 0.19]], 6, 4, 4),
  makeDetail(1308, 0.628, 1, 345, [['usdt_swap_count', 0.38], ['avg_transfer_amount', 0.25], ['unique_counterparties', 0.16]], 7, 3, 3),
  makeDetail(1457, 0.620, 1, 115, [['transfer_frequency', 0.35], ['crypto_transfer_count', 0.28], ['avg_transfer_amount', 0.18]], 5, 6, 2),
  makeDetail(1223, 0.612, 1, 275, [['unique_counterparties', 0.37], ['usdt_swap_count', 0.26], ['twd_transfer_count', 0.17]], 8, 4, 5),
  makeDetail(1674, 0.604, 1, 155, [['avg_transfer_amount', 0.39], ['transfer_frequency', 0.24], ['crypto_transfer_count', 0.19]], 6, 3, 3),
  makeDetail(1349, 0.596, 1, 430, [['crypto_transfer_count', 0.36], ['unique_counterparties', 0.27], ['usdt_swap_count', 0.16]], 7, 5, 4),
];

// Some normal nodes (status=0)
const normalEntries: NodeDetailResponse[] = [
  makeDetail(2001, 0.12, 0, 1820, [['account_age_days', -0.25], ['transfer_frequency', -0.18], ['avg_transfer_amount', 0.08]], 3, 2, 1),
  makeDetail(2002, 0.08, 0, 2540, [['account_age_days', -0.30], ['unique_counterparties', -0.12], ['twd_transfer_count', 0.05]], 2, 1, 2),
  makeDetail(2003, 0.15, 0, 980,  [['transfer_frequency', 0.10], ['account_age_days', -0.22], ['avg_transfer_amount', -0.09]], 4, 3, 1),
  makeDetail(2004, 0.19, 0, 1450, [['avg_transfer_amount', 0.12], ['account_age_days', -0.20], ['unique_counterparties', -0.07]], 3, 2, 2),
  makeDetail(2005, 0.07, 0, 3200, [['account_age_days', -0.35], ['twd_transfer_count', -0.14], ['transfer_frequency', 0.06]], 2, 1, 1),
  makeDetail(2006, 0.11, 0, 2100, [['account_age_days', -0.28], ['avg_transfer_amount', -0.11], ['unique_counterparties', 0.07]], 3, 3, 2),
  makeDetail(2007, 0.16, 0, 760,  [['transfer_frequency', 0.13], ['account_age_days', -0.19], ['usdt_swap_count', -0.08]], 4, 2, 1),
  makeDetail(2008, 0.09, 0, 1680, [['account_age_days', -0.27], ['unique_counterparties', -0.13], ['avg_transfer_amount', 0.06]], 2, 2, 2),
  makeDetail(2009, 0.14, 0, 890,  [['avg_transfer_amount', 0.11], ['account_age_days', -0.21], ['twd_transfer_count', -0.07]], 3, 1, 1),
  makeDetail(2010, 0.18, 0, 1320, [['transfer_frequency', 0.14], ['account_age_days', -0.18], ['unique_counterparties', -0.09]], 4, 3, 2),
];

const allEntries = [...fraudEntries, ...normalEntries];

export const mockNodeDetails: Map<number, NodeDetailResponse> = new Map(
  allEntries.map(entry => [entry.user_id, entry])
);

export function getNodeDetail(userId: number): NodeDetailResponse | null {
  return mockNodeDetails.get(userId) ?? null;
}


