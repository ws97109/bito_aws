import type { StatsResponse } from '../types/index';

export const mockStats: StatsResponse = {
  total_nodes: 500,
  fraud_nodes: 52,
  normal_nodes: 448,
  fraud_ratio: 0.104,
  risk_distribution: [
    { range: '[0, 0.2)', count: 210 },
    { range: '[0.2, 0.4)', count: 138 },
    { range: '[0.4, 0.6)', count: 82 },
    { range: '[0.6, 0.8)', count: 45 },
    { range: '[0.8, 1.0]', count: 25 },
  ],
  relation_counts: {
    r1: 1243,
    r2: 876,
    r3: 534,
  },
};
