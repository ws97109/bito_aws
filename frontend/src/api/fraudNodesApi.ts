import { mockFraudNodes } from '../mock/fraudNodesData';
import type { FraudNodesResponse } from '../types/index';

// TODO: Replace with: return apiFetch<FraudNodesResponse>('/api/fraud-nodes');
export async function getFraudNodes(): Promise<FraudNodesResponse> {
  return Promise.resolve([...mockFraudNodes].sort((a, b) => b.risk_score - a.risk_score));
}
