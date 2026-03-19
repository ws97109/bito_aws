import { getSubgraphData } from '../mock/subgraphData';
import type { SubgraphResponse } from '../types/index';

// TODO: Replace with: return apiFetch<SubgraphResponse>(`/api/subgraph/${userId}?hops=${hops}`);
export async function getSubgraph(userId: number, hops: number = 2): Promise<SubgraphResponse> {
  return Promise.resolve(getSubgraphData(userId, hops));
}
