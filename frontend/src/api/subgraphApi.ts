import { getComputedSubgraph } from '../utils/graphDataStore';
import type { SubgraphResponse } from '../types/index';

export async function getSubgraph(userId: number, hops: number = 2): Promise<SubgraphResponse> {
  return getComputedSubgraph(userId, hops);
}
