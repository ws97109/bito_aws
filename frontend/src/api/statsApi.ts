import { getComputedStats } from '../utils/graphDataStore';
import type { StatsResponse } from '../types/index';

export async function getStats(): Promise<StatsResponse> {
  return getComputedStats();
}
