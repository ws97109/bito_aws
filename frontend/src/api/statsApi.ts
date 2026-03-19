import { mockStats } from '../mock/statsData';
import type { StatsResponse } from '../types/index';

// TODO: Replace with: return apiFetch<StatsResponse>('/api/stats');
export async function getStats(): Promise<StatsResponse> {
  return Promise.resolve(mockStats);
}
