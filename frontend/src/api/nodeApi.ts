import { getComputedNodeDetail } from '../utils/graphDataStore';
import { ApiError } from './client';
import type { NodeDetailResponse } from '../types/index';

export async function getNodeDetail(userId: number): Promise<NodeDetailResponse> {
  const detail = await getComputedNodeDetail(userId);
  if (!detail) throw new ApiError(404, `Node ${userId} not found`);
  return detail;
}
