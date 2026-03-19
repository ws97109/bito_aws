import { getNodeDetail as getMockNodeDetail } from '../mock/nodeData';
import { ApiError } from './client';
import type { NodeDetailResponse } from '../types/index';

// TODO: Replace with: return apiFetch<NodeDetailResponse>(`/api/node/${userId}`);
export async function getNodeDetail(userId: number): Promise<NodeDetailResponse> {
  const detail = getMockNodeDetail(userId);
  if (!detail) throw new ApiError(404, `Node ${userId} not found`);
  return Promise.resolve(detail);
}
