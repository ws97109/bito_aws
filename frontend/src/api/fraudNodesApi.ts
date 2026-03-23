import { getBlacklistNodes } from '../utils/graphDataStore';
import type { FraudNodesResponse } from '../types/index';

export async function getFraudNodes(): Promise<FraudNodesResponse> {
  return getBlacklistNodes();
}
