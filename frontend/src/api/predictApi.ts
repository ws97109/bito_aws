import { getPredictData } from '../utils/graphDataStore';
import type { PredictNode } from '../types/index';

export async function getPredictNodes(): Promise<PredictNode[]> {
  return getPredictData();
}
