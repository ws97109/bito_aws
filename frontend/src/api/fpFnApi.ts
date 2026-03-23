import { getFpFnData } from '../utils/graphDataStore';
import type { FpFnNode } from '../types/index';

export async function getFpFnNodes(): Promise<{ fp: FpFnNode[]; fn: FpFnNode[] }> {
  return getFpFnData();
}
