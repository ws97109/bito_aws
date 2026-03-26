import type { ShapWaterfallResponse, FpFnMode } from '../types/index';
import { getShapForUser } from '../utils/graphDataStore';

/**
 * GET /api/shap/waterfall?mode=fp|fn[&user_id=xxx]
 *
 * Uses real per-user feature values from features.csv via getShapForUser.
 * Falls back to mock data if the user is not found in features.csv.
 */
export async function getShapWaterfall(
  mode: FpFnMode,
  userId?: number,
): Promise<ShapWaterfallResponse> {
  return getShapForUser(mode, userId);
}
