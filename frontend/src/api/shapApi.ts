import type { ShapWaterfallResponse, FpFnMode } from '../types/index';
import { mockFpShap, mockFnShap } from '../mock/shapData';

/**
 * GET /api/shap/waterfall?mode=fp|fn[&user_id=xxx]
 *
 * Backend implementation guide:
 *   1. Read shap_all_features.csv  (columns: user_id, feat1, feat2, …)
 *   2. Filter rows for the FP/FN user list (from classification report)
 *   3. If user_id provided: return single-row SHAP; else return column-mean
 *   4. Derive base_value from shap explainer's expected_value field
 *   5. Sort features by |shap_value| desc, return top N
 *
 * Replace the mock fallback below with a real apiFetch call once the
 * backend endpoint is live.
 */
export async function getShapWaterfall(
  mode: FpFnMode,
  userId?: number,
): Promise<ShapWaterfallResponse> {
  // TODO: swap for real API call
  // const qs = userId != null ? `&user_id=${userId}` : '';
  // return apiFetch<ShapWaterfallResponse>(`/api/shap/waterfall?mode=${mode}${qs}`);

  await new Promise(r => setTimeout(r, 200)); // simulate latency
  const base = mode === 'fp' ? mockFpShap : mockFnShap;

  if (userId != null) {
    // Simulate per-user variation by slightly perturbing contributions
    const seed = userId % 7;
    return {
      ...base,
      user_id: userId,
      features: base.features.map((f, i) => ({
        ...f,
        contribution: parseFloat((f.contribution + (i === seed ? 0.08 : -0.03)).toFixed(3)),
      })),
    };
  }

  return base;
}
