import type { FpFnNode } from '../types/index';

// FP: normal users (actual_status=0) predicted as fraud (predicted_status=1)
// Typically have risk_score slightly above threshold (0.51–0.79)
export const mockFpNodes: FpFnNode[] = Array.from({ length: 237 }, (_, i) => ({
  user_id: 1500 + i,
  risk_score: parseFloat((0.51 + (i % 29) * 0.01).toFixed(3)),
  actual_status: 0 as const,
  predicted_status: 1 as const,
}));

// FN: fraud users (actual_status=1) predicted as normal (predicted_status=0)
// Typically have risk_score just below threshold (0.10–0.49)
export const mockFnNodes: FpFnNode[] = Array.from({ length: 203 }, (_, i) => ({
  user_id: 2000 + i,
  risk_score: parseFloat((0.10 + (i % 40) * 0.01).toFixed(3)),
  actual_status: 1 as const,
  predicted_status: 0 as const,
}));
