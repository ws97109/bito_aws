/**
 * Fairness audit results.
 *
 * Numbers are derived from the model's holdout-set predictions
 * grouped by protected attributes; the raw computation lives in
 * `notebooks/fairness_audit.ipynb` and is summarised here for the
 * dashboard. Samples count reflects the holdout split, not full
 * training data. Source: model run 2026-04-18 (commit hash in
 * project_fairness_audit_results.md).
 */
export const FAIRNESS_SOURCE = {
  notebook: 'final_model/output/baseline_loo/fairness_summary.json',
  runDate:  '2026-04-21',
  split:    'test',
};


export type FairnessStatus = 'PASS' | 'WARNING' | 'FAIL';

export interface FairnessCheck {
  id: string;
  /** i18n key for the attribute name (e.g. fairness.attr.gender) */
  attributeKey: string;
  status: FairnessStatus;
  /** Disparate Impact ratio (min/max group FPR). 1.0 = perfect parity. */
  disparateImpact: number;
  /** Groups with their FPR. */
  groups: { key: string; labelKey: string; fpr: number; fnr: number; count: number }[];
  /** Short summary i18n key. */
  summaryKey: string;
}

// Numbers below are from the LOO-enabled model's test-set audit
// (2026-04-21). With the new model's Precision=0.928 the FPR scale is
// an order of magnitude lower than the baseline model; gender/age
// DIR now fail the 0.8 threshold because the model is so selective
// that rare-class exposure becomes very uneven by demographic group.
export const FAIRNESS_CHECKS: FairnessCheck[] = [
  {
    id: 'gender',
    attributeKey: 'fairness.attr.gender',
    status: 'FAIL',
    disparateImpact: 0.285,
    groups: [
      { key: 'female', labelKey: 'fairness.group.female', fpr: 0.0043, fnr: 0.235, count: 2721 },
      { key: 'male',   labelKey: 'fairness.group.male',   fpr: 0.0011, fnr: 0.275, count: 7483 },
    ],
    summaryKey: 'fairness.summary.gender',
  },
  {
    id: 'age',
    attributeKey: 'fairness.attr.age',
    status: 'FAIL',
    disparateImpact: 0.343,
    groups: [
      { key: 'under30',  labelKey: 'fairness.group.under30',  fpr: 0.0016, fnr: 0.209, count: 3761 },
      { key: 'age30_50', labelKey: 'fairness.group.age30_50', fpr: 0.0016, fnr: 0.262, count: 5217 },
      { key: 'over50',   labelKey: 'fairness.group.over50',   fpr: 0.0043, fnr: 0.271, count: 1226 },
    ],
    summaryKey: 'fairness.summary.age',
  },
  {
    id: 'career',
    attributeKey: 'fairness.attr.career',
    status: 'FAIL',
    disparateImpact: 0.729,
    groups: [
      { key: 'low_risk',  labelKey: 'fairness.group.career_low',  fpr: 0.0018, fnr: 0.268, count: 9090 },
      { key: 'high_risk', labelKey: 'fairness.group.career_high', fpr: 0.0028, fnr: 0.146, count: 1114 },
    ],
    summaryKey: 'fairness.summary.career',
  },
  {
    id: 'income',
    attributeKey: 'fairness.attr.income',
    status: 'FAIL',
    disparateImpact: 0.557,
    groups: [
      { key: 'low',    labelKey: 'fairness.group.income_low',    fpr: 0.0020, fnr: 0.238, count: 9400 },
      { key: 'high',   labelKey: 'fairness.group.income_high',   fpr: 0.0013, fnr: 0.476, count: 804 },
    ],
    summaryKey: 'fairness.summary.income',
  },
];

export const FAIRNESS_STATUS_STYLE: Record<FairnessStatus, { chip: string; dot: string; text: string }> = {
  PASS:    { chip: 'bg-emerald-900/60 text-emerald-300 ring-1 ring-emerald-500/50', dot: 'bg-emerald-400', text: 'text-emerald-400' },
  WARNING: { chip: 'bg-amber-900/60 text-amber-300 ring-1 ring-amber-500/50',       dot: 'bg-amber-400',   text: 'text-amber-400'   },
  FAIL:    { chip: 'bg-red-900/60 text-red-300 ring-1 ring-red-500/50',             dot: 'bg-red-400',     text: 'text-red-400'     },
};
