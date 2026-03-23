import type { ShapWaterfallResponse } from '../types/index';

// Mock data mirrors what the backend would parse from shap_values CSV.
// base_value = E[f(x)] derived from shap_base_value or computed as mean(shap_values.sum(axis=1))
const BASE_VALUE = -2.885;

// FP group average: normal users pushed above fraud threshold
// feature_value = representative value for this cohort
export const mockFpShap: ShapWaterfallResponse = {
  mode: 'fp',
  user_id: null,
  base_value: BASE_VALUE,
  features: [
    { feature_name: '共用 IP 節點數',  feature_value: '2174', contribution:  0.62 },
    { feature_name: '高風險鄰居比例',  feature_value: '39%',  contribution:  0.43 },
    { feature_name: '交易頻率（7天）', feature_value: '7',    contribution:  0.38 },
    { feature_name: '資金損失紀錄',    feature_value: '0',    contribution:  0.18 },
    { feature_name: '每週交易次數',    feature_value: '40',   contribution:  0.05 },
    { feature_name: '國家代碼',        feature_value: '39',   contribution:  0.02 },
    { feature_name: '帳戶年齡（天）',  feature_value: '821',  contribution: -0.33 },
    { feature_name: '共用錢包數',      feature_value: '0',    contribution: -0.28 },
    { feature_name: '詐騙鄰居數',      feature_value: '13',   contribution: -0.22 },
    { feature_name: '職業類型',        feature_value: '1',    contribution: -0.08 },
  ],
};

// FN group average: fraud users that stayed below fraud threshold
export const mockFnShap: ShapWaterfallResponse = {
  mode: 'fn',
  user_id: null,
  base_value: BASE_VALUE,
  features: [
    { feature_name: '帳戶年齡（天）',  feature_value: '1205', contribution: -0.62 },
    { feature_name: '平均交易金額',    feature_value: '低',   contribution: -0.48 },
    { feature_name: '交易頻率（7天）', feature_value: '3',    contribution: -0.33 },
    { feature_name: '共用錢包數',      feature_value: '0',    contribution: -0.21 },
    { feature_name: '職業類型',        feature_value: '2',    contribution: -0.09 },
    { feature_name: '資金增益紀錄',    feature_value: '0',    contribution: -0.07 },
    { feature_name: '國家代碼',        feature_value: '5',    contribution: -0.02 },
    { feature_name: '詐騙鄰居比例',    feature_value: '12%',  contribution:  0.28 },
    { feature_name: '共用 IP 節點數',  feature_value: '1',    contribution:  0.14 },
    { feature_name: '教育程度',        feature_value: '12',   contribution:  0.03 },
  ],
};
