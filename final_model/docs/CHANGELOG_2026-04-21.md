# Changelog — 2026-04-21 模型改進實驗

## Added

- `model/improved_evaluation.py`
  - `IsotonicCalibrator` / `calibrate_on_holdout` / `calibrate_on_test`
  - `threshold_max_f1 / threshold_max_fbeta / threshold_min_cost`
  - `evaluate_proba / evaluate_at_threshold` helpers
  - `run_post_hoc_improvements` — 一鍵比較 baseline / F1 / F2 / cost / calibrated
- `model/pu_learning.py`
  - `pu_oof_probabilities` — 5-fold OOF via `pulearn`
  - `pu_full_fit_predict` — 全訓練集 fit + test predict
  - 支援 `BaggingPuClassifier` 與 `ElkanotoPuClassifier`
- `model/tabpfn_base.py`
  - `tabpfn_oof_probabilities` — per-fold 子抽樣至 8k rows
  - `tabpfn_fit_predict_test`
  - 需 `TABPFN_TOKEN` 環境變數
- `model/run_improved.py` — P0-1/P0-2 實驗 driver
- `model/run_improved_v2.py` — 加上 PU learning 的整合實驗
- `model/run_final.py` — 多閾值提交檔案產生器
- `model/test_tabpfn_quick.py` — TabPFN 獨立 benchmark
- `model/IMPROVEMENTS_README.md` — 操作指南
- `docs/model_improvements_2026-04.md` — 完整實驗報告

## Changed

- 無（完全不動 `main.py` / `ensemble.py` / 特徵工程 / GNN）

## Test runs

```
output/baseline/          # main.py --skip_gnn  (快速版 baseline)
output/baseline_gnn/      # main.py             (完整版 baseline，含 GNN)
output/improved/          # P0-1/P0-2 輸出
output/improved_v2/       # PU bagging 整合
output/improved_v2_elk/   # PU elkanoto 整合
output/final/             # 最終多閾值提交
```

## Metrics snapshot

`output/baseline/metrics.json`（skip_gnn）：

```
AUC-ROC=0.8641, AUC-PR=0.3262, F1=0.3746, P=0.3552, R=0.3963
oof_threshold=0.8804, sweep_threshold=0.8780
```

`output/improved/metrics_improved.json` 摘要：

```
raw max_f1   : thr=0.8780  F1=0.3746  (與 baseline 相同)
raw max_f2   : thr=0.7861  F1=0.3154  F2=0.4224  R=0.5457  ← 最佳 F2
raw min_cost : thr=0.7861  F1=0.3154  cost minimised
cal_sigmoid  : rank-preserving (AUC-PR 不變)
cal_isotonic : AUC-PR 下降 0.04（overfit holdout）
```

## Rollback plan

所有改動可安全丟掉：

```bash
# 丟掉本分支，回到 main
git checkout main
git branch -D feat/model-improvements

# 或只丟新檔案，保留其他
git checkout feat/model-improvements
rm model/improved_evaluation.py model/pu_learning.py model/tabpfn_base.py \
   model/run_improved.py model/run_improved_v2.py model/run_final.py \
   model/test_tabpfn_quick.py model/IMPROVEMENTS_README.md \
   docs/model_improvements_2026-04.md docs/CHANGELOG_2026-04-21.md
```

沒有任何產品程式碼被修改。
