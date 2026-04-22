# 模型改進套件 — 使用說明

本目錄下新增的檔案都是**後處理（post-processing）**，不修改 `main.py` 或 `ensemble.py`。跑法都一樣：先跑一次 `main.py` 產生 baseline，再跑 improved runner。

## 跑法

```bash
cd final_model/model

# 1) Baseline（產生 ensemble_model.joblib）
python main.py --output ../output/baseline            # 完整版含 GNN
# 或
python main.py --output ../output/baseline --skip_gnn # 快速版（~3 分鐘）

# 2) P0-1/P0-2 閾值比較
python run_improved.py \
    --baseline_dir ../output/baseline \
    --output_dir ../output/improved

# 3) P0-3 PU learning 整合（實驗性）
python run_improved_v2.py \
    --pu_method bagging \
    --baseline_dir ../output/baseline \
    --output_dir ../output/improved_v2

# 4) Final（產生多閾值提交檔案）
python run_final.py \
    --baseline_dir ../output/baseline \
    --output_dir ../output/final
```

## 輸出產物

- `output/improved/metrics_improved.json` — P0-1/P0-2 所有變體對照
- `output/improved_v2/metrics_v2.json` — PU learning ablation
- `output/final/FINAL_REPORT.json` — 最終報告
- `output/final/submission_max_f1.csv` — F1 最佳化的提交
- `output/final/submission_max_f2.csv` — F2 最佳化（Recall 優先）
- `output/final/submission_min_cost.csv` — 成本最小化

## 模組索引

| 模組 | 提供 |
|------|------|
| `improved_evaluation.py` | `threshold_max_f1 / max_fbeta / min_cost`，`IsotonicCalibrator` |
| `pu_learning.py` | `pu_oof_probabilities`、`pu_full_fit_predict`（bagging / elkanoto） |
| `tabpfn_base.py` | TabPFN OOF（需要 `TABPFN_TOKEN`） |

## 發現摘要（詳見 `docs/model_improvements_2026-04.md`）

- ✅ **F-beta / 成本閾值**：相同模型、不同閾值 → F2 提升到 0.4224（+12.8% vs 原 F1=0.374）
- ❌ **機率校準**：Isotonic overfit，Sigmoid 保序無增益
- △ **PU learning**：bagging 略降 AUC-PR，Elkanoto 微升 (+0.002)
- ⛔ **TabPFN**：被 Prior Labs license 擋住，需申請 API token
