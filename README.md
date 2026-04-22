<h1 align="center">BitoGuard: Intelligent Compliance Risk Radar</h1>

<p align="center">
  <strong>AI-Powered Blacklist Detection for Cryptocurrency Exchanges — LOO Toxicity Features + Stacking Ensemble (F1 = 0.8277)</strong>
</p>

<p align="center">
  <a href="https://ws97109.github.io/Bio_AWS_Workshop/"><img src="https://img.shields.io/badge/Live_Demo-GitHub_Pages-blue?logo=github" alt="Live Demo"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.9+-blue.svg?logo=python" alt="Python"></a>
  <a href="https://xgboost.ai/"><img src="https://img.shields.io/badge/Stacking-XGB%2FLGB%2FCat-orange.svg" alt="Stacking Ensemble"></a>
  <a href="https://reactjs.org/"><img src="https://img.shields.io/badge/React-18-blue.svg?logo=react" alt="React"></a>
  <a href="https://www.typescriptlang.org/"><img src="https://img.shields.io/badge/TypeScript-5-blue.svg?logo=typescript" alt="TypeScript"></a>
  <img src="https://img.shields.io/badge/🏆_Agent_for_Truth-Hackathon-gold" alt="Hackathon">
</p>

<p align="center">
  <a href="README_ZH.md">中文</a> | <a href="README.md">English</a>
</p>

---

## Demo

<p align="center">
  <img src="assets/demo.gif" alt="BitoGuard Dashboard Demo" width="800">
</p>

> Interactive 3D risk dashboard — real-time visualization of transaction network graphs, node risk scores, and SHAP explainability analysis

---

## Overview

<table>
<tr>
<td width="60%">

Cryptocurrency exchanges face a critical **mule account (blacklisted user)** problem — these accounts are exploited for money laundering, fraud fund transfers, and other illicit activities.

This project analyzes **770,000+ transaction records** to build an end-to-end risk detection system. After an extensive experimentation phase (documented in [`final_model/docs/`](final_model/docs/)), the final architecture is driven by **LOO (Leave-One-Out) Toxicity features** — a graph-signal-to-tabular-feature transformation ported from the [1st-place BitoGuard repo](https://github.com/gttthuang/Bito) that lifted F1 from 0.37 to 0.83.

- **LOO Toxicity Features (14 dims)**: For each user, compute blacklist density across their shared-wallet / shared-IP / direct-transfer neighbours — with leave-one-out self-label removal to prevent target leakage
- **3-Model Stacking Ensemble**: XGBoost + LightGBM (Focal Loss α=0.75, γ=2.0) + CatBoost with a Logistic Regression meta-learner
- **Unsupervised Anomaly Scores**: Isolation Forest + HBOS + LOF as additional base features
- **SHAP Explainability**: Per-case risk attribution + counterfactual suggestions + SSR stability checks
- **Fairness Audit**: Bias detection across gender, age, career, and income

</td>
<td width="40%">

<img src="assets/architecture.svg" alt="System Architecture" width="100%">

</td>
</tr>
</table>

---

## Competition & Events

This project was developed for the [**Agent for Truth: Disinformation Defense Hackathon**](https://www.ai-expo.tw/kiro_hackathon_2026/), tackling the **BitoPro — Cryptocurrency Transaction Security** challenge track.

| Item | Details |
|------|---------|
| **Hackathon** | Agent for Truth: Disinformation Defense Hackathon |
| **Organizers** | DIGITIMES × National Development Council × AWS |
| **Partners** | BitoPro, Gogolook |
| **Date** | March 26–27, 2026 |
| **Venue** | AWS Nanshan Office / Taipei Expo Park (Taiwan AI EXPO 2026) |
| **Challenge** | Cryptocurrency Transaction Security — Mule Account & Fraud Detection |

<table>
<tr>
<td width="50%">
<p align="center">
  <img src="assets/ai-expo-2026.jpeg" alt="Taiwan AI EXPO 2026" width="100%">
</p>
<p align="center"><sub><b>Taiwan AI EXPO 2026</b> — Team presenting BitoGuard at the venue</sub></p>
</td>
<td width="50%">
<p align="center">
  <img src="assets/hackathon-team.jpeg" alt="Agent for Truth Hackathon" width="100%">
</p>
<p align="center"><sub><b>Agent for Truth Hackathon</b> — Competition at Taipei Expo Park</sub></p>
</td>
</tr>
</table>

---

## Key Features

| | | | |
|:---:|:---:|:---:|:---:|
| **LOO Toxicity (14 dims)** | **Feature Engineering (81)** | **3-Model Stacking** | **SHAP Explainability** |
| Wallet / IP / neighbour | 11 categories, 77 selected | XGB + LGB (Focal) + Cat | Global + Local + Counterfactual |

| | | | |
|:---:|:---:|:---:|:---:|
| **Fairness Audit** | **Interactive Dashboard** | **Anomaly Detection** | **Multi-threshold Submissions** |
| 4-dimension bias check | React + D3 force-graph | IF / HBOS / LOF | max_f1 / max_f2 / min_cost |

---

## Performance

Test-set metrics after the LOO-Toxicity breakthrough (10,204 rows, 328 positives, 5-fold CV stable):

<table>
<tr>
<td width="25%" align="center">
<h3>0.9778</h3>
<sub>AUC-ROC</sub>
</td>
<td width="25%" align="center">
<h3>0.8600</h3>
<sub>AUC-PR</sub>
</td>
<td width="25%" align="center">
<h3>0.7470</h3>
<sub>Recall</sub>
</td>
<td width="25%" align="center">
<h3>0.8277</h3>
<sub>F1-Score</sub>
</td>
</tr>
</table>

<details>
<summary><b>Baseline vs. LOO — the breakthrough</b></summary>

| Metric | Baseline (pre-LOO) | With LOO Toxicity | Δ |
|--------|-------------------|-------------------|---|
| **F1** | 0.3746 | **0.8277** | **+121%** |
| **AUC-PR** | 0.3262 | **0.8600** | +164% |
| **AUC-ROC** | 0.8641 | **0.9778** | +13% |
| **Precision** | 0.3552 | **0.9280** | +161% |
| **Recall** | 0.3963 | **0.7470** | +88% |

Full experiment log: [`final_model/docs/LOO_TOXICITY_REPORT.md`](final_model/docs/LOO_TOXICITY_REPORT.md)

</details>

---

## System Architecture

<p align="center">
  <img src="assets/architecture.svg" alt="System Architecture" width="100%"/>
</p>

---

## Full Pipeline

### Step 1: Data Loading & Validation

Load data from 5 transaction tables + user info table, with strict column type conversion, missing value reporting, and blacklist ratio validation.

### Step 2: Feature Engineering (95 → 77 → 78 dimensions)

Extract **95 features** from 5 raw transaction tables across **11 categories**. 81 are behavioral; the final 14 are **LOO Toxicity** ported from the 1st-place BitoGuard repo.

| # | Category | Count | Key Features | Detection Intent |
|---|----------|-------|--------------|------------------|
| 1 | **User Demographics** | 15 | `kyc_speed_sec`, `account_age_days`, `reg_hour` | KYC anomaly, late-night registration |
| 2 | **Fiat Transactions** | 14 | `twd_dep_sum`, `twd_net_flow`, `twd_smurf_flag` | Net outflow, smurfing |
| 3 | **Crypto Transactions** | 15 | `crypto_wit_sum`, `crypto_wallet_hash_nunique` | Multi-wallet dispersed withdrawals |
| 4 | **Trading/Swap** | 9 | `trading_buy_ratio`, `swap_sum` | One-sided buying, wash trading |
| 5 | **IP & Fund Velocity** | 5 | `ip_unique_count`, `ip_night_ratio`, `fund_stay_sec` | IP hopping, rapid in-out |
| 6 | **Graph Topology** | 5 | `pagerank_score`, `connected_component_size` | Fund hubs, fraud clusters |
| 7 | **Cross-table Derived** | 4 | `total_tx_count`, `weekend_tx_ratio` | Activity acceleration |
| 8 | **AML Red Flags** | 6 | `twd_to_crypto_out_ratio`, `same_day_in_out_count` | Fiat-to-crypto funnel |
| 9 | **Temporal Patterns** | 7 | `tx_interval_mean`, `amount_p90_p10_ratio` | Regular patterns, burst trading |
| 10 | **Composite Risk** | 1 | `composite_risk_score` | Multi-dimension weighted score |
| 11 | **LOO Toxicity** ⭐ | **14** | `w_tox_{max,mean,std}`, `toxic_w_{ratio,count}`, `ip_tox_{mean,max}`, `toxic_ip_count`, `relation_blacklist_{ratio,count}`, `neighbor_tox_{mean,max}`, `toxic_neighbor_count`, `n_neighbors` | **Shared-wallet / IP / peer blacklist density with leave-one-out leakage prevention** |

<details>
<summary><b>LOO Toxicity — the breakthrough</b></summary>

For each `(user i, wallet w)` pair on a shared wallet:

```
tox(w, i) = (bl_count_w − is_bl_i + S × global_rate) / (total_count_w − 1 + S)
S = 50   (smoothing for low-count wallets)
```

The `− is_bl_i` and `− 1` are the **Leave-One-Out** correction — the user's own label never contributes to their own toxicity, preventing target leakage.

**Why it works**: traditional features describe "what this user does" (behavior). LOO toxicity describes "who this user hangs out with" (association). AML is fundamentally a co-conspiracy crime — mule accounts cluster on the same wallets, IPs, and transfer peers. This insight is directly encoded as features, rather than being implicitly learned by a GNN.

**Class discrimination** (mean value, blacklist vs. normal):
- `toxic_ip_count`: 0.002 vs **0.099** (55× lift)
- `toxic_w_count`: 0.34 vs **0.96** (2.8× lift)
- `relation_blacklist_count`: 0.003 vs **0.023** (9× lift)

Credit: [gttthuang/Bito](https://github.com/gttthuang/Bito) — ported faithfully from their `build_toxicity_features` function.

</details>

<details>
<summary><b>Feature Selection Process (95 → 78)</b></summary>

1. **Zero-variance removal**: 1 feature (`has_kyc_level2`)
2. **High-correlation removal** (threshold ≥ 0.95): 14 highly collinear features
3. **Zero-importance removal** (LightGBM): 3 features
4. After selection: **77 dims**
5. Fairness audit removes protected attributes: `is_female`, `age` (−2) → **75 dims**
6. Add anomaly detection scores: IF + HBOS + LOF (+3) → **78 dims final**

</details>

### Step 3: Anomaly Detection Features

| Algorithm | Output Feature | Principle |
|-----------|---------------|-----------|
| **Isolation Forest** | `if_score` | Random partition isolation — shorter paths = more anomalous |
| **HBOS** | `hbos_score` | Histogram density estimation — low-density regions = anomalous |
| **LOF** | `lof_score` | Local outlier factor — greater deviation from neighborhood density |

### Step 4: Stacking Ensemble

Two-layer stacking architecture with three base learners using **different loss functions** to maximize model diversity:

<p align="center">
  <img src="assets/stacking-ensemble.svg" alt="Stacking Ensemble Architecture" width="100%"/>
</p>

<details>
<summary><b>Imbalance Handling Strategy</b></summary>

- **Focal Loss** (LightGBM): α=0.75, γ=2.0 — auto-increase loss weight for borderline samples
- **scale_pos_weight=50** (XGBoost / CatBoost): positive-negative ratio weighting
- **Borderline-SMOTE** (optional): 30% oversampling for borderline minority class only

</details>

### Step 5: SHAP Explainability Analysis

**Global Explanation** — Top 10 Feature Importance (after LOO integration):

| Rank | Feature | Description | SHAP Share | Cumulative |
|------|---------|-------------|-----------|------------|
| 1 | **`w_tox_max`** 🟢 | **Max wallet toxicity (LOO)** | **25.4%** | 25.4% |
| 2 | **`w_tox_mean`** 🟢 | **Mean wallet toxicity (LOO)** | **9.2%** | 34.5% |
| 3 | **`neighbor_tox_mean`** 🟢 | **Mean 2-hop neighbor toxicity** | **8.5%** | 43.1% |
| 4 | `tx_interval_median` | Median transaction interval | 3.0% | 46.1% |
| 5 | `twd_net_flow` | Fiat net inflow | 2.9% | 48.9% |
| 6 | `weekend_tx_ratio` | Weekend transaction ratio | 2.7% | 51.7% |
| 7 | `if_score` | Isolation Forest score | 2.4% | 54.1% |
| 8 | `career_freq` | Career frequency | 2.3% | 56.4% |
| 9 | `account_age_days` | Account age | 2.2% | 58.6% |
| 10 | `swap_sum` | Total swap amount | 2.1% | 60.8% |

> **LOO features combined ≈ 48% of total importance** — the top-3 alone (all LOO) cover 43%. This empirically validates the "association beats behavior" hypothesis for AML detection.

<details>
<summary><b>Local Explanation + Counterfactual + SSR Stability</b></summary>

**Local Explanation**: Per-user SHAP Waterfall Plot — base value → feature push/pull → final prediction

**Counterfactual Analysis**: Auto-suggest which feature adjustments can reduce risk
- Example: "Adjusting KYC completion speed from 54,799s to 0 could reduce risk score by 0.014"

**SSR Stability Verification**: Perturb feature values at ε = 0.05 ~ 0.20 to verify SHAP ranking robustness

</details>

### Step 6: Fairness Audit

| Protected Attribute | Result | DPD | TPR Gap | FPR Gap | DIR |
|---------------------|--------|-----|---------|---------|-----|
| **Gender** | **FAIL** | 0.039 | 0.041 | 0.003 | 0.285 |
| **Age** | **FAIL** | 0.030 | 0.062 | 0.003 | 0.343 |
| **Career Risk** | **FAIL** | 0.009 | 0.122 | 0.001 | 0.729 |
| **Income Source** | **FAIL** | 0.012 | 0.238 | 0.001 | 0.557 |

<details>
<summary><b>Key Findings & Recommendations</b></summary>

With the new model's very high Precision (0.928), absolute FPR drops to <0.5% across all demographic groups, so **FPR Gap passes everywhere**. But the Disparate Impact Ratio (min/max positive-rate) fails because the rare class becomes so selective that small group-level differences amplify into large ratios.

- `is_female` and `age` are already **excluded from the training feature set** (protected attributes)
- `is_high_risk_career` and `is_high_risk_income` are retained — regulatory basis from AML frameworks
- The DIR failures are informational rather than actionable — the model is working *correctly* by being highly selective, but selection rates naturally diverge on small subgroups

Full audit data: [`final_model/output/baseline_loo/fairness_summary.json`](final_model/output/baseline_loo/fairness_summary.json)

</details>

---

## Interactive Risk Dashboard

Built with React + TypeScript + Vite, supporting three viewing modes:

| Mode | Features |
|------|----------|
| **Fraud Mode** | 2D/3D force-directed transaction network graph, KPI stats, high-risk user list, per-node SHAP analysis |
| **FP/FN Mode** | Misclassification analysis + SHAP Waterfall charts explaining model errors |
| **Predict Mode** | 12,753 unlabeled user predictions with risk scores + Top SHAP feature contributions |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Machine Learning** | XGBoost, CatBoost, LightGBM (Focal Loss), Scikit-learn |
| **Graph Features** | LOO Toxicity (vectorized Pandas) — optional: PyTorch Geometric HeteroGNN |
| **Imbalance Handling** | Focal Loss, scale_pos_weight, F-beta / cost-sensitive thresholds |
| **Anomaly Detection** | Isolation Forest, HBOS, LOF |
| **Explainability** | SHAP (TreeExplainer), SSR Stability, Counterfactual Analysis |
| **Fairness** | Demographic Parity, Equalized Odds, Disparate Impact |
| **Frontend** | React 18 + TypeScript + Vite 5 |
| **Visualization** | react-force-graph-2d/3d, Three.js, Recharts |
| **Styling** | Tailwind CSS 3 |

---

## Quick Start

### Model Training

```bash
# Install Python dependencies
pip install xgboost catboost lightgbm scikit-learn shap torch torch_geometric imbalanced-learn pyod

# Run full pipeline (12 automated steps)
cd final_model/model
python main.py --data_dir ../../adjust_data/train --output ../output

# Skip GNN — recommended (ablation showed GNN slightly hurts F1 vs LOO-only)
python main.py --data_dir ../../adjust_data/train --output ../output --skip_gnn

# Post-hoc: produce multi-threshold submissions (max_f1 / max_f2 / min_cost)
python run_final_from_predictions.py \
    --baseline_dir ../output \
    --output_dir ../output/final

# Rebuild relation-graph CSVs for the frontend (no GNN training)
python build_graph_export.py \
    --data_dir ../../adjust_data/train \
    --predict_dir ../../adjust_data/predict \
    --risk_scores ../output/all_user_risk_scores.csv \
    --output_dir ../output
```

### Frontend Dashboard

```bash
cd frontend
npm install
npm run dev          # Dev mode (http://localhost:5173)
npm run build        # Production build
npm test             # Run unit tests (vitest)
```

**Keyboard shortcuts**

| Key | Action |
|-----|--------|
| `1`–`7` | Jump to Overview / Features / Blacklist / FP / FN / Predict / Compare |
| `⌘/Ctrl + K` | Open command palette (search sections & user IDs) |
| `/` | Focus the search input |
| `Shift + P` | Toggle print-friendly mode |

**Deep links**

Every page exposes its state in the URL hash: `#/fp?user=226`, `#/predict?user=124785`, etc. These can be bookmarked or shared.

### Deployment (GitHub Pages)

Deployment is fully automated. Pushing to `main` with any change under `frontend/` triggers the `Deploy frontend to GitHub Pages` workflow (see `.github/workflows/deploy-pages.yml`), which builds with `npm ci && npm run build` and publishes to Pages via `actions/deploy-pages`. No manual build / gh-pages branch maintenance needed.

```bash
git push origin main
gh run watch           # optional — follow the deploy
```

---

## Project Structure

<details>
<summary><b>Expand full directory</b></summary>

```
Bio_AWS_Workshop/
├── final_model/                          # Core ML Pipeline
│   ├── model/
│   │   ├── main.py                    # Main training entry (12 steps)
│   │   ├── Feature_engineering.py     # Feature engineering (11 cats, 95 dims incl. LOO)
│   │   ├── feature_selection.py       # Feature selection (95 → 77)
│   │   ├── anomaly_detection.py       # Unsupervised anomaly detection (IF/HBOS/LOF)
│   │   ├── Gnn_model.py              # Heterogeneous GNN (optional — ablation showed no gain)
│   │   ├── ensemble.py               # Stacking Ensemble (XGB + LGB-Focal + Cat)
│   │   ├── shap_explainer.py         # SHAP explainability + SSR + counterfactual
│   │   ├── fairness_audit.py         # 4-dimension fairness audit
│   │   ├── improved_evaluation.py    # F-beta / cost-sensitive threshold strategies
│   │   ├── pu_learning.py            # PU learning base learner (tested — marginal)
│   │   ├── tabpfn_base.py            # TabPFN foundation model (requires license)
│   │   ├── run_final_from_predictions.py  # Multi-threshold submission producer
│   │   ├── build_graph_export.py     # Relation-graph CSVs for frontend (no GNN)
│   │   ├── build_frontend_data.py    # Derived SHAP/risk CSVs for frontend
│   │   └── pseudo_labeling.py        # Semi-supervised Pseudo-Labeling (tested — marginal)
│   ├── docs/
│   │   ├── LOO_TOXICITY_REPORT.md    # LOO breakthrough writeup
│   │   ├── model_improvements_2026-04.md  # Full experiment log
│   │   └── CHANGELOG_2026-04-21.md
│   └── output/                        # Model outputs
│
├── Yu_model/                          # Fund tracing model
│   └── trace_back_model/             # Fraud fund chain tracking
│
├── frontend/                           # React interactive dashboard
│   ├── src/
│   │   ├── components/               # UI components
│   │   ├── utils/                    # Data processing & graph logic
│   │   └── types/                    # TypeScript types
│   └── output/                        # CSV data for frontend
│
├── assets/                            # Architecture diagrams & media
└── docs/                              # Documentation
```

</details>

---

## Acknowledgments

The LOO Toxicity feature family (Section 2 / Step 2) is a faithful port from the **1st-place BitoGuard repo** by [gttthuang](https://github.com/gttthuang/Bito) from the same competition. Their insight — that shared-wallet / shared-IP blacklist density encoded as tabular features outperforms a GNN on this scale of data — was the single biggest contributor to lifting our F1 from 0.37 to 0.83. Credit where credit is due.

We verified the method reproduces independently on our pipeline and documented the full before/after in [`final_model/docs/LOO_TOXICITY_REPORT.md`](final_model/docs/LOO_TOXICITY_REPORT.md).

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <a href="https://github.com/ws97109/Bio_AWS_Workshop">
    <img src="https://img.shields.io/github/stars/ws97109/Bio_AWS_Workshop.svg?style=social" alt="GitHub Stars">
  </a>
  <a href="https://github.com/ws97109/Bio_AWS_Workshop/fork">
    <img src="https://img.shields.io/github/forks/ws97109/Bio_AWS_Workshop.svg?style=social" alt="GitHub Forks">
  </a>
</p>

<p align="center">
  <sub>BitoGuard — AI-Powered AML Risk Detection for Cryptocurrency Exchanges</sub>
</p>
