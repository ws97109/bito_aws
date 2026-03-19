#!/bin/bash
# ============================================================
# ARM-GraphSAGE → XGBoost Full Pipeline Runner
# ============================================================
# Usage:
#   bash run_pipeline.sh [--labels /path/to/labels.csv] [--skip_gnn]
#
# Required env:
#   Run from the directory that CONTAINS arm_xgboost/
#   e.g.  cd /mnt/nfs/maokao_1/Bio_AWS_Workshop/ARM-GraphSAGE
#         bash arm_xgboost/run_pipeline.sh --labels /path/to/fraud_labels.csv
# ============================================================

set -e   # exit on any error

# ── Defaults ────────────────────────────────────────────────
CONFIG="arm_xgboost/configs/config.yaml"
RESULTS="./results"
FEATURES="${RESULTS}/features"
GRAPHS="${RESULTS}/graphs"
LABELS=""
SKIP_GNN=false

# ── Argument parsing ─────────────────────────────────────────
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --labels)   LABELS="$2";    shift ;;
        --skip_gnn) SKIP_GNN=true           ;;
        --config)   CONFIG="$2";    shift ;;
        --results)  RESULTS="$2";   shift ;
                    FEATURES="${RESULTS}/features";
                    GRAPHS="${RESULTS}/graphs" ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

echo ""
echo "=================================================="
echo "  ARM-GraphSAGE → XGBoost Pipeline"
echo "=================================================="
echo "  Config  : ${CONFIG}"
echo "  Results : ${RESULTS}"
echo "  Labels  : ${LABELS:-<none>}"
echo "  Skip GNN: ${SKIP_GNN}"
echo "=================================================="
echo ""

# ── Module 1: Feature engineering ───────────────────────────
echo ">>> MODULE 1: Feature Engineering"
python arm_xgboost/01_feature_pipeline.py \
    --config "${CONFIG}" \
    --output "${FEATURES}/user_features.csv"

echo ""

# ── Module 2: ARM mining ─────────────────────────────────────
echo ">>> MODULE 2: ARM Mining"
python arm_xgboost/02_arm_mining.py \
    --config       "${CONFIG}" \
    --features     "${FEATURES}/user_features.csv" \
    --out_rules    "${FEATURES}/arm_rules.pkl" \
    --out_features "${FEATURES}/arm_features.csv"

echo ""

# ── Module 3: Causal Forest ──────────────────────────────────
echo ">>> MODULE 3: Causal Forest (CATE)"
LABELS_ARG=""
if [ -n "${LABELS}" ]; then
    LABELS_ARG="--labels ${LABELS}"
fi

python arm_xgboost/03_causal_forest.py \
    --config   "${CONFIG}" \
    --features "${FEATURES}/user_features.csv" \
    ${LABELS_ARG} \
    --output   "${FEATURES}/cate_scores.csv"

echo ""

# ── Module 4: Feature fusion ─────────────────────────────────
echo ">>> MODULE 4: Feature Fusion"
python arm_xgboost/04_feature_fusion.py \
    --config  "${CONFIG}" \
    --tabular "${FEATURES}/user_features.csv" \
    --arm     "${FEATURES}/arm_features.csv" \
    --cate    "${FEATURES}/cate_scores.csv" \
    --output  "${FEATURES}/node_features.csv"

echo ""

# ── Module 5: Graph construction ─────────────────────────────
echo ">>> MODULE 5: Heterogeneous Graph Construction"
python arm_xgboost/05_build_graph.py \
    --config        "${CONFIG}" \
    --node_features "${FEATURES}/node_features.csv" \
    --output        "${GRAPHS}/graph.pt"

echo ""

# ── Module 6: GNN training + embedding extraction ────────────
if [ "${SKIP_GNN}" = true ]; then
    echo ">>> MODULE 6: Skipped (--skip_gnn)"
    if [ ! -f "${FEATURES}/gnn_embeddings.npy" ]; then
        echo "ERROR: --skip_gnn set but gnn_embeddings.npy does not exist."
        echo "       Run Module 6 at least once before using --skip_gnn."
        exit 1
    fi
else
    echo ">>> MODULE 6: GNN Pre-training & Embedding Extraction"

    GNN_LABELS_ARG=""
    if [ -n "${LABELS}" ]; then
        GNN_LABELS_ARG="--labels ${LABELS}"
    fi

    GNN_CATE_ARG=""
    if [ -f "${FEATURES}/cate_scores.csv" ]; then
        GNN_CATE_ARG="--cate ${FEATURES}/cate_scores.csv"
    fi

    python arm_xgboost/06_train_gnn_embedding.py \
        --config  "${CONFIG}" \
        --graph   "${GRAPHS}/graph.pt" \
        ${GNN_LABELS_ARG} \
        ${GNN_CATE_ARG} \
        --out_emb "${FEATURES}/gnn_embeddings.npy" \
        --out_map "${FEATURES}/gnn_user_id_map.csv"
fi

echo ""

# ── Module 7: XGBoost classifier ─────────────────────────────
echo ">>> MODULE 7: XGBoost Classifier"

if [ -z "${LABELS}" ]; then
    echo "ERROR: --labels is required for Module 7 (XGBoost needs fraud labels)."
    exit 1
fi

XGB_ARM_ARG=""
if [ -f "${FEATURES}/arm_features.csv" ]; then
    XGB_ARM_ARG="--arm ${FEATURES}/arm_features.csv"
fi

XGB_CATE_ARG=""
if [ -f "${FEATURES}/cate_scores.csv" ]; then
    XGB_CATE_ARG="--cate ${FEATURES}/cate_scores.csv"
fi

python arm_xgboost/07_xgboost_classifier.py \
    --config  "${CONFIG}" \
    --emb     "${FEATURES}/gnn_embeddings.npy" \
    --emb_map "${FEATURES}/gnn_user_id_map.csv" \
    --tabular "${FEATURES}/user_features.csv" \
    --labels  "${LABELS}" \
    ${XGB_ARM_ARG} \
    ${XGB_CATE_ARG}

echo ""
echo "=================================================="
echo "  PIPELINE COMPLETE"
echo "  Outputs:"
echo "    Fraud scores : ${FEATURES}/fraud_scores.csv"
echo "    XGB model    : ${RESULTS}/models/xgb_model.pkl"
echo "    GNN model    : ${RESULTS}/models/gnn_best.pt"
echo "    SHAP report  : ${RESULTS}/reports/shap_importance.csv"
echo "    Metrics      : ${RESULTS}/reports/xgb_test_metrics.json"
echo "=================================================="
