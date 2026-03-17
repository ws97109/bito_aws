#!/bin/bash

################################################################################
# BitoGuard Complete Pipeline Execution Script
#
# This script runs the complete BitoGuard pipeline from raw data to trained model
################################################################################

set -e  # Exit on error

echo "=================================================================================================="
echo "                         BitoGuard GraphSAGE-ARM-CF Pipeline                                      "
echo "=================================================================================================="
echo ""

# Configuration
CONFIG_PATH="configs/config.yaml"
OUTPUT_DIR="results"
FEATURES_DIR="${OUTPUT_DIR}/features"
GRAPHS_DIR="${OUTPUT_DIR}/graphs"
MODELS_DIR="${OUTPUT_DIR}/models"

# Create output directories
echo "[SETUP] Creating output directories..."
mkdir -p $FEATURES_DIR
mkdir -p $GRAPHS_DIR
mkdir -p $MODELS_DIR
echo "✓ Directories created"
echo ""

################################################################################
# Step 1: Feature Engineering
################################################################################
echo "=================================================================================================="
echo "STEP 1: FEATURE ENGINEERING PIPELINE"
echo "=================================================================================================="

python bitoguard_feature_pipeline.py \
    --config $CONFIG_PATH \
    --output ${FEATURES_DIR}/user_features.csv

if [ $? -ne 0 ]; then
    echo "✗ Feature engineering failed"
    exit 1
fi

echo "✓ Feature engineering completed"
echo ""

################################################################################
# Step 2: ARM Mining
################################################################################
echo "=================================================================================================="
echo "STEP 2: ASSOCIATION RULE MINING"
echo "=================================================================================================="

python arm_mining.py \
    --config $CONFIG_PATH \
    --features ${FEATURES_DIR}/user_features.csv \
    --output_rules ${FEATURES_DIR}/arm_rules.pkl \
    --output_features ${FEATURES_DIR}/arm_features.csv

if [ $? -ne 0 ]; then
    echo "✗ ARM mining failed"
    exit 1
fi

echo "✓ ARM mining completed"
echo ""

################################################################################
# Step 3: Causal Forest
################################################################################
echo "=================================================================================================="
echo "STEP 3: CAUSAL FOREST ESTIMATION"
echo "=================================================================================================="

python causal_forest.py \
    --config $CONFIG_PATH \
    --features ${FEATURES_DIR}/user_features.csv \
    --output ${FEATURES_DIR}/cate_scores.csv

if [ $? -ne 0 ]; then
    echo "⚠ Causal Forest estimation failed (possibly econml not installed)"
    echo "Creating dummy CATE scores instead..."

    python create_dummy_cate.py \
        --features ${FEATURES_DIR}/user_features.csv \
        --output ${FEATURES_DIR}/cate_scores.csv

    if [ $? -ne 0 ]; then
        echo "✗ Failed to create dummy CATE scores"
        exit 1
    fi
fi

echo "✓ Causal Forest completed"
echo ""

################################################################################
# Step 4: Feature Fusion
################################################################################
echo "=================================================================================================="
echo "STEP 4: FEATURE FUSION"
echo "=================================================================================================="

python feature_fusion.py \
    --config $CONFIG_PATH \
    --tabular ${FEATURES_DIR}/user_features.csv \
    --arm ${FEATURES_DIR}/arm_features.csv \
    --cate ${FEATURES_DIR}/cate_scores.csv \
    --output ${FEATURES_DIR}/node_features.csv

if [ $? -ne 0 ]; then
    echo "✗ Feature fusion failed"
    exit 1
fi

echo "✓ Feature fusion completed"
echo ""

################################################################################
# Step 5: Heterogeneous Graph Construction
################################################################################
echo "=================================================================================================="
echo "STEP 5: HETEROGENEOUS GRAPH CONSTRUCTION"
echo "=================================================================================================="

python bitoguard_hetero_graph.py \
    --config $CONFIG_PATH \
    --node_features ${FEATURES_DIR}/node_features.csv \
    --arm_rules ${FEATURES_DIR}/arm_rules.pkl \
    --output ${GRAPHS_DIR}/graph.pt

if [ $? -ne 0 ]; then
    echo "✗ Graph construction failed"
    exit 1
fi

echo "✓ Graph construction completed"
echo ""

################################################################################
# Step 6: Model Training (Optional - commented out by default)
################################################################################
echo "=================================================================================================="
echo "STEP 6: MODEL TRAINING (Optional)"
echo "=================================================================================================="

# Uncomment to train model
# python train_example.py

echo "⚠ Model training skipped (uncomment in script to enable)"
echo ""

################################################################################
# Pipeline Summary
################################################################################
echo "=================================================================================================="
echo "                              PIPELINE EXECUTION COMPLETE                                         "
echo "=================================================================================================="
echo ""
echo "Output files generated:"
echo "  Features:"
echo "    - ${FEATURES_DIR}/user_features.csv"
echo "    - ${FEATURES_DIR}/arm_features.csv"
echo "    - ${FEATURES_DIR}/arm_rules.pkl"
echo "    - ${FEATURES_DIR}/cate_scores.csv"
echo "    - ${FEATURES_DIR}/node_features.csv"
echo ""
echo "  Graphs:"
echo "    - ${GRAPHS_DIR}/graph.pt"
echo ""
echo "Next steps:"
echo "  1. Review feature statistics and ARM rules"
echo "  2. Run model training: python train_example.py"
echo "  3. Evaluate model performance"
echo ""
echo "=================================================================================================="
