#!/bin/bash

################################################################################
# BitoGuard 快速啟動腳本
# 使用方法: bash QUICK_START.sh
################################################################################

echo "========================================================================"
echo "  BitoGuard 快速啟動與訓練"
echo "========================================================================"
echo ""

# 設定顏色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: 檢查環境
echo -e "${YELLOW}[Step 1/5] 檢查 Python 環境...${NC}"
if ! command -v python &> /dev/null; then
    echo -e "${RED}✗ Python 未安裝！${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python 版本: $(python --version)${NC}"
echo ""

# Step 2: 檢查必要套件
echo -e "${YELLOW}[Step 2/5] 檢查必要套件...${NC}"
python -c "import torch, torch_geometric, pandas, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ 缺少必要套件！請執行: pip install -r requirements.txt${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 所有必要套件已安裝${NC}"
echo ""

# Step 3: 檢查資料檔案
echo -e "${YELLOW}[Step 3/5] 檢查資料檔案...${NC}"
DATA_DIR="../../Data"
REQUIRED_FILES=("user_info.csv" "twd_transfer.csv" "usdt_swap.csv" "usdt_twd_trading.csv" "crypto_transfer.csv")
MISSING_FILES=0

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$DATA_DIR/$file" ]; then
        echo -e "${GREEN}✓ $file${NC}"
    else
        echo -e "${RED}✗ $file (缺少)${NC}"
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
done

if [ $MISSING_FILES -gt 0 ]; then
    echo -e "${RED}警告: 缺少 $MISSING_FILES 個資料檔案${NC}"
    read -p "是否繼續? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo ""

# Step 4: 執行 Pipeline (如果尚未執行)
echo -e "${YELLOW}[Step 4/5] 執行資料處理 Pipeline...${NC}"
if [ ! -f "results/graphs/graph.pt" ]; then
    echo "開始執行 Pipeline (這可能需要 5-15 分鐘)..."
    bash run_pipeline.sh

    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Pipeline 執行失敗！${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Pipeline 執行完成${NC}"
else
    echo -e "${GREEN}✓ 圖資料已存在，跳過 Pipeline${NC}"
fi
echo ""

# Step 5: 開始訓練
echo -e "${YELLOW}[Step 5/5] 開始訓練模型...${NC}"
echo "========================================================================"
echo ""

# 讀取配置中的 epochs 數量
EPOCHS=$(grep -A 20 "^training:" configs/config.yaml | grep "epochs:" | awk '{print $2}')
echo "訓練配置:"
echo "  - Epochs: $EPOCHS"
echo "  - 配置文件: configs/config.yaml"
echo ""

read -p "按 Enter 開始訓練 (或 Ctrl+C 取消)..."

python train_example.py

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo -e "${GREEN}✓ 訓練完成！${NC}"
    echo "========================================================================"
    echo ""
    echo "生成的檔案:"
    if [ -f "results/models/best_model.pt" ]; then
        echo -e "  ${GREEN}✓${NC} results/models/best_model.pt"
    fi
    if [ -f "results/features/gnn_embeddings.npy" ]; then
        echo -e "  ${GREEN}✓${NC} results/features/gnn_embeddings.npy"
    fi
    echo ""
else
    echo ""
    echo "========================================================================"
    echo -e "${RED}✗ 訓練失敗！請檢查錯誤訊息${NC}"
    echo "========================================================================"
    exit 1
fi
