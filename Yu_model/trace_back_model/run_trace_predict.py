"""
run_trace_predict.py
====================
使用 GNN 預測結果（white_to_black + blacklist_analysis）作為詐騙節點來源，
並結合 gnn_node_list / gnn_edge_list 擴充圖結構，跑反向追溯。

輸出：trace_back_model/output/fraud_chains_predict.csv

用法：
    python run_trace_predict.py
    python run_trace_predict.py --max_hops 4
    python run_trace_predict.py --skip_cxgnn
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).parent
ROOT_DIR = THIS_DIR.parent.parent          # Bio_AWS_Workshop/
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from fraud_source_tracer import FraudSourceTracer
from cxgnn_adapter import CXGNNAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR   = ROOT_DIR / "adjust_data" / "train"
DEFAULT_WEI_OUTPUT = ROOT_DIR / "Wei_model" / "output"
DEFAULT_OUTPUT     = THIS_DIR / "output"


# ── 資料載入 ──────────────────────────────────────────────────────────────────

def load_tx_data(data_dir: Path) -> pd.DataFrame:
    """載入 crypto_transfer。"""
    candidates = [data_dir / "crypto_transfer.csv", data_dir / "crypto_transfer_train.csv"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"找不到：{candidates[0]} 或 {candidates[1]}")
    df = pd.read_csv(path)
    logger.info("  載入 %-35s  %d 筆", path.name, len(df))
    return df


def load_gnn_graph(wei_output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """載入 gnn_node_list / gnn_edge_list。"""
    node_path = wei_output_dir / "gnn_node_list.csv"
    edge_path = wei_output_dir / "gnn_edge_list.csv"
    gnn_node = pd.read_csv(node_path)
    gnn_edge = pd.read_csv(edge_path, low_memory=False)
    logger.info("  GNN nodes: %d  edges: %d", len(gnn_node), len(gnn_edge))
    return gnn_node, gnn_edge


def load_fraud_nodes(wei_output_dir: Path) -> tuple[list[str], pd.DataFrame]:
    """
    從 white_to_black + blacklist_analysis 取得詐騙節點。
    兩份檔案已經都是詐騙節點（不需要再過濾 status）。

    回傳
    ----
    fraud_ids : list[str]   節點 ID，格式為 "user_XXXXX"
    risk_df   : DataFrame   index=user_id(int)，欄位=[risk_score, status]
    """
    w2b = pd.read_csv(wei_output_dir / "white_to_black.csv")
    bl  = pd.read_csv(wei_output_dir / "blacklist_analysis.csv", low_memory=False)

    # 只取需要的欄位
    w2b = w2b[["user_id", "risk_score"]].copy()
    bl  = bl[["user_id", "risk_score"]].copy()

    combined = pd.concat([w2b, bl], ignore_index=True)
    combined = combined.drop_duplicates(subset="user_id")
    combined["status"] = 1

    logger.info(
        "  white_to_black: %d  blacklist_analysis: %d  合計（去重）: %d",
        len(w2b), len(bl), len(combined),
    )

    risk_df   = combined.set_index("user_id")
    fraud_ids = [f"user_{uid}" for uid in combined["user_id"]]
    return fraud_ids, risk_df


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main(
    data_dir:      Path  = DEFAULT_DATA_DIR,
    wei_output:    Path  = DEFAULT_WEI_OUTPUT,
    output_dir:    Path  = DEFAULT_OUTPUT,
    max_hops:      int   = 5,
    min_amount:    float = 0.0,
    skip_cxgnn:    bool  = False,
    num_epochs:    int   = 50,
) -> pd.DataFrame:

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("[Step 1] 載入交易資料（crypto / twd）")
    print("="*60)
    crypto_df = load_tx_data(data_dir)

    print("\n" + "="*60)
    print("[Step 2] 載入 GNN 圖資料（node_list / edge_list）")
    print("="*60)
    gnn_node, gnn_edge = load_gnn_graph(wei_output)

    print("\n" + "="*60)
    print("[Step 3] 識別詐騙節點（white_to_black + blacklist_analysis）")
    print("="*60)
    fraud_ids, risk_df = load_fraud_nodes(wei_output)
    print(f"  詐騙節點總數：{len(fraud_ids)}")

    print("\n" + "="*60)
    print("[Step 4] Method B — 時序反向 BFS 追溯")
    print("="*60)
    print(f"  max_hops={max_hops}  min_amount={min_amount}")

    tracer  = FraudSourceTracer(crypto_df, risk_df, gnn_edge_df=gnn_edge)
    results = tracer.trace(fraud_ids, max_hops=max_hops, min_amount=min_amount)

    print(f"  成功追溯：{len(results)} / {len(fraud_ids)} 個詐騙節點")

    if not results:
        print("  [警告] 所有節點都沒有找到上游路徑。")
        return pd.DataFrame()

    hop_counts = [r.hop_count for r in results]
    print(f"  平均跳數：{sum(hop_counts)/len(hop_counts):.1f}")
    print(f"  最長路徑：{max(hop_counts)} hops")
    wallet_src = sum(1 for r in results if r.source_node_type == "wallet")
    user_src   = sum(1 for r in results if r.source_node_type == "user")
    print(f"  源頭類型：wallet={wallet_src}  user={user_src}")

    if not skip_cxgnn:
        print("\n" + "="*60)
        print("[Step 5] Method C — CXGNN 因果驗證")
        print("="*60)
        fraud_labels = {
            f"user_{uid}": int(row["status"])
            for uid, row in risk_df.iterrows()
        }
        adapter   = CXGNNAdapter(num_epochs=num_epochs)
        validated = adapter.validate(results, fraud_labels)
        ok_count  = sum(1 for v in validated if v.validation_ok)
        print(f"  CXGNN 驗證成功：{ok_count} / {len(validated)}")
        out_df = adapter.to_dataframe(validated)
    else:
        print("\n[Step 5] 跳過 CXGNN（--skip_cxgnn）")
        out_df = tracer.to_dataframe(results)

    print("\n" + "="*60)
    print("[Step 6] 輸出結果")
    print("="*60)
    out_path = output_dir / "fraud_chains_predict.csv"
    out_df.to_csv(out_path, index=False)
    print(f"  fraud_chains_predict.csv  →  {out_path}")
    print(f"  總筆數：{len(out_df)}")

    if not out_df.empty:
        preview_cols = ["fraud_node_id", "source_node_id", "source_node_type",
                        "hop_count", "total_amount_twd"]
        if "causal_confidence" in out_df.columns:
            preview_cols.append("causal_confidence")
        print("\n  前 5 筆：")
        print(out_df[preview_cols].head().to_string(index=False))

    print(f"\n{'='*60}\n  完成！\n{'='*60}")
    return out_df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud Source Tracer — GNN Predict 版本")
    parser.add_argument("--data_dir",    type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--wei_output",  type=str, default=str(DEFAULT_WEI_OUTPUT))
    parser.add_argument("--output_dir",  type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max_hops",    type=int,   default=5)
    parser.add_argument("--min_amount",  type=float, default=0.0)
    parser.add_argument("--skip_cxgnn", action="store_true")
    parser.add_argument("--cxgnn_epochs", type=int,  default=50)
    args = parser.parse_args()

    main(
        data_dir=Path(args.data_dir),
        wei_output=Path(args.wei_output),
        output_dir=Path(args.output_dir),
        max_hops=args.max_hops,
        min_amount=args.min_amount,
        skip_cxgnn=args.skip_cxgnn,
        num_epochs=args.cxgnn_epochs,
    )