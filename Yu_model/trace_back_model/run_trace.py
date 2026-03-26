"""
trace_back_model 獨立啟動腳本
==============================
不需要先跑 Wei_model，直接從原始 CSV 跑完整追溯流程。

用法：
    # 最簡單（用 adjust_data/train/ 預設資料）
    python run_trace.py

    # 指定資料目錄
    python run_trace.py --data_dir ../../adjust_data/train

    # 同時使用 Wei_model 輸出的風險評分（讓源頭節點的分數更準）
    python run_trace.py --risk_csv ../../Wei_model/output/all_user_risk_scores.csv

    # 跳過 CXGNN 驗證（只跑 Method B，速度快）
    python run_trace.py --skip_cxgnn

    # 指定最大追溯跳數
    python run_trace.py --max_hops 4
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

# ── 確保 trace_back_model 本身在 sys.path ────────────────────────────────────
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


# ── 預設資料路徑 ─────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR = ROOT_DIR / "adjust_data" / "train"
DEFAULT_OUTPUT   = THIS_DIR / "output"


# ── 資料載入 ─────────────────────────────────────────────────────────────────

def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    載入 crypto_transfer / twd_transfer / user_info。
    回傳 (crypto_df, twd_df, user_info_df)。
    """
    required = {
        "crypto_transfer": "crypto_transfer",
        "twd_transfer":    "twd_transfer",
        "user_info":       "user_info",
    }
    dfs = {}
    for stem, key in required.items():
        # 支援 <stem>.csv 和 <stem>_train.csv 兩種命名
        candidates = [
            data_dir / f"{stem}.csv",
            data_dir / f"{stem}_train.csv",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(
                f"找不到資料檔：{candidates[0]} 或 {candidates[1]}"
            )
        dfs[key] = pd.read_csv(path)
        logger.info("  載入 %-30s  %d 筆", path.name, len(dfs[key]))

    return dfs["crypto_transfer"], dfs["twd_transfer"], dfs["user_info"]


def build_risk_df(user_info: pd.DataFrame) -> pd.DataFrame:
    """
    建立 risk_df（index=user_id，含 risk_score 與 status）。
    直接使用 user_info.status 作為標準答案（0=正常, 1=詐騙）。
    """
    logger.info("  使用 user_info.status 作為詐騙標記（ground truth）")
    df = user_info[["user_id", "status"]].copy()
    df["risk_score"] = df["status"].astype(float)
    return df.set_index("user_id")


# ── 主流程 ───────────────────────────────────────────────────────────────────

def main(
    data_dir:   Path  = DEFAULT_DATA_DIR,
    output_dir: Path  = DEFAULT_OUTPUT,
    max_hops:   int   = 5,
    min_amount: float = 0.0,
    skip_cxgnn: bool  = False,
    num_epochs: int   = 50,
) -> pd.DataFrame:

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 載入資料 ──────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("[Step 1] 載入資料")
    print("="*55)
    crypto_df, twd_df, user_info = load_data(data_dir)

    risk_df = build_risk_df(user_info)

    # ── 找出詐騙節點 ──────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("[Step 2] 識別詐騙節點")
    print("="*55)

    fraud_ids = risk_df[risk_df["status"] == 1].index.tolist()
    print(f"  詐騙節點數：{len(fraud_ids)}  （來源：user_info.status）")

    if not fraud_ids:
        print("  [警告] 找不到任何詐騙節點，結束。")
        return pd.DataFrame()

    # ── Method B：時序反向追溯 ────────────────────────────────────────────
    print("\n" + "="*55)
    print("[Step 3] Method B — 時序反向 BFS 追溯")
    print("="*55)
    print(f"  max_hops={max_hops}  min_amount={min_amount}")

    tracer  = FraudSourceTracer(crypto_df, twd_df, risk_df)
    results = tracer.trace(fraud_ids, max_hops=max_hops, min_amount=min_amount)

    print(f"  成功追溯：{len(results)} / {len(fraud_ids)} 個詐騙節點")

    if not results:
        print("  [警告] 所有節點都沒有找到上游路徑，請確認資料中有交易邊。")
        return pd.DataFrame()

    # 統計
    hop_counts = [r.hop_count for r in results]
    print(f"  平均跳數：{sum(hop_counts)/len(hop_counts):.1f}")
    print(f"  最長路徑：{max(hop_counts)} hops")

    wallet_sources = sum(1 for r in results if r.source_node_type == "wallet")
    user_sources   = sum(1 for r in results if r.source_node_type == "user")
    bank_sources   = sum(1 for r in results if r.source_node_type == "bank")
    print(f"  源頭類型：wallet={wallet_sources}  user={user_sources}  bank={bank_sources}")

    # ── Method C：CXGNN 因果驗證（可選）──────────────────────────────────
    if not skip_cxgnn:
        print("\n" + "="*55)
        print("[Step 4] Method C — CXGNN 因果驗證")
        print("="*55)

        fraud_labels = {
            f"user_{uid}": int(row["status"])
            for uid, row in risk_df.iterrows()
            if "status" in risk_df.columns
        }

        adapter   = CXGNNAdapter(num_epochs=num_epochs)
        validated = adapter.validate(results, fraud_labels)
        ok_count  = sum(1 for v in validated if v.validation_ok)
        print(f"  CXGNN 驗證成功：{ok_count} / {len(validated)}")

        out_df = adapter.to_dataframe(validated)
    else:
        print("\n[Step 4] 跳過 CXGNN（--skip_cxgnn）")
        out_df = tracer.to_dataframe(results)

    # ── 輸出 ──────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("[Step 5] 輸出結果")
    print("="*55)

    out_path = output_dir / "fraud_chains.csv"
    out_df.to_csv(out_path, index=False)
    print(f"  fraud_chains.csv  →  {out_path}")
    print(f"  總筆數：{len(out_df)}")

    # 印出前 5 筆摘要
    if not out_df.empty:
        print("\n  前 5 筆追溯結果：")
        preview_cols = [
            "fraud_node_id", "source_node_id", "source_node_type",
            "hop_count", "total_amount_twd",
        ]
        if "causal_confidence" in out_df.columns:
            preview_cols.append("causal_confidence")
        print(out_df[preview_cols].head().to_string(index=False))

    print(f"\n{'='*55}")
    print("  完成！")
    print(f"{'='*55}")

    return out_df


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud Source Tracer — 獨立啟動")
    parser.add_argument(
        "--data_dir", type=str,
        default=str(DEFAULT_DATA_DIR),
        help="原始 CSV 資料目錄（含 crypto_transfer.csv / twd_transfer.csv / user_info.csv）",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=str(DEFAULT_OUTPUT),
        help="輸出目錄（預設 trace_back_model/output/）",
    )
    parser.add_argument(
        "--max_hops", type=int, default=5,
        help="最大追溯跳數（預設 5）",
    )
    parser.add_argument(
        "--min_amount", type=float, default=0.0,
        help="忽略金額低於此值的交易邊（TWD，預設 0）",
    )
    parser.add_argument(
        "--skip_cxgnn", action="store_true",
        help="跳過 CXGNN 因果驗證（只跑 Method B，速度更快）",
    )
    parser.add_argument(
        "--cxgnn_epochs", type=int, default=50,
        help="CXGNN 訓練輪數（預設 50）",
    )

    args = parser.parse_args()
    main(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        max_hops=args.max_hops,
        min_amount=args.min_amount,
        skip_cxgnn=args.skip_cxgnn,
        num_epochs=args.cxgnn_epochs,
    )
