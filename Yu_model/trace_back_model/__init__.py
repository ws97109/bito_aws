"""
trace_back_model
================
詐騙源頭追溯插件，獨立於 Wei_model 主流程之外。

模組：
    fraud_source_tracer  — Method B：時序反向 BFS 追溯資金鏈
    cxgnn_adapter        — Method C：CXGNN 因果子圖驗證

快速用法：
    import pandas as pd
    from trace_back_model import FraudSourceTracer, CXGNNAdapter
    # Step 1：載入原始資料
    user_info = pd.read_csv("adjust_data/train/user_info.csv")
    crypto    = pd.read_csv("adjust_data/train/crypto_transfer_train.csv")
    twd       = pd.read_csv("adjust_data/train/twd_transfer_train.csv")

    # Step 2：建立 risk_df（以 status ground truth 為標準）
    risk_df   = user_info[["user_id","status"]].copy()
    risk_df["risk_score"] = risk_df["status"].astype(float)
    risk_df   = risk_df.set_index("user_id")

    # Step 3：建立追溯器，找詐騙源頭
    tracer    = FraudSourceTracer(crypto, twd, risk_df)
    fraud_ids = risk_df[risk_df["status"] == 1].index.tolist()
    chains   = tracer.trace(fraud_ids, max_hops=5)

    # Step 3（可選）：CXGNN 因果驗證
    fraud_labels = {f"user_{uid}": int(s) for uid, s in risk_df["status"].items()}
    adapter      = CXGNNAdapter()
    validated    = adapter.validate(chains, fraud_labels)
    df_out       = adapter.to_dataframe(validated)

    # Step 4：輸出給前端
    df_out.to_csv("output/fraud_chains.csv", index=False)
"""
from .fraud_source_tracer import FraudSourceTracer, TraceResult, TraceEdge
from .cxgnn_adapter import CXGNNAdapter, ValidatedChain

__all__ = [
    "FraudSourceTracer",
    "TraceResult",
    "TraceEdge",
    "CXGNNAdapter",
    "ValidatedChain",
]
