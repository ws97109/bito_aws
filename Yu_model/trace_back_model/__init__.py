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

    # Step 1：載入 user_info（Wei_model 輸出的已標記詐騙節點）
    bl   = pd.read_csv("Wei_model/output/blacklist_analysis.csv")
    w2b  = pd.read_csv("Wei_model/output/white_to_black.csv")
    user_info = pd.concat([bl, w2b], ignore_index=True).drop_duplicates("user_id")

    # Step 2：載入交易資料與 GNN 圖資料
    crypto   = pd.read_csv("adjust_data/train/crypto_transfer_train.csv")
    twd      = pd.read_csv("adjust_data/train/twd_transfer_train.csv")
    gnn_edge = pd.read_csv("Wei_model/output/gnn_edge_list.csv")
    gnn_node = pd.read_csv("Wei_model/output/gnn_node_list.csv")

    # Step 3：建立 risk_df（直接使用 Wei_model 的 risk_score；全部皆為詐騙節點）
    risk_df  = user_info[["user_id", "risk_score"]].drop_duplicates("user_id").set_index("user_id")
    fraud_ids = risk_df.index.tolist()

    # Step 4：建立追溯器，找詐騙源頭
    tracer = FraudSourceTracer(crypto, risk_df)
    chains = tracer.trace(fraud_ids, max_hops=5)

    # Step 4（可選）：CXGNN 因果驗證
    fraud_labels = {f"user_{uid}": 1 for uid in risk_df.index}
    adapter      = CXGNNAdapter()
    validated    = adapter.validate(chains, fraud_labels)
    df_out       = adapter.to_dataframe(validated)

    # Step 5：輸出給前端
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
