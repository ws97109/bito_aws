"""
CXGNN Adapter — Method C
把 FraudSourceTracer 找到的追溯路徑子圖，轉換成 CXGNN 格式，
用因果推論驗證哪些邊是「真正因果的」而非只是時序巧合。

CXGNN 原始設計：graph-level classification → 找因果子圖
本 adapter 的改編：
  - 把每條追溯路徑當作一個獨立小圖
  - 路徑中已知詐騙節點 node_label = 1，其餘 = 0
  - CXGNN alg2 找出哪些節點最有因果貢獻
  - 把因果分數反映到路徑的邊上，輸出 causal_score

用法：
    from cxgnn_adapter import CXGNNAdapter
    from fraud_source_tracer import FraudSourceTracer, TraceResult

    adapter  = CXGNNAdapter(cxgnn_dir="../CXGNN")
    results  = adapter.validate(trace_results, fraud_labels)
    df_out   = adapter.to_dataframe(results)
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from fraud_source_tracer import TraceResult, USER_PREFIX

logger = logging.getLogger(__name__)


# ── 輸出資料類別（immutable）────────────────────────────────────────────────

@dataclass(frozen=True)
class ValidatedChain:
    """CXGNN 驗證後的追溯鏈。"""
    trace:          TraceResult
    causal_scores:  dict[str, float]   # node_id → causal contribution score (0~1)
    best_causal_node: str              # CXGNN 認為最具因果貢獻的節點
    causal_confidence: float           # 整條鏈的平均因果分數
    validation_ok:  bool               # CXGNN 是否成功執行


# ── 主類別 ───────────────────────────────────────────────────────────────────

class CXGNNAdapter:
    """
    Parameters
    ----------
    num_epochs : int
        CXGNN alg1 的訓練輪數，路徑短時可用較少 epochs（預設 50）。
    """

    def __init__(self, num_epochs: int = 50) -> None:
        self._num_epochs = num_epochs
        self._alg2        = None
        self._CausalGraph = None
        self._loaded      = False

    def _lazy_load(self) -> bool:
        """載入同資料夾內的 CXGNN 模組（alg1/alg2/causal）。"""
        if self._loaded:
            return True

        # 確保 trace_back_model 自己的目錄在 sys.path，讓 alg2 可以 import alg1
        this_dir = str(Path(__file__).parent)
        if this_dir not in sys.path:
            sys.path.insert(0, this_dir)

        try:
            import alg2
            from causal import CausalGraph

            self._alg2        = alg2
            self._CausalGraph = CausalGraph
            self._loaded      = True
            logger.info("CXGNN modules loaded from %s", this_dir)
            return True
        except ImportError as exc:
            logger.warning(
                "CXGNN import failed (%s). "
                "Validation will be skipped; causal_score will be set to 0.",
                exc,
            )
            return False

    # ── 驗證流程 ─────────────────────────────────────────────────────────────

    def validate(
        self,
        trace_results: list[TraceResult],
        fraud_labels:  dict[str, int],
    ) -> list[ValidatedChain]:
        """
        對每條追溯鏈執行 CXGNN 因果驗證。

        Parameters
        ----------
        trace_results : list[TraceResult]
            FraudSourceTracer.trace() 的輸出。
        fraud_labels : dict[str → 0/1]
            已知詐騙標記，key = node_id（含 'user_' 前綴），value = 0 or 1。
            可從 risk_df['status'] 建立：
                {f"user_{uid}": int(s) for uid, s in risk_df['status'].items()}

        Returns
        -------
        list[ValidatedChain]
        """
        ok = self._lazy_load()
        results = []
        for trace in trace_results:
            chain = (
                self._validate_one(trace, fraud_labels)
                if ok
                else self._fallback(trace)
            )
            results.append(chain)
        return results

    def _validate_one(
        self,
        trace:        TraceResult,
        fraud_labels: dict[str, int],
    ) -> ValidatedChain:
        """
        把單條路徑轉成 CXGNN CausalGraph 並執行 alg2。
        """
        nodes = list(trace.path_nodes)

        # ── 節點重新編號（CXGNN 需要 int node id）────────────────────────
        node_to_int: dict[str, int] = {n: i for i, n in enumerate(nodes)}
        int_to_node: dict[int, str] = {i: n for n, i in node_to_int.items()}
        n_nodes = len(nodes)

        # ── 路徑邊轉成 CXGNN path 格式（undirected int tuple list）────────
        path_tuples: list[tuple[int, int]] = []
        for edge in trace.path_edges:
            if edge.src in node_to_int and edge.dst in node_to_int:
                path_tuples.append(
                    (node_to_int[edge.src], node_to_int[edge.dst])
                )

        if not path_tuples:
            return self._fallback(trace)

        # ── 建立 CausalGraph ──────────────────────────────────────────────
        V = list(range(n_nodes))
        try:
            cg = self._CausalGraph(V=V, path=path_tuples)
        except Exception as exc:
            logger.warning("CausalGraph build failed for %s: %s", trace.fraud_node_id, exc)
            return self._fallback(trace)

        # ── 建立 node_label DataFrame（詐騙=1，其他=0）───────────────────
        labels = {
            i: fraud_labels.get(int_to_node[i], 0)
            for i in range(n_nodes)
        }
        data = pd.DataFrame.from_dict(
            {"node_label": labels}, orient="columns"
        )
        role_id = [labels[i] for i in range(n_nodes)]

        # ── 執行 CXGNN alg2 ──────────────────────────────────────────────
        try:
            models, _, _, best_exp_p, _, best_new_v, best_node = self._alg2.alg_2(
                cg, self._num_epochs, data, role_id
            )
        except Exception as exc:
            logger.warning("CXGNN alg2 failed for %s: %s", trace.fraud_node_id, exc)
            return self._fallback(trace)

        # ── 把每個節點的 expected_p 當作因果分數 ─────────────────────────
        causal_scores = {
            int_to_node[node_int]: float(info["expected_p"])
            for node_int, info in models.items()
            if node_int in int_to_node
        }

        best_causal_node = int_to_node.get(best_node, trace.source_node_id)
        avg_confidence   = (
            sum(causal_scores.values()) / len(causal_scores)
            if causal_scores else 0.0
        )

        return ValidatedChain(
            trace=trace,
            causal_scores=causal_scores,
            best_causal_node=best_causal_node,
            causal_confidence=round(avg_confidence, 4),
            validation_ok=True,
        )

    def _fallback(self, trace: TraceResult) -> ValidatedChain:
        """CXGNN 無法執行時的降級結果（causal_score 全為 0）。"""
        return ValidatedChain(
            trace=trace,
            causal_scores={n: 0.0 for n in trace.path_nodes},
            best_causal_node=trace.source_node_id,
            causal_confidence=0.0,
            validation_ok=False,
        )

    # ── 輸出 ─────────────────────────────────────────────────────────────────

    def to_dataframe(self, chains: list[ValidatedChain]) -> pd.DataFrame:
        """
        把 ValidatedChain 列表轉成扁平 DataFrame。

        額外欄位（在 FraudSourceTracer.to_dataframe 的基礎上）：
            best_causal_node    CXGNN 認為最具因果力的節點
            causal_confidence   整條路徑的平均因果分數 (0~1)
            causal_scores       JSON dict，每個節點的因果分數
            validation_ok       CXGNN 是否成功執行
        """
        import json
        rows = []
        for vc in chains:
            r = vc.trace
            edge_list = [
                {
                    "src":       e.src,
                    "dst":       e.dst,
                    "timestamp": str(e.timestamp) if e.timestamp is not None else None,
                    "amount":    round(e.amount, 2),
                    "type":      e.edge_type,
                    "causal_score": round(
                        vc.causal_scores.get(e.src, 0.0) +
                        vc.causal_scores.get(e.dst, 0.0), 4
                    ),
                }
                for e in r.path_edges
            ]
            rows.append({
                "fraud_node_id":      r.fraud_node_id,
                "source_node_id":     r.source_node_id,
                "source_node_type":   r.source_node_type,
                "path_nodes":         json.dumps(list(r.path_nodes)),
                "path_edges":         json.dumps(edge_list),
                "hop_count":          r.hop_count,
                "earliest_tx_time":   r.earliest_tx_time,
                "total_amount_twd":   round(r.total_amount, 2),
                "source_risk_score":  round(r.source_risk_score, 4),
                "best_causal_node":   vc.best_causal_node,
                "causal_confidence":  vc.causal_confidence,
                "causal_scores":      json.dumps(vc.causal_scores),
                "validation_ok":      vc.validation_ok,
            })
        return pd.DataFrame(rows)
