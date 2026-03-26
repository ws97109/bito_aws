"""
Fraud Source Tracer — Method B
時序反向追溯：從已知詐騙節點沿入邊方向找資金源頭。

演算法：
  1. 把 crypto_transfer / twd_transfer 建成有向圖（A→B 代表 A 把錢打給 B）
  2. 對每個詐騙節點做反向 BFS（沿入邊往上游走）
  3. 找到「最上游源頭」：無更早入邊的節點，或是另一個已知詐騙節點

用法：
    from fraud_source_tracer import FraudSourceTracer

    tracer = FraudSourceTracer(crypto_df, twd_df, risk_df)
    results = tracer.trace(fraud_node_ids=[1001, 1002, 1003], max_hops=5)
    df = tracer.to_dataframe(results)
    df.to_csv("fraud_chains.csv", index=False)
"""
from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── 節點 ID 前綴常數 ────────────────────────────────────────────────────────
USER_PREFIX   = "user_"
WALLET_PREFIX = "wallet_"
BANK_NODE     = "external_bank"


# ── 資料類別（immutable）─────────────────────────────────────────────────────

@dataclass(frozen=True)
class TraceEdge:
    """有向圖中的一條交易邊。"""
    src:       str
    dst:       str
    timestamp: Optional[pd.Timestamp]
    amount:    float
    edge_type: str    # 'deposit' | 'withdrawal' | 'internal' | 'twd_in' | 'twd_out'
    currency:  str = "TWD"


@dataclass(frozen=True)
class TraceResult:
    """單一詐騙節點的反向追溯結果。"""
    fraud_node_id:    str
    source_node_id:   str
    source_node_type: str          # 'user' | 'wallet' | 'bank'
    path_nodes:       tuple[str, ...]   # (source, ..., fraud_node)
    path_edges:       tuple[TraceEdge, ...]
    hop_count:        int
    earliest_tx_time: Optional[pd.Timestamp]
    total_amount:     float
    source_risk_score: float       # 0.0 if wallet / unknown


# ── 主類別 ───────────────────────────────────────────────────────────────────

class FraudSourceTracer:
    """
    Parameters
    ----------
    crypto_df : DataFrame
        crypto_transfer 資料表（需含 created_at, user_id, relation_user_id,
        from_wallet/from_wallet_hash, to_wallet/to_wallet_hash,
        kind, sub_kind, ori_samount, currency）。
    twd_df : DataFrame
        twd_transfer 資料表（需含 created_at, user_id, kind, ori_samount）。
    risk_df : DataFrame
        Index = user_id (int)，需含 'risk_score' 欄位。
        由 Wei_model main.py Step 9 輸出的 user_risk_scores.csv 讀入即可。
    """

    def __init__(
        self,
        crypto_df: pd.DataFrame,
        twd_df:    pd.DataFrame,
        risk_df:   pd.DataFrame,
    ) -> None:
        self._risk_df     = risk_df
        # reverse_graph[dst] = list of TraceEdge（入邊列表）
        self._reverse_adj: dict[str, list[TraceEdge]] = {}
        self._build_reverse_graph(crypto_df, twd_df)

    # ── 建圖 ─────────────────────────────────────────────────────────────────

    def _build_reverse_graph(
        self,
        crypto_df: pd.DataFrame,
        twd_df:    pd.DataFrame,
    ) -> None:
        all_edges: list[TraceEdge] = (
            self._parse_crypto(crypto_df) +
            self._parse_twd(twd_df)
        )

        reverse: dict[str, list[TraceEdge]] = {}
        for edge in all_edges:
            reverse.setdefault(edge.dst, []).append(edge)

        self._reverse_adj = reverse
        logger.info(
            "Reverse graph: %d dst-nodes, %d edges total",
            len(reverse),
            sum(len(v) for v in reverse.values()),
        )

    def _parse_crypto(self, df: pd.DataFrame) -> list[TraceEdge]:
        if df is None or df.empty:
            return []

        df = df.copy()
        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

        from_col = "from_wallet_hash" if "from_wallet_hash" in df.columns else "from_wallet"
        to_col   = "to_wallet_hash"   if "to_wallet_hash"   in df.columns else "to_wallet"

        edges: list[TraceEdge] = []

        # ── 外部轉帳 (sub_kind == 0) ──────────────────────────────────────
        if "sub_kind" in df.columns:
            ext = df[df["sub_kind"] == 0].copy()
        else:
            ext = df.copy()

        # wallet → user（加值 kind=0）
        dep = ext[ext["kind"] == 0].dropna(subset=[from_col])
        if not dep.empty:
            srcs   = WALLET_PREFIX + dep[from_col].astype(str)
            dsts   = USER_PREFIX   + dep["user_id"].astype(int).astype(str)
            rates  = dep["twd_srate"].fillna(0) if "twd_srate" in dep.columns else pd.Series([0]*len(dep))
            amounts = dep["ori_samount"].fillna(0) * 1e-8 * rates * 1e-8
            edges += [
                TraceEdge(
                    src=s, dst=d,
                    timestamp=ts,
                    amount=float(amt),
                    edge_type="deposit",
                    currency=str(cur),
                )
                for s, d, ts, amt, cur in zip(
                    srcs, dsts,
                    dep["created_at"] if "created_at" in dep.columns else [None]*len(dep),
                    amounts,
                    dep.get("currency", pd.Series(["crypto"]*len(dep))),
                )
            ]

        # user → wallet（提領 kind=1）
        wit = ext[ext["kind"] == 1].dropna(subset=[to_col])
        if not wit.empty:
            srcs   = USER_PREFIX   + wit["user_id"].astype(int).astype(str)
            dsts   = WALLET_PREFIX + wit[to_col].astype(str)
            rates  = wit["twd_srate"].fillna(0) if "twd_srate" in wit.columns else pd.Series([0]*len(wit))
            amounts = wit["ori_samount"].fillna(0) * 1e-8 * rates * 1e-8
            edges += [
                TraceEdge(
                    src=s, dst=d,
                    timestamp=ts,
                    amount=float(amt),
                    edge_type="withdrawal",
                    currency=str(cur),
                )
                for s, d, ts, amt, cur in zip(
                    srcs, dsts,
                    wit["created_at"] if "created_at" in wit.columns else [None]*len(wit),
                    amounts,
                    wit.get("currency", pd.Series(["crypto"]*len(wit))),
                )
            ]

        # ── 內轉 (sub_kind == 1) ──────────────────────────────────────────
        if "sub_kind" in df.columns:
            intr = df[
                (df["sub_kind"] == 1) & df["relation_user_id"].notna()
            ].copy()
            intr["relation_user_id"] = intr["relation_user_id"].astype(int)
            if not intr.empty:
                srcs    = USER_PREFIX + intr["relation_user_id"].astype(str)
                dsts    = USER_PREFIX + intr["user_id"].astype(int).astype(str)
                rates   = intr["twd_srate"].fillna(0) if "twd_srate" in intr.columns else pd.Series([0]*len(intr))
                amounts = intr["ori_samount"].fillna(0) * 1e-8 * rates * 1e-8
                edges += [
                    TraceEdge(
                        src=s, dst=d,
                        timestamp=ts,
                        amount=float(amt),
                        edge_type="internal",
                        currency=str(cur),
                    )
                    for s, d, ts, amt, cur in zip(
                        srcs, dsts,
                        intr["created_at"] if "created_at" in intr.columns else [None]*len(intr),
                        amounts,
                        intr.get("currency", pd.Series(["crypto"]*len(intr))),
                    )
                ]

        return edges

    def _parse_twd(self, df: pd.DataFrame) -> list[TraceEdge]:
        if df is None or df.empty:
            return []

        df = df.copy()
        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

        edges: list[TraceEdge] = []

        # kind=0: 銀行→user（入金）
        dep = df[df["kind"] == 0]
        if not dep.empty:
            dsts    = USER_PREFIX + dep["user_id"].astype(int).astype(str)
            amounts = dep["ori_samount"].fillna(0) * 1e-8
            edges += [
                TraceEdge(
                    src=BANK_NODE, dst=d,
                    timestamp=ts,
                    amount=float(amt),
                    edge_type="twd_in",
                )
                for d, ts, amt in zip(
                    dsts,
                    dep["created_at"] if "created_at" in dep.columns else [None]*len(dep),
                    amounts,
                )
            ]

        # kind=1: user→銀行（出金）
        wit = df[df["kind"] == 1]
        if not wit.empty:
            srcs    = USER_PREFIX + wit["user_id"].astype(int).astype(str)
            amounts = wit["ori_samount"].fillna(0) * 1e-8
            edges += [
                TraceEdge(
                    src=s, dst=BANK_NODE,
                    timestamp=ts,
                    amount=float(amt),
                    edge_type="twd_out",
                )
                for s, ts, amt in zip(
                    srcs,
                    wit["created_at"] if "created_at" in wit.columns else [None]*len(wit),
                    amounts,
                )
            ]

        return edges

    # ── 追溯 ─────────────────────────────────────────────────────────────────

    def trace(
        self,
        fraud_node_ids: list,
        max_hops:   int   = 5,
        min_amount: float = 0.0,
    ) -> list[TraceResult]:
        """
        對每個詐騙節點做反向 BFS，找最上游源頭。

        Parameters
        ----------
        fraud_node_ids : list of int or str（不需加 'user_' 前綴）
        max_hops       : 最大追溯跳數（預設 5）
        min_amount     : 忽略金額低於此值的邊（過濾噪音）

        Returns
        -------
        list[TraceResult]，每個詐騙節點最多一筆結果。
        """
        results = []
        for nid in fraud_node_ids:
            key = (
                nid if isinstance(nid, str) and nid.startswith(USER_PREFIX)
                else f"{USER_PREFIX}{int(nid)}"
            )
            result = self._bfs_one(key, max_hops, min_amount)
            if result is not None:
                results.append(result)
        return results

    def _bfs_one(
        self,
        fraud_key:  str,
        max_hops:   int,
        min_amount: float,
    ) -> Optional[TraceResult]:
        """
        反向 BFS：從 fraud_key 往入邊方向走，找最佳源頭節點。
        佇列元素：(current_node, path_nodes, path_edges, earliest_ts, total_amount)
        """
        State = tuple[str, list[str], list[TraceEdge], Optional[pd.Timestamp], float]

        queue:   deque[State] = deque()
        visited: set[str]     = {fraud_key}
        queue.append((fraud_key, [fraud_key], [], None, 0.0))

        best:       Optional[TraceResult] = None
        best_score: float                 = -1.0

        while queue:
            node, path, edges_so_far, earliest, total_amt = queue.popleft()
            hops = len(path) - 1

            # 已達上限 or 該節點無入邊 → 此路徑的源頭
            no_predecessors = node not in self._reverse_adj
            if hops >= max_hops or no_predecessors:
                if hops > 0:
                    candidate = self._make_result(fraud_key, path, edges_so_far, earliest, total_amt)
                    score = self._score(candidate)
                    if score > best_score:
                        best_score, best = score, candidate
                continue

            # 過濾金額太小的入邊
            predecessors = [
                e for e in self._reverse_adj[node]
                if e.amount >= min_amount
            ]
            # 按時間排序（最早的優先探索）
            predecessors.sort(key=lambda e: e.timestamp or pd.Timestamp.max)

            branched = False
            for pred_edge in predecessors:
                pred = pred_edge.src
                if pred in visited:
                    continue
                visited.add(pred)
                branched = True

                new_earliest = _min_timestamp(earliest, pred_edge.timestamp)
                queue.append((
                    pred,
                    [pred] + path,
                    [pred_edge] + edges_so_far,
                    new_earliest,
                    total_amt + pred_edge.amount,
                ))

            # 所有前驅都已走過 → 此節點也是一個源頭
            if not branched and hops > 0:
                candidate = self._make_result(fraud_key, path, edges_so_far, earliest, total_amt)
                score = self._score(candidate)
                if score > best_score:
                    best_score, best = score, candidate

        return best

    def _make_result(
        self,
        fraud_key: str,
        path:      list[str],
        edges:     list[TraceEdge],
        earliest:  Optional[pd.Timestamp],
        total_amt: float,
    ) -> TraceResult:
        source = path[0]
        if source == BANK_NODE:
            stype = "bank"
        elif source.startswith(WALLET_PREFIX):
            stype = "wallet"
        else:
            stype = "user"

        source_risk = 0.0
        if stype == "user":
            uid = int(source.replace(USER_PREFIX, ""))
            if uid in self._risk_df.index and "risk_score" in self._risk_df.columns:
                source_risk = float(self._risk_df.loc[uid, "risk_score"])

        return TraceResult(
            fraud_node_id=fraud_key,
            source_node_id=source,
            source_node_type=stype,
            path_nodes=tuple(path),
            path_edges=tuple(edges),
            hop_count=len(path) - 1,
            earliest_tx_time=earliest,
            total_amount=total_amt,
            source_risk_score=source_risk,
        )

    def _score(self, r: TraceResult) -> float:
        """
        綜合評分（越高 = 越可疑的源頭）。
        hop_count × 0.3 + 金額 × 0.4 + source_risk_score × 0.3
        """
        hop_score = min(r.hop_count / 5.0, 1.0)
        amt_score = min(r.total_amount / 1_000_000, 1.0)
        return hop_score * 0.3 + amt_score * 0.4 + r.source_risk_score * 0.3

    # ── 輸出 ─────────────────────────────────────────────────────────────────

    def to_dataframe(self, results: list[TraceResult]) -> pd.DataFrame:
        """
        把 TraceResult 列表轉成扁平 DataFrame，適合存 CSV 及前端讀取。

        欄位：
            fraud_node_id       已知詐騙節點
            source_node_id      最上游源頭（user_X / wallet_ADDR）
            source_node_type    'user' | 'wallet' | 'bank'
            path_nodes          JSON list，[source, ..., fraud_node]
            path_edges          JSON list，每條邊的 {src,dst,ts,amount,type}
            hop_count           路徑跳數
            earliest_tx_time    路徑中最早的交易時間
            total_amount_twd    路徑金額加總
            source_risk_score   源頭節點的模型風險分數（0~1）
        """
        rows = []
        for r in results:
            edge_list = [
                {
                    "src":       e.src,
                    "dst":       e.dst,
                    "timestamp": str(e.timestamp) if e.timestamp is not None else None,
                    "amount":    round(e.amount, 2),
                    "type":      e.edge_type,
                    "currency":  e.currency,
                }
                for e in r.path_edges
            ]
            rows.append({
                "fraud_node_id":     r.fraud_node_id,
                "source_node_id":    r.source_node_id,
                "source_node_type":  r.source_node_type,
                "path_nodes":        json.dumps(list(r.path_nodes)),
                "path_edges":        json.dumps(edge_list),
                "hop_count":         r.hop_count,
                "earliest_tx_time":  r.earliest_tx_time,
                "total_amount_twd":  round(r.total_amount, 2),
                "source_risk_score": round(r.source_risk_score, 4),
            })
        return pd.DataFrame(rows)


# ── 工具函式 ─────────────────────────────────────────────────────────────────

def _min_timestamp(
    a: Optional[pd.Timestamp],
    b: Optional[pd.Timestamp],
) -> Optional[pd.Timestamp]:
    """回傳兩個 Timestamp 中較早的那個（None 視為最晚）。"""
    if a is None:
        return b
    if b is None or pd.isna(b):
        return a
    return a if a <= b else b
