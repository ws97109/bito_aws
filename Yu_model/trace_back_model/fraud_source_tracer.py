"""
Fraud Source Tracer — Method B
時序反向追溯：從已知詐騙節點沿入邊方向找資金源頭。

演算法：
  1. 把 crypto_transfer / twd_transfer 建成有向圖（A→B 代表 A 把錢打給 B）
  2. 對每個詐騙節點做反向 BFS（沿入邊往上游走）
  3. 找到「最上游源頭」：無更早入邊的節點，或是另一個已知詐騙節點

用法：
    from fraud_source_tracer import FraudSourceTracer

    tracer = FraudSourceTracer(crypto_df, risk_df)
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
    source_node_type: str          # 'user' | 'wallet'
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
    gnn_edge_df : DataFrame, optional
        Wei_model 輸出的 gnn_edge_list.csv（需含 source, target, edge_type）。
        補充 crypto/twd 未涵蓋的圖結構邊，擴充反向追溯的可達路徑。
    """

    def __init__(
        self,
        crypto_df:   pd.DataFrame,
        risk_df:     pd.DataFrame,
        gnn_edge_df: pd.DataFrame | None = None,
        twd_df:      pd.DataFrame | None = None,
    ) -> None:
        self._risk_df      = risk_df
        self._reverse_adj: dict[str, list[TraceEdge]] = {}
        self._tx_lookup:   dict[tuple[str, str], tuple[float, pd.Timestamp | None]] = {}

        # 先建金額查詢表（供 gnn_edges 補充金額用）
        self._build_tx_lookup(crypto_df, twd_df or pd.DataFrame())
        # 再建反向圖
        self._build_reverse_graph(crypto_df, gnn_edge_df)

    # ── 建圖 ─────────────────────────────────────────────────────────────────

    def _build_reverse_graph(
        self,
        crypto_df:   pd.DataFrame,
        gnn_edge_df: pd.DataFrame | None,
    ) -> None:
        all_edges: list[TraceEdge] = (
            self._parse_crypto(crypto_df) +
            self._parse_twd() +
            self._parse_gnn_edges(gnn_edge_df)
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

    def _parse_twd(self) -> list[TraceEdge]:
        # twd_transfer 的所有記錄都是 user 與外部銀行帳戶之間的進出金，
        # 原始資料未定義對手節點，無法形成有效的圖邊，略過。
        return []

    def _build_tx_lookup(
        self,
        crypto_df: pd.DataFrame,
        twd_df:    pd.DataFrame,
    ) -> None:
        """
        從 crypto_transfer 建立 (src, dst) → (amount_twd, timestamp) 查詢表。
        供 _parse_gnn_edges 補充金額與時間資訊。
        """
        lookup: dict[tuple[str, str], tuple[float, pd.Timestamp | None]] = {}

        if crypto_df is not None and not crypto_df.empty:
            df = crypto_df.copy()
            if "created_at" in df.columns:
                df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

            from_col = "from_wallet_hash" if "from_wallet_hash" in df.columns else "from_wallet"
            to_col   = "to_wallet_hash"   if "to_wallet_hash"   in df.columns else "to_wallet"
            rates = df["twd_srate"].fillna(0) if "twd_srate" in df.columns else 0
            amounts = df["ori_samount"].fillna(0) * 1e-8 * rates * 1e-8

            # deposit: wallet → user
            dep = df[(df.get("sub_kind", 0) == 0) & (df["kind"] == 0)].copy() if "kind" in df.columns else pd.DataFrame()
            if not dep.empty and from_col in dep.columns:
                for _, row in dep.iterrows():
                    src = f"{WALLET_PREFIX}{row[from_col]}"
                    dst = f"{USER_PREFIX}{int(row['user_id'])}"
                    amt = float(row.get("ori_samount", 0) or 0) * 1e-8 * float(row.get("twd_srate", 0) or 0) * 1e-8
                    ts  = row.get("created_at")
                    lookup[(src, dst)] = (amt, ts if pd.notna(ts) else None)

            # internal: user → user
            if "sub_kind" in df.columns:
                intr = df[(df["sub_kind"] == 1) & df["relation_user_id"].notna()]
                for _, row in intr.iterrows():
                    src = f"{USER_PREFIX}{int(row['relation_user_id'])}"
                    dst = f"{USER_PREFIX}{int(row['user_id'])}"
                    amt = float(row.get("ori_samount", 0) or 0) * 1e-8 * float(row.get("twd_srate", 0) or 0) * 1e-8
                    ts  = row.get("created_at")
                    lookup[(src, dst)] = (amt, ts if pd.notna(ts) else None)

        self._tx_lookup = lookup
        logger.info("TX lookup built: %d entries", len(lookup))

    def _parse_gnn_edges(self, df: pd.DataFrame | None) -> list[TraceEdge]:
        """
        解析 gnn_edge_list.csv，只納入「資金流入」方向的邊：
          ✓ wallet_funds_user   (wallet → user，入金來源，加入反向圖)
          ✓ user_transfers_user (user → user，轉帳，加入反向圖)
          ✗ user_sends_wallet   (user → wallet，出金/提領，【排除】)

        排除 user_sends_wallet 的原因：
          反向追溯是找「誰把錢打進詐騙帳戶」，
          user_sends_wallet 是詐騙者把錢提出去，
          不是資金來源，加進去只會造成「詐騙者 → wallet → 再往上追」
          的錯誤路徑，讓追溯方向反過來。

        金額補充：GNN edge_list 沒有金額，從 _tx_lookup 補入。
        重複邊去重：同一 (src, dst) 只保留金額最大的一筆。
        """
        if df is None or df.empty:
            return []

        # 只保留資金流入方向的邊
        INCOMING_TYPES = {"wallet_funds_user", "user_transfers_user"}
        filtered = df[df["edge_type"].isin(INCOMING_TYPES)].copy()

        if filtered.empty:
            logger.warning("GNN edge_list 中找不到 wallet_funds_user 或 user_transfers_user 邊，請確認欄位名稱")
            return []

        # 去重：同一 (src, dst) 只保留一筆（優先保留有金額的）
        seen: dict[tuple[str, str], TraceEdge] = {}

        for row in filtered.itertuples(index=False):
            src = str(row.source)
            dst = str(row.target)
            edge_type_val = str(row.edge_type)

            # 從 tx_lookup 補金額和時間
            amt, ts = self._tx_lookup.get((src, dst), (0.0, None))

            key = (src, dst)
            if key not in seen or amt > seen[key].amount:
                seen[key] = TraceEdge(
                    src=src,
                    dst=dst,
                    timestamp=ts,
                    amount=amt,
                    edge_type=edge_type_val,
                )

        edges = list(seen.values())
        logger.info(
            "GNN edges parsed: %d (wallet_funds_user=%d, user_transfers_user=%d, 去重後=%d)",
            len(filtered),
            len(filtered[filtered["edge_type"] == "wallet_funds_user"]),
            len(filtered[filtered["edge_type"] == "user_transfers_user"]),
            len(edges),
        )
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
        if source.startswith(WALLET_PREFIX):
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

        評分邏輯：
          hop_score        × 0.4  跳數越多代表追溯越深
          amt_score        × 0.3  金額越大越可疑
          source_risk_score × 0.3  源頭本身的詐騙風險分數

        注意：GNN edge_list 邊的 amount=0（無金額資訊），
        為避免金額為 0 的路徑永遠得低分，
        當路徑所有邊 amount 都是 0 時，amt_score 改用 hop_score 代替。
        """
        hop_score = min(r.hop_count / 5.0, 1.0)
        if r.total_amount > 0:
            amt_score = min(r.total_amount / 1_000_000, 1.0)
        else:
            # GNN 邊無金額：用跳數替代，確保深層路徑仍能被選到
            amt_score = hop_score
        return hop_score * 0.4 + amt_score * 0.3 + r.source_risk_score * 0.3

    # ── 輸出 ─────────────────────────────────────────────────────────────────

    def to_dataframe(self, results: list[TraceResult]) -> pd.DataFrame:
        """
        把 TraceResult 列表轉成扁平 DataFrame，適合存 CSV 及前端讀取。

        欄位：
            fraud_node_id       已知詐騙節點
            source_node_id      最上游源頭（user_X / wallet_ADDR）
            source_node_type    'user' | 'wallet'
            path_nodes          JSON list，[source, ..., fraud_node]
            path_edges          JSON list，每條邊的 {src,dst,ts,amount,type}
            hop_count           路徑跳數
            earliest_tx_time    路徑中最早的交易時間
            total_amount_twd    路徑金額加總
            source_risk_score   源頭節點的模型風險分數（0~1）
            causal_confidence   啟發式因果信心分數（無 CXGNN 時的替代值）

        causal_confidence（啟發式版本）計算邏輯：
            = hop_score × 0.4 + amt_score × 0.3 + risk_score × 0.3
            等同 _score() 的邏輯，正規化到 [0, 1]。
            當 CXGNN 可執行時，此欄會被 cxgnn_adapter 的 Causal Shapley 覆蓋。
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

            # 啟發式 causal_confidence（CXGNN 不可用時的替代）
            hop_score = min(r.hop_count / 5.0, 1.0)
            amt_score = min(r.total_amount / 1_000_000, 1.0) if r.total_amount > 0 else hop_score
            heuristic_confidence = round(
                hop_score * 0.4 + amt_score * 0.3 + r.source_risk_score * 0.3,
                4,
            )

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
                "causal_confidence":  heuristic_confidence,
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