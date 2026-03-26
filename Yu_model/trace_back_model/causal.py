"""
causal.py — Directed Temporal Causal Graph
============================================
在原始 CXGNN 的 CausalGraph 基礎上，加入：
  1. 有向邊（Pa/Ch 分開）：符合 Pearl SCM 父節點定義
  2. 時間戳記錄（可選）：支援 do(v, t) 只影響 t 之後節點
  3. Topological ordering：確保因果方向一致

理論依據：
  Pearl (2009) Causality, Ch.1 — SCM 的有向無環圖（DAG）結構
  父節點集合 Pa(Vi) ⊆ V \ {Vi} 定義因果方向
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import networkx as nx
import numpy as np


class CausalGraph:
    """
    有向因果圖（Directed Acyclic Causal Graph）。

    邊語義：(u → v) 代表「u 是 v 的直接原因（parent）」，
    對應 Pearl SCM 中的 v ← f_v(Pa(v), U_v)。

    Parameters
    ----------
    V   : iterable，節點集合（任意 hashable，通常是 int）
    path: list of (src, dst) tuple，**有向邊**，src → dst
    unobserved_edges: list of (u, v)，雙向隱變數相關邊（保留原始介面）
    timestamps: dict {(src, dst): timestamp}，邊的時間戳（可選）
    """

    def __init__(
        self,
        V,
        path=None,
        unobserved_edges=None,
        timestamps: Optional[dict] = None,
    ):
        if path is None:
            path = []
        if unobserved_edges is None:
            unobserved_edges = []

        self.v = list(V)
        self.set_v = set(self.v)

        # ── 有向鄰接結構 ──────────────────────────────────────────────────────
        # pa[v]  = 父節點集（直接原因）
        # ch[v]  = 子節點集（直接結果）
        # fn[v]  = 無向一階鄰居（保留給 alg1 相容）
        self.pa: dict[int, set] = {node: set() for node in self.v}
        self.ch: dict[int, set] = {node: set() for node in self.v}
        self.fn: dict[int, set] = {node: set() for node in self.v}
        self.sn: dict[int, set] = {node: set() for node in self.v}
        self.on: dict[int, set] = {node: set() for node in self.v}

        # 有向邊集合（不排序，保留方向性）
        self.directed_edges: set[tuple] = set()
        # 時間戳 {(src, dst): timestamp}
        self.timestamps: dict[tuple, any] = timestamps or {}

        # 為了與原始 alg1/alg2 相容，保留 self.p（無向版本）
        self.p: set[tuple] = set()

        self.ue = set(map(tuple, map(sorted, unobserved_edges)))

        for v1, v2 in path:
            self._add_directed_edge(v1, v2)

        # target_node / neighbor caches（由 categorize_neighbors 設定）
        self.target_node = None
        self.one_hop_neighbors: set = set()
        self.two_hop_neighbors: set = set()
        self.out_of_neighborhood: set = set()

    def _add_directed_edge(self, src, dst) -> None:
        """加入有向邊 src → dst，同步更新所有鄰接結構。"""
        if src not in self.set_v or dst not in self.set_v:
            return
        self.directed_edges.add((src, dst))
        self.pa[dst].add(src)   # src 是 dst 的父節點
        self.ch[src].add(dst)   # dst 是 src 的子節點
        self.fn[src].add(dst)   # 無向鄰居（相容）
        self.fn[dst].add(src)
        # 無向版本（相容 alg1）
        self.p.add(tuple(sorted((src, dst))))

    # ── 父節點 / 子節點 查詢 ──────────────────────────────────────────────────

    def parents(self, node) -> set:
        """回傳 node 的直接父節點（因果上游）。"""
        return self.pa.get(node, set())

    def children(self, node) -> set:
        """回傳 node 的直接子節點（因果下游）。"""
        return self.ch.get(node, set())

    def ancestors(self, node) -> set:
        """回傳 node 的所有祖先（遞迴父節點）。"""
        visited = set()
        queue = deque(self.parents(node))
        while queue:
            cur = queue.popleft()
            if cur not in visited:
                visited.add(cur)
                queue.extend(self.parents(cur))
        return visited

    def descendants(self, node) -> set:
        """回傳 node 的所有後代（遞迴子節點）。"""
        visited = set()
        queue = deque(self.children(node))
        while queue:
            cur = queue.popleft()
            if cur not in visited:
                visited.add(cur)
                queue.extend(self.children(cur))
        return visited

    # ── 鄰域分類（與 alg1 相容，改為以父節點優先）─────────────────────────────

    def categorize_neighbors(self, target_node):
        """
        以 target_node 為中心分類鄰域。

        改動（vs 原始版）：
          one_hop_neighbors 優先包含「父節點」（因果上游），
          再補「子節點」，確保反向追溯時上游節點得到較高權重。
        """
        if target_node not in self.set_v:
            return target_node, set(), set(), set()

        # 一階鄰居：父節點（上游）+ 子節點（下游）
        one_hop = self.pa[target_node] | self.ch[target_node]

        # 二階鄰居
        two_hop = set()
        for n in one_hop:
            two_hop |= self.pa[n] | self.ch[n]
        two_hop -= one_hop
        two_hop.discard(target_node)

        out_of_neighborhood = (
            self.set_v - one_hop - two_hop - {target_node}
        )

        self.target_node          = target_node
        self.one_hop_neighbors    = one_hop
        self.two_hop_neighbors    = two_hop
        self.out_of_neighborhood  = out_of_neighborhood
        self.sn[target_node]      = two_hop
        self.on[target_node]      = out_of_neighborhood

        return target_node, one_hop, two_hop, out_of_neighborhood

    # ── 拓撲排序（Kahn's algorithm）──────────────────────────────────────────

    def topological_order(self) -> list:
        """
        回傳節點的拓撲排序（上游 → 下游）。
        若圖中有環（資料層面不應出現），回傳可排序的部分。
        """
        in_degree = {v: len(self.pa[v]) for v in self.v}
        queue = deque([v for v in self.v if in_degree[v] == 0])
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for child in self.ch[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        return order

    # ── 時間過濾（Temporal do-calculus 支援）─────────────────────────────────

    def parents_before(self, node, cutoff_time) -> set:
        """
        回傳在 cutoff_time 之前就已存在邊的父節點。
        用於 do(v, t)：只有 t' < t 的邊才構成因果前驅。
        """
        if not self.timestamps:
            return self.parents(node)
        result = set()
        for pa in self.parents(node):
            ts = self.timestamps.get((pa, node))
            if ts is None or ts <= cutoff_time:
                result.add(pa)
        return result

    # ── 舊介面相容（graph_search）─────────────────────────────────────────────

    def graph_search(self, cg, v1, v2=None, edge_type="path", target_node=None):
        """保留原始介面，內部改用有向鄰接結構。"""
        if target_node is not None:
            self.categorize_neighbors(target_node)

        q = deque([v1])
        seen = {v1}
        while q:
            cur = q.popleft()
            if edge_type == "path":
                neighbors = self.fn[cur]
            else:
                neighbors = self.sn.get(target_node, set()) | self.on.get(target_node, set())

            for nb in neighbors:
                if nb not in seen:
                    if v2 is not None and nb == v2:
                        return True
                    seen.add(nb)
                    q.append(nb)

        return seen if v2 is None else False

    def degrees(self) -> dict:
        return {node: len(self.fn[node]) for node in self.v}

    def __iter__(self):
        return iter(self.v)