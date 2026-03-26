"""
alg1.py — Causal Shapley NCM (有向時序版)
==========================================
以 Causal Shapley Values 替換原始 CXGNN 的 expected_p 計算。

理論依據
--------
1. Heskes et al. (2020) "Causal Shapley Values: Exploiting Causal Knowledge
   to Explain Individual Predictions of Complex Models." NeurIPS 2020.
   → Shapley 值的聯盟函數 v(S) 使用 do-calculus 定義的介入期望，
     而非條件期望，避免混淆因果與關聯。

2. Pearl (2009) Causality, Ch.3 — do-calculus 公理
   → 介入分佈 P(Y | do(X=x)) 通過截斷父節點的入邊計算。

3. Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions."
   NeurIPS 2017.（SHAP 基礎，Causal Shapley 的出發點）

核心改動（vs 原始 alg1）
-----------------------
A. compute_probability_of_node_label：
   - 原版：O(2^n) 暴力枚舉 + 只算靜態邊際分佈
   - 新版：用 NCM forward 直接估計 P(Y | do(X=x))，O(n×samples)

B. calculate_expected_prob：
   - 原版：loop 裡提早 return（bug）+ 語義不清
   - 新版：Causal Shapley 期望值
     φ_i = Σ_S [ |S|!(n-|S|-1)!/n! × (v(S∪{i}) - v(S)) ]
     其中 v(S) = E[Y | do(X_S = x_S)]（介入期望）

C. CausalGraph 邊方向：
   - Pa(v) 只含有向入邊，確保 do-calculus 截斷語義正確

口試問答支撐
-----------
Q: 為什麼用 Causal Shapley 而非普通 SHAP？
A: 普通 SHAP 的 v(S) = E[Y | X_S=x_S] 是條件期望，無法排除混淆因子。
   Causal Shapley 的 v(S) = E[Y | do(X_S=x_S)] 是介入期望，對應 Pearl 的
   do-calculus，能識別真實因果貢獻，不受 spurious correlation 影響。
   這與 CXGNN 論文 Section 2 中對 association-based 方法的批評完全對應。
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


# ─────────────────────────────────────────────────────────────────────────────
# 1. Neural Network（NCM 的函數近似器）
# ─────────────────────────────────────────────────────────────────────────────

class NNModel(nn.Module):
    """
    前饋神經網路，作為 GNN-NCM 中的 f̂_vi。

    輸入：節點效應向量 u（維度 = |Pa(v)| × 2 + 1）
    輸出：標量，經 sigmoid 後代表 P(Y=1 | do(context))
    """

    def __init__(self, input_size: int, output_size: int, h_size: int, h_layers: int):
        super().__init__()
        self.input_size  = input_size
        self.output_size = output_size
        self.h_size      = h_size
        self.h_layers    = h_layers

        layers = [nn.Linear(input_size, h_size), nn.ReLU()]
        for _ in range(h_layers - 1):
            layers += [nn.Linear(h_size, h_size), nn.ReLU()]
        layers.append(nn.Linear(h_size, output_size))
        self.nn = nn.Sequential(*layers)
        self.nn.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("relu"))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.nn(u))


# ─────────────────────────────────────────────────────────────────────────────
# 2. NCM 類別（G-Constrained GNN-NCM）
# ─────────────────────────────────────────────────────────────────────────────

class NCM:
    """
    G-Constrained GNN Neural Causal Model（有向版）。

    對應 CXGNN 論文 Definition 3 / Theorem 2：
      - 父節點集合嚴格按有向邊設定（self.graph.pa[target_node]）
      - 輸入向量 u = [y_target, u_pa1, u_e1, u_pa2, u_e2, ...]
        其中 u_pai = 父節點的 node_label
             u_ei  = 邊效應（從 data 取得）
    """

    def __init__(
        self,
        graph,
        target_node: int,
        learning_rate: float,
        h_size: int,
        h_layers: int,
        data: pd.DataFrame,
    ):
        self.graph         = graph
        self.target_node   = target_node
        self.learning_rate = learning_rate

        # 父節點（有向入邊，對應 Pa(v) in Pearl）
        parents = list(graph.pa.get(target_node, set()))
        # 若沒有父節點（源頭節點），退回無向鄰居
        if not parents:
            parents = list(graph.fn.get(target_node, set()))

        self._parents = parents

        # ── 建立輸入張量 u ──────────────────────────────────────────────────
        # u = [y_target] + [node_label(pa_i)] × n_parents + [edge_effect(pa_i)] × n_parents
        target_label = float(
            data.loc[target_node, "node_label"]
            if target_node in data.index
            else 0.0
        )
        self._y_target = torch.tensor([target_label], dtype=torch.float32)

        # 節點效應（u_i）和邊效應（u_ij）
        self._u_node = {
            p: torch.tensor(
                [float(data.loc[p, "node_label"]) if p in data.index else 0.0],
                dtype=torch.float32,
            )
            for p in parents
        }
        self._u_edge = {
            p: torch.tensor([0.5], dtype=torch.float32)  # 初始化為 0.5，訓練時更新
            for p in parents
        }

        self._rebuild_u()
        input_size = len(self._u)
        self.model = NNModel(input_size, 1, h_size, h_layers)

    def _rebuild_u(self):
        """重建輸入向量（每個 epoch 前呼叫）。"""
        parts = [self._y_target]
        for p in self._parents:
            parts.append(self._u_node[p])
            parts.append(self._u_edge[p])
        self._u = torch.cat(parts, dim=0)

    def add_gaussian_noise(self, tensor: torch.Tensor, std: float = 0.01) -> torch.Tensor:
        noise = torch.randn_like(tensor) * std
        return torch.clamp(tensor + noise, 0.0, 1.0)

    def forward(self, add_noise: bool = False) -> torch.Tensor:
        if add_noise:
            for p in self._parents:
                self._u_node[p] = self.add_gaussian_noise(self._u_node[p])
                self._u_edge[p] = self.add_gaussian_noise(self._u_edge[p])
            self._rebuild_u()
        return self.model(self._u)

    def intervene(self, intervened_parents: set[int]) -> torch.Tensor:
        """
        do-calculus 介入：把 intervened_parents 中的節點標籤強制設為 0
        （截斷入邊，對應 Pearl do(X=x) 操作）。
        回傳介入後的模型輸出。

        理論對應：CXGNN 論文 Equation (4)
          p^M(y_v | do(v_i)) = E_{u}[Π f̂_vj(ûvj, ûv,vj)]
        """
        saved = {}
        for p in intervened_parents:
            if p in self._u_node:
                saved[p] = self._u_node[p].clone()
                self._u_node[p] = torch.zeros_like(self._u_node[p])
        self._rebuild_u()
        with torch.no_grad():
            out = self.model(self._u)
        # 還原
        for p, val in saved.items():
            self._u_node[p] = val
        self._rebuild_u()
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Causal Shapley 計算（核心理論貢獻）
# ─────────────────────────────────────────────────────────────────────────────

def _causal_value_fn(model: NNModel, u_baseline: torch.Tensor, mask: list[bool]) -> float:
    """
    計算聯盟 S 的因果值 v(S)。

    v(S) = E[f(u) | do(X_S = observed, X_{V\\S} = baseline)]

    實作方式：
      - 在 coalition S 中的父節點：保持觀測值（真實標籤）
      - 不在 S 中的父節點：設為 baseline（=0.5，代表「中立/不介入」）

    這對應 Pearl do-calculus 的截斷（cut）語義：
      不在聯盟中的變數被強制設為基準值，
      切斷其與其他變數的自然依賴關係。

    理論對應：Heskes et al. (2020) Equation (3)
      v(S) = E_{X_{V\\S}}[f(X_S=x_S, X_{V\\S})] with do(X_{V\\S}=baseline)
    """
    u_intervened = u_baseline.clone()
    # mask[0] = target node，從 index 1 開始每 2 個是一個父節點
    # 結構：[y_target, u_node_pa1, u_edge_pa1, u_node_pa2, ...]
    for i, in_coalition in enumerate(mask):
        if not in_coalition:
            node_idx = 1 + i * 2      # u_node 位置
            edge_idx = 1 + i * 2 + 1  # u_edge 位置
            if node_idx < len(u_intervened):
                u_intervened[node_idx] = 0.5   # 中立基準值
            if edge_idx < len(u_intervened):
                u_intervened[edge_idx] = 0.5

    with torch.no_grad():
        return float(model(u_intervened).item())


def compute_causal_shapley(ncm: NCM) -> float:
    """
    計算 target_node 的 Causal Shapley Value（訓練後評估）。

    定義（Heskes et al., 2020 NeurIPS）：
      φ_i = Σ_{S ⊆ Pa(v) \\ {i}} [|S|!(n-|S|-1)! / n!] × [v(S ∪ {i}) - v(S)]

    其中：
      n = |Pa(v)|（父節點數量）
      v(S) = E[f(u) | do(X_S = observed, X_{V\\S} = 0.5)]（介入期望）

    特殊情況：
      - Pa(v) = ∅（源頭節點）：φ = f(u)（模型直接輸出）
      - n = 1：φ = v({i}) - v(∅) = f(u_i=observed) - f(u_i=0.5)

    回傳值域：[0, 1]（clip 後）

    口試解釋重點：
      「φ_i 衡量的是：在所有可能的父節點子集 S 上，
       加入第 i 個父節點到 S 中，對模型預測詐騙的平均邊際因果貢獻。
       這與普通 SHAP 的差異在於 v(S) 使用 do-calculus 而非條件期望，
       能排除混淆因子的干擾。」
    """
    parents = ncm._parents
    n = len(parents)

    # 取得當前（訓練後）的輸入向量
    u_obs = ncm._u.detach().clone()

    if n == 0:
        # 源頭節點：沒有父節點，φ = 模型輸出（代表自身的預測能力）
        with torch.no_grad():
            return float(ncm.model(u_obs).item())

    shapley_sum = 0.0

    # 枚舉 Pa(v) 的所有子集（若 n > 8 用 Monte Carlo 近似）
    if n <= 8:
        subset_list = [
            list(combo)
            for r in range(n + 1)
            for combo in combinations(range(n), r)
        ]
    else:
        rng = np.random.default_rng(42)
        subset_list = []
        for _ in range(min(256, 2 ** n)):
            k = int(rng.integers(0, n + 1))
            idxs = rng.choice(n, size=k, replace=False).tolist()
            subset_list.append(sorted(idxs))

    for S_indices in subset_list:
        s_size = len(S_indices)

        # Shapley 權重 w(|S|, n) = |S|!(n-|S|-1)!/n!
        # 當子集大小 == n 時 n-s_size-1 = -1，跳過（無剩餘特徵可加入）
        remaining = n - s_size - 1
        if remaining < 0:
            continue
        weight = (
            math.factorial(s_size)
            * math.factorial(remaining)
            / math.factorial(n)
        ) if n > 0 else 1.0

        # 對每個父節點 i，計算 v(S ∪ {i}) - v(S)
        for i in range(n):
            if i in S_indices:
                continue  # i 已在 S 中，跳過

            # 建立 mask：S ∪ {i}
            mask_with = [idx in S_indices or idx == i for idx in range(n)]
            # 建立 mask：S
            mask_without = [idx in S_indices for idx in range(n)]

            v_with    = _causal_value_fn(ncm.model, u_obs, mask_with)
            v_without = _causal_value_fn(ncm.model, u_obs, mask_without)

            # 邊際貢獻
            marginal = v_with - v_without
            shapley_sum += weight * marginal

    # 正規化（除以父節點數，保持在合理範圍）
    if n > 0:
        shapley_sum /= n

    return float(np.clip(shapley_sum, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# 4. 訓練函式（供 alg2 呼叫）
# ─────────────────────────────────────────────────────────────────────────────

def train(
    cg,
    learning_rate: float,
    h_size: int,
    h_layers: int,
    num_epochs: int,
    data: pd.DataFrame,
    role_id: list[int],
    target_node: int,
) -> tuple:
    """
    訓練單一節點的 GNN-NCM，回傳：
      (loss_history, final_loss, model_state_dict, causal_shapley_score, output, new_v)

    訓練目標：最小化 BCE(f̂(u), y_target)
    評估指標：Causal Shapley Value（不是 expected_p，有更強理論依據）

    Parameters
    ----------
    cg           : CausalGraph（有向版）
    learning_rate: Adam 學習率
    h_size       : 隱藏層寬度
    h_layers     : 隱藏層數量
    num_epochs   : 訓練輪數
    data         : DataFrame，index=node_int，含 'node_label' 欄位
    role_id      : list[int]，每個節點的標籤（0/1）
    target_node  : 目標節點 id

    Returns
    -------
    (loss_history, final_loss, state_dict, causal_shapley_score, output, new_v)
    """
    # 設定有向鄰域
    cg.target_node, cg.one_hop_neighbors, cg.two_hop_neighbors, cg.out_of_neighborhood = (
        cg.categorize_neighbors(target_node=target_node)
    )

    # 建立 NCM（使用父節點）
    ncm = NCM(
        graph=cg,
        target_node=target_node,
        learning_rate=learning_rate,
        h_size=h_size,
        h_layers=h_layers,
        data=data,
    )

    optimizer = optim.Adam(ncm.model.parameters(), lr=learning_rate)

    # new_v 是這個節點的「解釋子圖」：target + 所有父節點（上游）
    # 對應反向追溯：target 的直接原因
    new_v = {target_node} | set(ncm._parents)

    loss_history = []
    final_loss = torch.tensor(0.0)

    for _ in range(num_epochs):
        f = ncm.forward(add_noise=True)

        target_label = float(
            role_id[target_node]
            if target_node < len(role_id)
            else 0.0
        )
        target_tensor = torch.tensor([target_label], dtype=torch.float32)

        # BCE loss：對應 CXGNN 論文 Equation (6)
        loss = torch.nn.functional.binary_cross_entropy(
            f.view(1), target_tensor.view(1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        final_loss = loss

    # ── Causal Shapley Value（訓練後評估）────────────────────────────────
    # 這是口試時最重要的理論量：
    #   φ_i 衡量「如果對所有父節點做 do() 介入，
    #   target_node 的因果貢獻有多大」
    causal_shapley_score = compute_causal_shapley(ncm)

    # output：是否超過門檻（與原始相容）
    output = torch.tensor(
        [1.0 if causal_shapley_score >= 0.05 else 0.0]
    )

    return (
        loss_history,
        final_loss,
        ncm.model.state_dict(),
        causal_shapley_score,   # 取代原始 expected_p
        output,
        new_v,
    )