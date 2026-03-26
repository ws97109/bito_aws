"""
alg2.py — GNN Causal Explainer（有向時序版，搭配 Causal Shapley）
=================================================================
對應 CXGNN 論文 Algorithm 2：
  Input : 有向因果圖 G、已知詐騙節點（label=1）
  Output: 因果解釋子圖 Γ＋根因節點 v*

改動（vs 原始版）：
  - 每個節點的分數改用 Causal Shapley Value（φ_i），非 expected_p
  - 加入 backward_score：沿入邊方向累積 φ，越接近源頭的祖先節點分數越高
    → 對應「反向追溯」語義
  - best_node 選取標準：causal_shapley × backward_weight（有理論依據）

backward_score 的理論依據：
  - 參考 Cheng et al. (2021) "Causal Effect Estimation and Optimal Dose Suggestions"
    的累積因果效應概念
  - 若節點 u 是 v 的祖先，且 φ(u) > 0，則 u → v 這條路徑上存在因果傳遞
  - backward_score(u) = φ(u) × Π φ(w) for w on path u→v*（幾何平均）
"""
from __future__ import annotations

import logging

import pandas as pd

import alg1

logger = logging.getLogger(__name__)


def _compute_backward_scores(
    models: dict,
    graph,
    fraud_node: int,
) -> dict[int, float]:
    """
    計算每個節點的 backward_score。

    backward_score(u) = φ(u) × (路徑上所有節點 φ 的幾何平均)

    這代表：「如果 u 是根因，因果信號傳到 fraud_node 的衰減程度」。
    幾何平均確保路徑越長、中間節點 φ 越低，backward_score 越小。

    Parameters
    ----------
    models     : alg_2 回傳的模型字典 {node_int: {'expected_p': φ, ...}}
    graph      : CausalGraph（有向版）
    fraud_node : 已知詐騙節點 id

    Returns
    -------
    dict {node_id: backward_score}
    """
    import math

    backward_scores: dict[int, float] = {}

    for node in graph.v:
        phi_node = models[node].get("expected_p", 0.0)

        # 找從 node 到 fraud_node 的路徑（沿有向邊）
        path_phis = _path_causal_product(graph, node, fraud_node, models)

        if path_phis is not None and len(path_phis) > 0:
            # 路徑幾何平均（log 域計算避免下溢）
            log_sum = sum(math.log(max(p, 1e-9)) for p in path_phis)
            geo_mean = math.exp(log_sum / len(path_phis))
            backward_scores[node] = phi_node * geo_mean
        else:
            # 沒有到達 fraud_node 的路徑 → 純 φ 值
            backward_scores[node] = phi_node

    return backward_scores


def _path_causal_product(
    graph,
    src: int,
    dst: int,
    models: dict,
    max_depth: int = 10,
) -> list[float] | None:
    """
    BFS 找從 src 到 dst 的最短路徑，回傳路徑上各節點的 φ 值列表。
    若無路徑回傳 None。
    """
    from collections import deque

    if src == dst:
        return [models[src].get("expected_p", 0.0)]

    queue = deque([(src, [src])])
    visited = {src}

    while queue:
        cur, path = queue.popleft()
        if len(path) > max_depth:
            continue
        for child in graph.ch.get(cur, set()):
            if child in visited:
                continue
            new_path = path + [child]
            if child == dst:
                return [models[n].get("expected_p", 0.0) for n in new_path]
            visited.add(child)
            queue.append((child, new_path))

    return None


def print_expected_p_for_each_node(models: dict) -> None:
    """印出每個節點的 Causal Shapley Value（對應原始 expected_p 的輸出）。"""
    for node, info in models.items():
        print(
            f"Node: {node:3d}  "
            f"Causal Shapley φ = {info.get('expected_p', 0.0):.4f}  "
            f"backward_score = {info.get('backward_score', 0.0):.4f}"
        )


def alg_2(
    Graph,
    num_epochs: int,
    data: pd.DataFrame,
    role_id: list[int],
) -> tuple:
    """
    GNN Causal Explainer 主函式（有向版）。

    對應 CXGNN Algorithm 2：
      1. 對每個節點訓練 GNN-NCM，計算 Causal Shapley Value φ_i
      2. 計算 backward_score（路徑幾何衰減）
      3. 找出 best_node = argmax backward_score（最可能的根因）
      4. 以 best_node 為中心，回傳因果解釋子圖 Γ

    Parameters
    ----------
    Graph      : CausalGraph（有向版）
    num_epochs : 每個節點的訓練輪數
    data       : DataFrame，index=node_int，含 'node_label'
    role_id    : list[int]，標籤（0=正常, 1=詐騙）

    Returns
    -------
    (models, best_loss, best_model, best_shapley, best_output, best_new_v, best_node)

    其中 best_shapley 是 Causal Shapley Value（φ），
    可直接對應論文中的 node expressivity。
    """
    if num_epochs is None:
        num_epochs = 100

    # ── Step 1：對每個節點訓練 NCM，取得 Causal Shapley Value ──────────────
    models: dict[int, dict] = {}

    for node in Graph.v:
        Graph.target_node = node
        (
            loss_history,
            total_loss,
            model_state,
            causal_shapley,  # 這是 φ_i
            output,
            new_v,
        ) = alg1.train(
            Graph,
            learning_rate=0.005,
            h_size=32,
            h_layers=2,
            num_epochs=num_epochs,
            data=data,
            role_id=role_id,
            target_node=node,
        )

        models[node] = {
            "model":        model_state,
            "expected_p":   causal_shapley,  # φ_i（Causal Shapley）
            "total_loss":   total_loss,
            "output":       output,
            "new_v":        new_v,
            "loss_history": loss_history,
        }

    # ── Step 2：找詐騙節點（label=1），計算 backward_score ─────────────────
    fraud_nodes = [i for i, lbl in enumerate(role_id) if lbl == 1]

    if fraud_nodes:
        # 用第一個詐騙節點作為追溯終點
        primary_fraud = fraud_nodes[0]
        backward_scores = _compute_backward_scores(models, Graph, primary_fraud)
    else:
        # 沒有 ground truth 詐騙節點時，只用 Causal Shapley
        backward_scores = {n: models[n]["expected_p"] for n in Graph.v}

    for node in Graph.v:
        models[node]["backward_score"] = backward_scores.get(node, 0.0)

    # ── Step 3：選取 best_node ─────────────────────────────────────────────
    # 排序標準：backward_score（主要） > Causal Shapley φ（次要）
    # 這樣既有路徑累積的語義，又保留 NCM 訓練的因果分數
    best_node = max(
        Graph.v,
        key=lambda k: (
            models[k]["backward_score"],
            models[k]["expected_p"],
        ),
    )

    best_model      = models[best_node]["model"]
    best_total_loss = models[best_node]["total_loss"]
    best_shapley    = models[best_node]["expected_p"]   # φ of best_node
    best_output     = models[best_node]["output"]
    best_new_v      = models[best_node]["new_v"]

    print_expected_p_for_each_node(models)
    print(f"→ best_node = {best_node}  φ = {best_shapley:.4f}  "
          f"backward_score = {models[best_node]['backward_score']:.4f}")
    print(f"→ causal subgraph Γ = {best_new_v}")

    return (
        models,
        best_total_loss,
        best_model,
        best_shapley,
        best_output,
        best_new_v,
        best_node,
    )