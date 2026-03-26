"""
Graph Neural Network Module
論文核心：用 GNN 捕捉錢包地址圖譜中的風險傳播
採用 GraphSAGE + 異質圖（用戶節點 + 錢包節點）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, BatchNorm
from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np
from typing import Dict, Tuple


# ─────────────────────────────────────────────
# 圖建構：從 crypto_transfer 建立用戶-錢包異質圖
# ─────────────────────────────────────────────

def build_transaction_graph(
    crypto: pd.DataFrame,
    user_features: pd.DataFrame,
) -> HeteroData:
    """
    節點類型：
      - user:   每個 user_id
      - wallet: 每個唯一鏈上錢包地址
    邊類型：
      - user   --[sends]-->    wallet  (提領)
      - wallet --[receives]--> user    (加值)
      - user   --[internal]--> user    (內轉, relation_user_id)
    """
    data = HeteroData()

    # ── 節點索引映射 ──
    all_users = user_features.index.tolist()
    user_to_idx = {uid: i for i, uid in enumerate(all_users)}

    external = crypto[crypto["sub_kind"] == 0].copy()
    wallets = pd.concat([
        external["from_wallet_hash"].dropna(),
        external["to_wallet_hash"].dropna(),
    ]).unique().tolist()
    wallet_to_idx = {w: i for i, w in enumerate(wallets)}

    # ── 用戶節點特徵 ──
    user_feat_np = user_features.values.astype(np.float32)
    data["user"].x = torch.tensor(user_feat_np)
    data["user"].num_nodes = len(all_users)

    # ── 錢包節點特徵（無特徵，用 degree 初始化）──
    data["wallet"].x = torch.zeros(len(wallets), 4, dtype=torch.float)
    data["wallet"].num_nodes = len(wallets)

    # ── 邊：user → wallet（提領）──
    withdrawals = external[external["kind"] == 1].dropna(subset=["to_wallet_hash"])
    src_u, dst_w = [], []
    for _, row in withdrawals.iterrows():
        uid = row["user_id"]
        wid = row["to_wallet_hash"]
        if uid in user_to_idx and wid in wallet_to_idx:
            src_u.append(user_to_idx[uid])
            dst_w.append(wallet_to_idx[wid])
    if src_u:
        data["user", "sends", "wallet"].edge_index = torch.tensor(
            [src_u, dst_w], dtype=torch.long
        )

    # ── 邊：wallet → user（加值）──
    deposits = external[external["kind"] == 0].dropna(subset=["from_wallet_hash"])
    src_w2, dst_u2 = [], []
    for _, row in deposits.iterrows():
        uid = row["user_id"]
        wid = row["from_wallet_hash"]
        if uid in user_to_idx and wid in wallet_to_idx:
            src_w2.append(wallet_to_idx[wid])
            dst_u2.append(user_to_idx[uid])
    if src_w2:
        data["wallet", "funds", "user"].edge_index = torch.tensor(
            [src_w2, dst_u2], dtype=torch.long
        )

    # ── 邊：user → user（內轉）──
    internal = crypto[
        (crypto["sub_kind"] == 1) &
        (crypto["relation_user_id"].notna())
    ].copy()
    src_uu, dst_uu = [], []
    for _, row in internal.iterrows():
        uid  = row["user_id"]
        peer = row["relation_user_id"]
        if uid in user_to_idx and peer in user_to_idx:
            src_uu.append(user_to_idx[uid])
            dst_uu.append(user_to_idx[int(peer)])
    if src_uu:
        data["user", "transfers", "user"].edge_index = torch.tensor(
            [src_uu, dst_uu], dtype=torch.long
        )

    data.user_to_idx    = user_to_idx
    data.wallet_to_idx  = wallet_to_idx

    return data


# ─────────────────────────────────────────────
# 模型定義：異質圖 SAGEConv
# ─────────────────────────────────────────────

class HeteroGNNEncoder(nn.Module):
    """
    論文架構：
      Layer 1: HeteroSAGE（收集鄰域資訊）
      Layer 2: GATConv（注意力加權）
      Layer 3: MLP 頭（輸出風險分數）
    """
    def __init__(self, user_in_dim: int, hidden_dim: int = 128, out_dim: int = 64):
        super().__init__()
        wallet_in_dim = 4

        # 第一層：異質 SAGE
        self.sage_u2w = SAGEConv((user_in_dim, wallet_in_dim), hidden_dim)
        self.sage_w2u = SAGEConv((wallet_in_dim, user_in_dim), hidden_dim)
        self.sage_u2u = SAGEConv(user_in_dim, hidden_dim)

        # BN
        self.bn_user   = BatchNorm(hidden_dim)
        self.bn_wallet = BatchNorm(hidden_dim)

        # 第二層：GAT（用戶同質圖）
        self.gat = GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=0.3)
        self.bn2 = BatchNorm(hidden_dim)

        # 投影頭
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        x_user   = data["user"].x
        x_wallet = data["wallet"].x

        # ── 收集各方向訊息 ──
        h_u_from_w = torch.zeros(x_user.size(0),   self.sage_w2u.out_channels, device=x_user.device)
        h_w_from_u = torch.zeros(x_wallet.size(0), self.sage_u2w.out_channels, device=x_user.device)
        h_u_from_u = torch.zeros_like(h_u_from_w)

        key_uw = ("user", "sends", "wallet")
        key_wu = ("wallet", "funds", "user")
        key_uu = ("user", "transfers", "user")

        if key_uw in data.edge_types:
            ei = data[key_uw].edge_index
            h_w_from_u = F.relu(self.sage_u2w((x_user, x_wallet), ei))

        if key_wu in data.edge_types:
            ei = data[key_wu].edge_index
            h_u_from_w = F.relu(self.sage_w2u((x_wallet, x_user), ei))

        if key_uu in data.edge_types:
            ei = data[key_uu].edge_index
            h_u_from_u = F.relu(self.sage_u2u(x_user, ei))

        # ── 融合用戶嵌入 ──
        h_user   = self.bn_user(h_u_from_w + h_u_from_u)
        h_wallet = self.bn_wallet(h_w_from_u)           # noqa: F841

        # ── GAT 精煉 ──
        if key_uu in data.edge_types:
            ei = data[key_uu].edge_index
            h_user = F.relu(self.gat(h_user, ei))
            h_user = self.bn2(h_user)

        return self.proj(h_user)


class BlacklistGNN(nn.Module):
    """完整分類模型：GNN 嵌入 + 表格特徵 + 分類頭"""

    def __init__(self, user_in_dim: int, tabular_dim: int, hidden: int = 128):
        super().__init__()
        self.gnn_encoder = HeteroGNNEncoder(user_in_dim, hidden, out_dim=64)

        self.classifier = nn.Sequential(
            nn.Linear(64 + tabular_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, data: HeteroData, tabular: torch.Tensor) -> torch.Tensor:
        gnn_emb = self.gnn_encoder(data)                    # [N_user, 64]
        combined = torch.cat([gnn_emb, tabular], dim=-1)    # [N_user, 64+T]
        return self.classifier(combined).squeeze(-1)        # [N_user]

    def predict_proba(self, data: HeteroData, tabular: torch.Tensor) -> torch.Tensor:
        logits = self.forward(data, tabular)
        return torch.sigmoid(logits)