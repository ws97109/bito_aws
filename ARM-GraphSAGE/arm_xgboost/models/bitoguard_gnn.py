"""
BitoGuard GNN Model — GraphSAGE-ARM-CF Encoder
Used as an embedding extractor; final classification is handled by XGBoost.

Architecture:
  1. Input projection per node type
  2. Two-layer HeteroConv with GATv2Conv + ARM edge weights
  3. CF-gated readout (optional)
  4. Output: user node embeddings [N, out_dim]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear
from typing import Dict, Tuple, Optional


class BitoGuardGNN(nn.Module):
    """
    ARM-weighted heterogeneous GNN encoder.

    Returns user node embeddings of shape [num_users, out_dim].
    These embeddings are fed directly to XGBoost — no classification head here.
    """

    def __init__(
        self,
        node_feature_dims: Dict[str, int],
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.3,
        use_arm_weights: bool = True,
        use_cf_gating: bool = True,
    ):
        super().__init__()

        self.hidden_dim     = hidden_dim
        self.out_dim        = out_dim
        self.use_arm_weights = use_arm_weights
        self.use_cf_gating  = use_cf_gating

        edge_dim = 1 if use_arm_weights else None

        # Per-node-type input projections
        self.input_projs = nn.ModuleDict({
            ntype: Linear(in_dim, hidden_dim)
            for ntype, in_dim in node_feature_dims.items()
        })

        # Layer 1: hidden_dim → hidden_dim × num_heads
        self.conv1 = HeteroConv({
            ('user',   'sends_to',       'wallet'): GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout, edge_dim=edge_dim, add_self_loops=False),
            ('wallet', 'receives_from',  'user'):   GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout, edge_dim=edge_dim, add_self_loops=False),
            ('user',   'transacts_with', 'user'):   GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout, edge_dim=edge_dim, add_self_loops=False),
        }, aggr='sum')

        # Layer 2: hidden_dim × num_heads → hidden_dim
        self.conv2 = HeteroConv({
            ('user',   'sends_to',       'wallet'): GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout, edge_dim=edge_dim, add_self_loops=False),
            ('wallet', 'receives_from',  'user'):   GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout, edge_dim=edge_dim, add_self_loops=False),
            ('user',   'transacts_with', 'user'):   GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout, edge_dim=edge_dim, add_self_loops=False),
        }, aggr='sum')

        if use_cf_gating:
            self.cf_gate = nn.Linear(1, hidden_dim)

        self.output_proj  = nn.Linear(hidden_dim, out_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple, torch.Tensor],
        edge_attr_dict: Optional[Dict[Tuple, torch.Tensor]] = None,
        cate_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns user node embeddings [num_users, out_dim].
        """
        # Input projection
        x_dict = {
            ntype: self.input_projs[ntype](x)
            for ntype, x in x_dict.items()
            if ntype in self.input_projs
        }

        # Layer 1
        kwargs1 = {'edge_attr_dict': edge_attr_dict} if (self.use_arm_weights and edge_attr_dict) else {}
        x_dict = self.conv1(x_dict, edge_index_dict, **kwargs1)
        x_dict = {k: self.dropout_layer(F.relu(v)) for k, v in x_dict.items()}

        # Layer 2
        kwargs2 = {'edge_attr_dict': edge_attr_dict} if (self.use_arm_weights and edge_attr_dict) else {}
        x_dict = self.conv2(x_dict, edge_index_dict, **kwargs2)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        user_emb = x_dict['user']

        # CF gating
        if self.use_cf_gating and cate_scores is not None:
            gate = torch.sigmoid(self.cf_gate(cate_scores))
            user_emb = user_emb * gate

        return self.output_proj(user_emb)   # [N, out_dim]


class FocalLoss(nn.Module):
    """Focal loss for imbalanced fraud detection."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        pt  = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


class BitoGuardClassifier(nn.Module):
    """
    GNN encoder + single-neuron classification head.

    Used during GNN pre-training (supervised with fraud labels).
    After training, only the encoder (BitoGuardGNN) is kept for embedding
    extraction — the classification head is discarded before XGBoost training.
    """

    def __init__(
        self,
        node_feature_dims: Dict[str, int],
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.3,
        use_arm_weights: bool = True,
        use_cf_gating: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()

        self.gnn = BitoGuardGNN(
            node_feature_dims=node_feature_dims,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_arm_weights=use_arm_weights,
            use_cf_gating=use_cf_gating,
        )
        self.classifier = nn.Linear(out_dim, 1)
        self.loss_fn    = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(
        self,
        x_dict, edge_index_dict,
        edge_attr_dict=None, cate_scores=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.gnn(x_dict, edge_index_dict, edge_attr_dict, cate_scores)
        logits     = self.classifier(embeddings).squeeze(-1)
        return embeddings, logits

    def compute_loss(self, logits, targets):
        return self.loss_fn(logits, targets)

    def predict(self, logits):
        return torch.sigmoid(logits)
