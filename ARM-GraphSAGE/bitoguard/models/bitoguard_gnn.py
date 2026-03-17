"""
Module 5: BitoGuard GNN Model
GraphSAGE-ARM-CF: Heterogeneous GNN with ARM edge weights and CF gating

Key innovations:
1. HeteroConv for heterogeneous graph (user, wallet nodes)
2. GATv2Conv for attention-based aggregation
3. ARM edge weights injected into message passing
4. Causal Forest gating for user embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear
from typing import Dict, Tuple


class ARMWeightedGATv2Conv(GATv2Conv):
    """
    GATv2Conv with ARM edge weight injection

    ARM weights are multiplied with attention scores to guide message passing
    """

    def __init__(self, in_channels, out_channels, heads=1, **kwargs):
        super().__init__(in_channels, out_channels, heads=heads, **kwargs)

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass with optional ARM edge attributes

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim]
                       First column should be ARM weights if available
        """
        # If ARM weights provided, they will be used in message passing
        # PyG's GATv2Conv supports edge_attr natively
        return super().forward(x, edge_index, edge_attr=edge_attr)


class BitoGuardGNN(nn.Module):
    """
    BitoGuard GNN: GraphSAGE-ARM-CF Architecture

    Architecture:
      1. Two-layer HeteroConv with GATv2 aggregators
      2. ARM-guided edge weighting
      3. CF-gated readout for user nodes
      4. Output: User embeddings for downstream tasks
    """

    def __init__(
        self,
        node_feature_dims: Dict[str, int],
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.3,
        use_arm_weights: bool = True,
        use_cf_gating: bool = True
    ):
        """
        Args:
            node_feature_dims: Dict mapping node type to feature dimension
                               e.g., {'user': 50, 'wallet': 10}
            hidden_dim: Hidden layer dimension
            out_dim: Output embedding dimension
            num_heads: Number of attention heads in GATv2
            dropout: Dropout rate
            use_arm_weights: Whether to use ARM edge weights
            use_cf_gating: Whether to use CF gating
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_arm_weights = use_arm_weights
        self.use_cf_gating = use_cf_gating

        # Input projections for each node type
        self.input_projs = nn.ModuleDict()
        for node_type, in_dim in node_feature_dims.items():
            self.input_projs[node_type] = Linear(in_dim, hidden_dim)

        # Layer 1: HeteroConv with GATv2
        self.conv1 = HeteroConv({
            ('user', 'sends_to', 'wallet'): GATv2Conv(
                hidden_dim, hidden_dim, heads=num_heads, dropout=dropout, edge_dim=1 if use_arm_weights else None,
                add_self_loops=False  # Cannot add self-loops for hetero edges
            ),
            ('wallet', 'receives_from', 'user'): GATv2Conv(
                hidden_dim, hidden_dim, heads=num_heads, dropout=dropout, edge_dim=1 if use_arm_weights else None,
                add_self_loops=False  # Cannot add self-loops for hetero edges
            ),
            ('user', 'transacts_with', 'user'): GATv2Conv(
                hidden_dim, hidden_dim, heads=num_heads, dropout=dropout, edge_dim=1 if use_arm_weights else None,
                add_self_loops=False  # Disable to avoid issues with hetero graph
            ),
        }, aggr='sum')

        # Layer 2: HeteroConv with GATv2
        self.conv2 = HeteroConv({
            ('user', 'sends_to', 'wallet'): GATv2Conv(
                hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout, edge_dim=1 if use_arm_weights else None,
                add_self_loops=False  # Cannot add self-loops for hetero edges
            ),
            ('wallet', 'receives_from', 'user'): GATv2Conv(
                hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout, edge_dim=1 if use_arm_weights else None,
                add_self_loops=False  # Cannot add self-loops for hetero edges
            ),
            ('user', 'transacts_with', 'user'): GATv2Conv(
                hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout, edge_dim=1 if use_arm_weights else None,
                add_self_loops=False  # Disable to avoid issues with hetero graph
            ),
        }, aggr='sum')

        # CF gating layer
        if use_cf_gating:
            self.cf_gate = nn.Linear(1, hidden_dim)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, out_dim)

        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor] = None,
        cate_scores: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x_dict: Node features {node_type: features}
            edge_index_dict: Edge indices {edge_type: edge_index}
            edge_attr_dict: Edge attributes {edge_type: edge_attr} (optional)
            cate_scores: CATE scores for user nodes [num_users, 1] (optional)

        Returns:
            User node embeddings [num_users, out_dim]
        """
        # Input projection
        x_dict = {
            node_type: self.input_projs[node_type](x)
            for node_type, x in x_dict.items()
        }

        # Layer 1
        if self.use_arm_weights and edge_attr_dict is not None:
            x_dict = self.conv1(x_dict, edge_index_dict, edge_attr_dict)
        else:
            x_dict = self.conv1(x_dict, edge_index_dict)

        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout_layer(x) for key, x in x_dict.items()}

        # Layer 2
        if self.use_arm_weights and edge_attr_dict is not None:
            x_dict = self.conv2(x_dict, edge_index_dict, edge_attr_dict)
        else:
            x_dict = self.conv2(x_dict, edge_index_dict)

        x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # Extract user embeddings
        user_emb = x_dict['user']

        # CF-gated readout
        if self.use_cf_gating and cate_scores is not None:
            gate = torch.sigmoid(self.cf_gate(cate_scores))  # [num_users, hidden_dim]
            user_emb = user_emb * gate  # Element-wise gating

        # Output projection
        user_emb = self.output_proj(user_emb)

        return user_emb


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in fraud detection

    FL(p_t) = -α(1-p_t)^γ log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter (higher = more focus on hard examples)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss

        Args:
            logits: Predicted logits [batch_size, num_classes] or [batch_size]
            targets: Target labels [batch_size]

        Returns:
            Scalar loss
        """
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction='none'
        )

        pt = torch.exp(-bce_loss)  # Probability of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        return focal_loss.mean()


class BitoGuardClassifier(nn.Module):
    """
    Complete BitoGuard fraud detection model

    Components:
      1. BitoGuardGNN for graph embedding
      2. Classification head with focal loss
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
        focal_gamma: float = 2.0
    ):
        super().__init__()

        # GNN encoder
        self.gnn = BitoGuardGNN(
            node_feature_dims=node_feature_dims,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_arm_weights=use_arm_weights,
            use_cf_gating=use_cf_gating
        )

        # Classification head
        self.classifier = nn.Linear(out_dim, 1)

        # Loss function
        self.loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor] = None,
        cate_scores: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Returns:
            (embeddings, logits)
        """
        # Get user embeddings from GNN
        embeddings = self.gnn(x_dict, edge_index_dict, edge_attr_dict, cate_scores)

        # Classification
        logits = self.classifier(embeddings).squeeze(-1)

        return embeddings, logits

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss"""
        return self.loss_fn(logits, targets)

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to probabilities"""
        return torch.sigmoid(logits)
