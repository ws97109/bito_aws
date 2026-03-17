"""
Module 4: Heterogeneous Graph Construction
Builds PyG HeteroData from crypto_transfer transactions

Graph structure:
  Nodes: user, wallet
  Edges: (user, sends_to, wallet), (wallet, receives_from, user), (user, transacts_with, user)

Output: graph.pt (PyG HeteroData object)
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from pathlib import Path
from typing import Dict, Tuple
import argparse
from datetime import timedelta

from utils.utils import load_config, save_pickle, log1p_transform


class HeteroGraphBuilder:
    """Build heterogeneous graph from transaction data"""

    def __init__(self, config: Dict):
        self.config = config
        self.graph_config = config['graph']
        self.data_dir = Path(config['data']['raw_data_dir'])

    def load_crypto_transfers(self) -> pd.DataFrame:
        """Load and preprocess crypto transfer data"""
        print("\nLoading crypto transfer data...")

        crypto_path = self.data_dir / self.config['data']['crypto_transfer']
        df = pd.read_csv(crypto_path)

        print(f"  Loaded {len(df)} crypto transfer records")

        # Preprocessing
        if self.graph_config['filter_null_relations']:
            df = df[df['relation_user_id'].notna()]
            print(f"  After filtering null relation_user_id: {len(df)}")

        if self.graph_config['filter_zero_amounts']:
            # 支援 amount 或 ori_samount
            amount_col = 'ori_samount' if 'ori_samount' in df.columns else 'amount'
            df = df[df[amount_col] > 0]
            print(f"  After filtering zero amounts: {len(df)}")

        # Convert timestamps
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

        # Time window filtering
        if 'window_days' in self.config['feature_engineering']:
            window_days = self.config['feature_engineering']['window_days']
            cutoff = df['created_at'].max() - timedelta(days=window_days)
            df = df[df['created_at'] >= cutoff]
            print(f"  After time window ({window_days} days): {len(df)}")

        return df

    def build_node_mappings(
        self,
        crypto_df: pd.DataFrame,
        node_features_df: pd.DataFrame
    ) -> Tuple[Dict, Dict]:
        """
        Create mappings from IDs to node indices

        Returns:
            (user_id_to_idx, wallet_id_to_idx)
        """
        print("\nBuilding node ID mappings...")

        # User nodes: from node_features (all users in dataset)
        user_ids = node_features_df['user_id'].unique()
        user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}

        # Wallet nodes: from crypto transfers
        # 支援兩種欄位名稱: from_wallet/to_wallet 或 from_wallet_hash/to_wallet_hash
        from_wallet_col = 'from_wallet_hash' if 'from_wallet_hash' in crypto_df.columns else 'from_wallet'
        to_wallet_col = 'to_wallet_hash' if 'to_wallet_hash' in crypto_df.columns else 'to_wallet'

        wallet_ids = pd.concat([
            crypto_df[from_wallet_col].dropna(),
            crypto_df[to_wallet_col].dropna()
        ]).unique()
        wallet_id_to_idx = {wid: idx for idx, wid in enumerate(wallet_ids)}

        print(f"  User nodes: {len(user_id_to_idx)}")
        print(f"  Wallet nodes: {len(wallet_id_to_idx)}")

        return user_id_to_idx, wallet_id_to_idx

    def prepare_node_features(
        self,
        node_features_df: pd.DataFrame,
        user_id_to_idx: Dict,
        wallet_id_to_idx: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare node feature tensors

        Returns:
            (user_features, wallet_features)
        """
        print("\nPreparing node features...")

        # User features: from node_features_df (excluding user_id column)
        feature_cols = [col for col in node_features_df.columns if col != 'user_id']
        user_features_list = []

        for user_id in sorted(user_id_to_idx.keys(), key=lambda x: user_id_to_idx[x]):
            user_row = node_features_df[node_features_df['user_id'] == user_id]
            if len(user_row) > 0:
                features = user_row[feature_cols].values[0]
            else:
                # User not in features (shouldn't happen, but handle gracefully)
                features = np.zeros(len(feature_cols))
            user_features_list.append(features)

        user_features = torch.tensor(np.array(user_features_list), dtype=torch.float)

        # Wallet features: simple degree-based features (can be enhanced later)
        # For now, use one-hot encoding or simple placeholder
        num_wallets = len(wallet_id_to_idx)
        wallet_features = torch.eye(min(num_wallets, 100))  # One-hot up to 100 dims

        if num_wallets > 100:
            # If too many wallets, use random projection
            wallet_features = torch.randn(num_wallets, 100)

        print(f"  User feature dim: {user_features.shape}")
        print(f"  Wallet feature dim: {wallet_features.shape}")

        return user_features, wallet_features

    def build_edges(
        self,
        crypto_df: pd.DataFrame,
        user_id_to_idx: Dict,
        wallet_id_to_idx: Dict,
        arm_rules: pd.DataFrame = None
    ) -> Dict:
        """
        Build edge indices and attributes for all edge types

        Returns:
            Dictionary of edge_index and edge_attr for each edge type
        """
        print("\nBuilding graph edges...")

        edges = {}

        # (user, sends_to, wallet)
        user_to_wallet_edges = []
        user_to_wallet_attrs = []

        for _, row in crypto_df.iterrows():
            user_id = row['user_id']
            # 支援兩種欄位名稱: to_wallet 或 to_wallet_hash
            to_wallet = row.get('to_wallet_hash', row.get('to_wallet'))

            if pd.notna(to_wallet) and user_id in user_id_to_idx and to_wallet in wallet_id_to_idx:
                user_idx = user_id_to_idx[user_id]
                wallet_idx = wallet_id_to_idx[to_wallet]

                user_to_wallet_edges.append([user_idx, wallet_idx])

                # Edge attributes: amount (log-transformed)
                # 支援 amount 或 ori_samount
                amount_val = row.get('ori_samount', row.get('amount', 0))
                amount = log1p_transform(pd.Series([amount_val])).iloc[0]
                user_to_wallet_attrs.append([amount])

        edges[('user', 'sends_to', 'wallet')] = {
            'edge_index': torch.tensor(user_to_wallet_edges, dtype=torch.long).t().contiguous(),
            'edge_attr': torch.tensor(user_to_wallet_attrs, dtype=torch.float)
        }

        print(f"  (user, sends_to, wallet): {len(user_to_wallet_edges)} edges")

        # (wallet, receives_from, user) - reverse edges
        wallet_to_user_edges = []
        wallet_to_user_attrs = []

        for _, row in crypto_df.iterrows():
            user_id = row['user_id']
            # 支援兩種欄位名稱: from_wallet 或 from_wallet_hash
            from_wallet = row.get('from_wallet_hash', row.get('from_wallet'))

            if pd.notna(from_wallet) and user_id in user_id_to_idx and from_wallet in wallet_id_to_idx:
                user_idx = user_id_to_idx[user_id]
                wallet_idx = wallet_id_to_idx[from_wallet]

                wallet_to_user_edges.append([wallet_idx, user_idx])

                # 支援 amount 或 ori_samount
                amount_val = row.get('ori_samount', row.get('amount', 0))
                amount = log1p_transform(pd.Series([amount_val])).iloc[0]
                wallet_to_user_attrs.append([amount])

        edges[('wallet', 'receives_from', 'user')] = {
            'edge_index': torch.tensor(wallet_to_user_edges, dtype=torch.long).t().contiguous() if wallet_to_user_edges else torch.empty((2, 0), dtype=torch.long),
            'edge_attr': torch.tensor(wallet_to_user_attrs, dtype=torch.float) if wallet_to_user_attrs else torch.empty((0, 1), dtype=torch.float)
        }

        print(f"  (wallet, receives_from, user): {len(wallet_to_user_edges)} edges")

        # (user, transacts_with, user)
        user_to_user_edges = []
        user_to_user_attrs = []

        for _, row in crypto_df.iterrows():
            user_id = row['user_id']
            relation_user_id = row['relation_user_id']

            if pd.notna(relation_user_id) and user_id in user_id_to_idx and relation_user_id in user_id_to_idx:
                user_idx = user_id_to_idx[user_id]
                relation_idx = user_id_to_idx[relation_user_id]

                user_to_user_edges.append([user_idx, relation_idx])

                # 支援 amount 或 ori_samount
                amount_val = row.get('ori_samount', row.get('amount', 0))
                amount = log1p_transform(pd.Series([amount_val])).iloc[0]
                # TODO: Add ARM edge weight if available
                user_to_user_attrs.append([amount])

                # Add reverse edge if configured
                if self.graph_config['add_reverse_edges']:
                    user_to_user_edges.append([relation_idx, user_idx])
                    user_to_user_attrs.append([amount])

        edges[('user', 'transacts_with', 'user')] = {
            'edge_index': torch.tensor(user_to_user_edges, dtype=torch.long).t().contiguous() if user_to_user_edges else torch.empty((2, 0), dtype=torch.long),
            'edge_attr': torch.tensor(user_to_user_attrs, dtype=torch.float) if user_to_user_attrs else torch.empty((0, 1), dtype=torch.float)
        }

        print(f"  (user, transacts_with, user): {len(user_to_user_edges)} edges")

        return edges

    def build_hetero_data(
        self,
        crypto_df: pd.DataFrame,
        node_features_df: pd.DataFrame,
        arm_rules: pd.DataFrame = None
    ) -> HeteroData:
        """
        Build complete HeteroData object

        Args:
            crypto_df: Crypto transfer transactions
            node_features_df: Node features from feature fusion
            arm_rules: ARM rules (optional)

        Returns:
            PyG HeteroData object
        """
        print("\n" + "="*80)
        print("HETEROGENEOUS GRAPH CONSTRUCTION")
        print("="*80)

        # Build node mappings
        user_id_to_idx, wallet_id_to_idx = self.build_node_mappings(crypto_df, node_features_df)

        # Prepare node features
        user_features, wallet_features = self.prepare_node_features(
            node_features_df, user_id_to_idx, wallet_id_to_idx
        )

        # Build edges
        edges = self.build_edges(crypto_df, user_id_to_idx, wallet_id_to_idx, arm_rules)

        # Create HeteroData
        data = HeteroData()

        # Add node features
        data['user'].x = user_features
        data['user'].num_nodes = len(user_id_to_idx)

        data['wallet'].x = wallet_features
        data['wallet'].num_nodes = len(wallet_id_to_idx)

        # Add edges
        for edge_type, edge_data in edges.items():
            data[edge_type].edge_index = edge_data['edge_index']
            data[edge_type].edge_attr = edge_data['edge_attr']

        # Store ID mappings as metadata
        data['user'].id_map = user_id_to_idx
        data['wallet'].id_map = wallet_id_to_idx

        print(f"\n{'='*80}")
        print("HETEROGENEOUS GRAPH COMPLETE")
        print(f"  Nodes: user={data['user'].num_nodes}, wallet={data['wallet'].num_nodes}")
        print(f"  Edges:")
        for edge_type in edges.keys():
            print(f"    {edge_type}: {data[edge_type].edge_index.shape[1]}")
        print(f"{'='*80}\n")

        return data


def main():
    parser = argparse.ArgumentParser(description='BitoGuard Heterogeneous Graph Construction')
    parser.add_argument('--config', type=str, default='bitoguard/configs/config.yaml')
    parser.add_argument('--node_features', type=str, required=True, help='node_features.csv')
    parser.add_argument('--arm_rules', type=str, default=None, help='arm_rules.pkl (optional)')
    parser.add_argument('--output', type=str, default=None, help='Output graph.pt path')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Initialize builder
    builder = HeteroGraphBuilder(config)

    # Load data
    crypto_df = builder.load_crypto_transfers()
    node_features_df = pd.read_csv(args.node_features)

    arm_rules = None
    if args.arm_rules:
        from utils.utils import load_pickle
        arm_rules = load_pickle(args.arm_rules)

    # Build graph
    hetero_data = builder.build_hetero_data(crypto_df, node_features_df, arm_rules)

    # Save output
    output_dir = Path(config['data']['output_dir']) / 'graphs'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = args.output if args.output else output_dir / config['data']['hetero_graph']
    torch.save(hetero_data, output_path)

    print(f"Heterogeneous graph saved to: {output_path}")


if __name__ == "__main__":
    main()
