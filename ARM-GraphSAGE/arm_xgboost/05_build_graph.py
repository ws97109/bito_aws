"""
Module 5: Heterogeneous Graph Construction
Builds a PyG HeteroData from crypto_transfer transactions.

Graph structure:
  Nodes : user, wallet
  Edges : (user, sends_to, wallet)
           (wallet, receives_from, user)
           (user, transacts_with, user)

Key change vs original:
  - 1e-8 unit scaling is applied to crypto_transfer.ori_samount and
    crypto_transfer.twd_srate before any edge-weight computation.

Input : node_features.csv (from Module 4), optional arm_rules.pkl
Output: results/graphs/graph.pt
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import timedelta
import argparse

from utils.utils import (
    load_config,
    apply_unit_scaling,
    log1p_transform,
    save_pickle,
    load_pickle,
    create_output_dirs,
)


class HeteroGraphBuilder:
    """Build a heterogeneous transaction graph."""

    def __init__(self, config: Dict):
        self.config     = config
        self.graph_cfg  = config['graph']
        self.data_dir   = Path(config['data']['raw_data_dir'])

    # ──────────────────────────────────────────────────────────────
    # Data loading
    # ──────────────────────────────────────────────────────────────

    def load_crypto_transfers(self) -> pd.DataFrame:
        """Load crypto_transfer and apply 1e-8 scaling to amount columns."""
        print("\nLoading crypto transfer data...")

        path = self.data_dir / self.config['data']['crypto_transfer']
        df   = pd.read_csv(path)
        print(f"  Raw records: {len(df):,}")

        # ← Apply 1e-8 scaling to crypto_transfer amount columns
        scale_cols = self.config.get('unit_scaling', {}).get(
            'crypto_transfer', ['ori_samount', 'twd_srate']
        )
        for col in scale_cols:
            if col in df.columns:
                df[col] = df[col] * 1e-8
                print(f"  Scaled: crypto_transfer.{col} × 1e-8")

        if self.graph_cfg['filter_null_relations']:
            df = df[df['relation_user_id'].notna()]
            print(f"  After filter null relation_user_id: {len(df):,}")

        if self.graph_cfg['filter_zero_amounts']:
            amount_col = 'ori_samount' if 'ori_samount' in df.columns else 'amount'
            df = df[df[amount_col] > 0]
            print(f"  After filter zero amounts: {len(df):,}")

        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
            window = self.config['feature_engineering'].get('window_days', 180)
            cutoff = df['created_at'].max() - timedelta(days=window)
            df = df[df['created_at'] >= cutoff]
            print(f"  After {window}-day window: {len(df):,}")

        return df

    # ──────────────────────────────────────────────────────────────
    # Node mappings
    # ──────────────────────────────────────────────────────────────

    def build_node_mappings(
        self,
        crypto_df: pd.DataFrame,
        node_features_df: pd.DataFrame,
    ) -> Tuple[Dict, Dict]:
        print("\nBuilding node ID mappings...")

        user_ids        = node_features_df['user_id'].unique()
        user_id_to_idx  = {uid: idx for idx, uid in enumerate(user_ids)}

        from_col = 'from_wallet_hash' if 'from_wallet_hash' in crypto_df.columns else 'from_wallet'
        to_col   = 'to_wallet_hash'   if 'to_wallet_hash'   in crypto_df.columns else 'to_wallet'

        wallet_ids      = pd.concat([
            crypto_df[from_col].dropna(),
            crypto_df[to_col].dropna()
        ]).unique()
        wallet_id_to_idx = {wid: idx for idx, wid in enumerate(wallet_ids)}

        print(f"  User nodes  : {len(user_id_to_idx):,}")
        print(f"  Wallet nodes: {len(wallet_id_to_idx):,}")
        return user_id_to_idx, wallet_id_to_idx

    # ──────────────────────────────────────────────────────────────
    # Node features
    # ──────────────────────────────────────────────────────────────

    def prepare_node_features(
        self,
        node_features_df: pd.DataFrame,
        user_id_to_idx: Dict,
        wallet_id_to_idx: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        print("\nPreparing node feature tensors...")

        feat_cols = [c for c in node_features_df.columns if c != 'user_id']

        user_feat_list = []
        for uid in sorted(user_id_to_idx, key=user_id_to_idx.get):
            row = node_features_df[node_features_df['user_id'] == uid]
            user_feat_list.append(
                row[feat_cols].values[0] if len(row) > 0 else np.zeros(len(feat_cols))
            )

        user_features = torch.tensor(np.array(user_feat_list), dtype=torch.float)

        num_wallets   = len(wallet_id_to_idx)
        wallet_dim    = min(num_wallets, 100)
        wallet_features = (
            torch.eye(wallet_dim) if num_wallets <= 100
            else torch.randn(num_wallets, 100)
        )

        print(f"  User   features: {user_features.shape}")
        print(f"  Wallet features: {wallet_features.shape}")
        return user_features, wallet_features

    # ──────────────────────────────────────────────────────────────
    # Edge construction
    # ──────────────────────────────────────────────────────────────

    def build_edges(
        self,
        crypto_df: pd.DataFrame,
        user_id_to_idx: Dict,
        wallet_id_to_idx: Dict,
    ) -> Dict:
        print("\nBuilding edges...")

        amount_col = 'ori_samount' if 'ori_samount' in crypto_df.columns else 'amount'
        to_col     = 'to_wallet_hash'   if 'to_wallet_hash'   in crypto_df.columns else 'to_wallet'
        from_col   = 'from_wallet_hash' if 'from_wallet_hash' in crypto_df.columns else 'from_wallet'

        def get_amount(row):
            val = row.get('ori_samount', row.get('amount', 0))
            return float(log1p_transform(pd.Series([val])).iloc[0])

        # (user, sends_to, wallet)
        u2w_edges, u2w_attrs = [], []
        for _, row in crypto_df.iterrows():
            uid = row['user_id']
            wid = row.get(to_col)
            if pd.notna(wid) and uid in user_id_to_idx and wid in wallet_id_to_idx:
                u2w_edges.append([user_id_to_idx[uid], wallet_id_to_idx[wid]])
                u2w_attrs.append([get_amount(row)])

        # (wallet, receives_from, user)
        w2u_edges, w2u_attrs = [], []
        for _, row in crypto_df.iterrows():
            uid = row['user_id']
            wid = row.get(from_col)
            if pd.notna(wid) and uid in user_id_to_idx and wid in wallet_id_to_idx:
                w2u_edges.append([wallet_id_to_idx[wid], user_id_to_idx[uid]])
                w2u_attrs.append([get_amount(row)])

        # (user, transacts_with, user)
        u2u_edges, u2u_attrs = [], []
        for _, row in crypto_df.iterrows():
            uid  = row['user_id']
            ruid = row.get('relation_user_id')
            if pd.notna(ruid) and uid in user_id_to_idx and ruid in user_id_to_idx:
                amt = get_amount(row)
                u2u_edges.append([user_id_to_idx[uid], user_id_to_idx[ruid]])
                u2u_attrs.append([amt])
                if self.graph_cfg['add_reverse_edges']:
                    u2u_edges.append([user_id_to_idx[ruid], user_id_to_idx[uid]])
                    u2u_attrs.append([amt])

        def to_edge_index(edges):
            if edges:
                return torch.tensor(edges, dtype=torch.long).t().contiguous()
            return torch.empty((2, 0), dtype=torch.long)

        def to_edge_attr(attrs):
            if attrs:
                return torch.tensor(attrs, dtype=torch.float)
            return torch.empty((0, 1), dtype=torch.float)

        edges = {
            ('user',   'sends_to',       'wallet'): {'edge_index': to_edge_index(u2w_edges), 'edge_attr': to_edge_attr(u2w_attrs)},
            ('wallet', 'receives_from',  'user'):   {'edge_index': to_edge_index(w2u_edges), 'edge_attr': to_edge_attr(w2u_attrs)},
            ('user',   'transacts_with', 'user'):   {'edge_index': to_edge_index(u2u_edges), 'edge_attr': to_edge_attr(u2u_attrs)},
        }

        for etype, edata in edges.items():
            print(f"  {etype}: {edata['edge_index'].shape[1]:,} edges")

        return edges

    # ──────────────────────────────────────────────────────────────
    # Assemble HeteroData
    # ──────────────────────────────────────────────────────────────

    def build(
        self,
        crypto_df: pd.DataFrame,
        node_features_df: pd.DataFrame,
    ) -> HeteroData:
        print("\n" + "=" * 80)
        print("MODULE 5: HETEROGENEOUS GRAPH CONSTRUCTION")
        print("=" * 80)

        user_id_to_idx, wallet_id_to_idx = self.build_node_mappings(crypto_df, node_features_df)
        user_feat, wallet_feat           = self.prepare_node_features(node_features_df, user_id_to_idx, wallet_id_to_idx)
        edges                            = self.build_edges(crypto_df, user_id_to_idx, wallet_id_to_idx)

        data = HeteroData()
        data['user'].x         = user_feat
        data['user'].num_nodes = len(user_id_to_idx)
        data['user'].id_map    = user_id_to_idx      # user_id → node_idx (stored for Module 6/7)

        data['wallet'].x         = wallet_feat
        data['wallet'].num_nodes = len(wallet_id_to_idx)

        for etype, edata in edges.items():
            data[etype].edge_index = edata['edge_index']
            data[etype].edge_attr  = edata['edge_attr']

        print(f"\nGRAPH COMPLETE")
        print(f"  user nodes  : {data['user'].num_nodes:,}")
        print(f"  wallet nodes: {data['wallet'].num_nodes:,}")
        print(f"  user feat dim: {user_feat.shape[1]}")
        print(f"{'=' * 80}\n")

        return data


def main():
    parser = argparse.ArgumentParser(description='Module 5 — Heterogeneous Graph Construction')
    parser.add_argument('--config',        default='arm_xgboost/configs/config.yaml')
    parser.add_argument('--node_features', required=True, help='node_features.csv from Module 4')
    parser.add_argument('--output',        default=None,  help='Override output graph.pt path')
    args = parser.parse_args()

    config = load_config(args.config)
    create_output_dirs(config)

    builder       = HeteroGraphBuilder(config)
    crypto_df     = builder.load_crypto_transfers()
    node_feat_df  = pd.read_csv(args.node_features)

    hetero_data = builder.build(crypto_df, node_feat_df)

    out_dir  = Path(config['data']['output_dir']) / 'graphs'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output or out_dir / config['data']['hetero_graph']
    torch.save(hetero_data, out_path)

    print(f"Saved → {out_path}")


if __name__ == '__main__':
    main()
