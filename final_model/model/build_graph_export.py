"""Produce graph_node_list.csv + graph_edge_list.csv directly from the
adjust_data/ tables, without running the GNN model.

Why this exists:
  - main.py only writes graph_*.csv as a side-effect of training the GNN.
  - Our LOO pipeline skips the GNN entirely, so that file is stale.
  - The frontend only needs the graph for visualisation, not for inference.
  - This script is O(seconds) and produces exactly the CSVs the frontend
    reads.

Edges modelled (mirroring Gnn_model.build_transaction_graph):
  - user --[sends]-->    wallet  (crypto_transfer kind=1, sub_kind=0, user → to_wallet_hash)
  - wallet --[funds]-->  user    (crypto_transfer kind=0, sub_kind=0, from_wallet_hash → user)
  - user --[transfers]-> user    (crypto_transfer sub_kind=1, relation_user_id)

Usage:
    python build_graph_export.py \
        --data_dir ../../adjust_data/train \
        --risk_scores ../output/baseline_loo/all_user_risk_scores.csv \
        --output_dir ../output/baseline_loo
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="adjust_data/train directory")
    ap.add_argument("--predict_dir", default=None,
                    help="Optional adjust_data/predict directory")
    ap.add_argument("--risk_scores", required=True,
                    help="Path to all_user_risk_scores.csv (drives node risk)")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load crypto_transfer (train + predict if available)
    frames = [pd.read_csv(os.path.join(args.data_dir, "crypto_transfer_train.csv"),
                          low_memory=False)]
    if args.predict_dir and os.path.isdir(args.predict_dir):
        pred_path = os.path.join(args.predict_dir, "crypto_transfer_predict.csv")
        if os.path.exists(pred_path):
            frames.append(pd.read_csv(pred_path, low_memory=False))
    crypto = pd.concat(frames, ignore_index=True)
    print(f"  loaded {len(crypto):,} crypto_transfer rows")

    # ── Load risk scores for node labels
    rs = pd.read_csv(args.risk_scores, low_memory=False, dtype={"true_label": str})
    rs["user_id"] = rs["user_id"].astype(int)
    score_map = dict(zip(rs["user_id"], rs["risk_score"]))
    label_map = {}
    for _, row in rs.iterrows():
        lbl = row["true_label"]
        if pd.isna(lbl) or lbl == "" or lbl == "nan":
            label_map[row["user_id"]] = None
        else:
            try:
                label_map[row["user_id"]] = int(float(lbl))
            except (ValueError, TypeError):
                label_map[row["user_id"]] = None
    print(f"  loaded {len(rs):,} users from risk scores")

    # ── Build wallet/user node sets from crypto
    external = crypto[crypto["sub_kind"] == 0].copy()
    wallet_from = "from_wallet_hash" if "from_wallet_hash" in external.columns else "from_wallet"
    wallet_to = "to_wallet_hash" if "to_wallet_hash" in external.columns else "to_wallet"

    wallets = pd.concat([
        external[wallet_from].dropna(),
        external[wallet_to].dropna(),
    ]).unique().tolist()
    wallets = [w for w in wallets if isinstance(w, str) and w]
    wallet_set = set(wallets)
    print(f"  unique wallets: {len(wallet_set):,}")

    # Users = everyone in risk_scores (so the graph is aligned with the model)
    all_user_ids = sorted(rs["user_id"].unique().tolist())
    print(f"  users in risk scores: {len(all_user_ids):,}")

    # ── Build node list
    node_rows = []
    for uid in all_user_ids:
        node_rows.append({
            "node_id":    f"user_{uid}",
            "node_type":  "user",
            "risk_score": score_map.get(uid, 0.0),
            "label":      label_map.get(uid),
        })
    for w in wallets:
        node_rows.append({
            "node_id":    f"wallet_{w}",
            "node_type":  "wallet",
            "risk_score": np.nan,
            "label":      np.nan,
        })
    nodes_df = pd.DataFrame(node_rows)
    nodes_df.to_csv(os.path.join(args.output_dir, "graph_node_list.csv"),
                    index=False)
    print(f"  wrote graph_node_list.csv: {len(nodes_df):,} nodes "
          f"({(nodes_df['node_type']=='user').sum():,} users + "
          f"{(nodes_df['node_type']=='wallet').sum():,} wallets)")

    # ── Build edge list
    edge_rows = []
    user_set = set(all_user_ids)

    # user → wallet (withdrawals: kind=1, sub_kind=0)
    wit = external[external["kind"] == 1].dropna(subset=[wallet_to])
    for _, row in wit.iterrows():
        uid = int(row["user_id"])
        w = row[wallet_to]
        if uid in user_set and w in wallet_set:
            edge_rows.append({
                "source":     f"user_{uid}",
                "target":     f"wallet_{w}",
                "source_raw": str(uid),
                "target_raw": w,
                "edge_type":  "user_sends_wallet",
            })

    # wallet → user (deposits: kind=0, sub_kind=0)
    dep = external[external["kind"] == 0].dropna(subset=[wallet_from])
    for _, row in dep.iterrows():
        uid = int(row["user_id"])
        w = row[wallet_from]
        if uid in user_set and w in wallet_set:
            edge_rows.append({
                "source":     f"wallet_{w}",
                "target":     f"user_{uid}",
                "source_raw": w,
                "target_raw": str(uid),
                "edge_type":  "wallet_funds_user",
            })

    # user → user (internal transfers: sub_kind=1)
    internal = crypto[crypto["sub_kind"] == 1].dropna(subset=["relation_user_id"])
    for _, row in internal.iterrows():
        uid = int(row["user_id"])
        rel = int(row["relation_user_id"])
        if uid in user_set and rel in user_set:
            edge_rows.append({
                "source":     f"user_{uid}",
                "target":     f"user_{rel}",
                "source_raw": str(uid),
                "target_raw": str(rel),
                "edge_type":  "user_transfers_user",
            })

    # Deduplicate (raw tables may have multiple tx between the same pair)
    edges_df = pd.DataFrame(edge_rows).drop_duplicates()
    edges_df.to_csv(os.path.join(args.output_dir, "graph_edge_list.csv"),
                    index=False)
    print(f"  wrote graph_edge_list.csv: {len(edges_df):,} edges")
    print("  edge_type breakdown:")
    print(edges_df["edge_type"].value_counts().to_string(header=False))

    # Also drop legacy-name aliases for any stragglers
    nodes_df.to_csv(os.path.join(args.output_dir, "gnn_node_list.csv"),
                    index=False)
    edges_df.to_csv(os.path.join(args.output_dir, "gnn_edge_list.csv"),
                    index=False)


if __name__ == "__main__":
    main()
