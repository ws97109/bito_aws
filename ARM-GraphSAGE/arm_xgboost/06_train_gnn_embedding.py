"""
Module 6: GNN Pre-training + Embedding Extraction
Trains BitoGuardGNN with fraud labels (supervised) or link-prediction loss
(unsupervised), then extracts 64-dim user node embeddings for XGBoost.

Outputs:
  results/features/gnn_embeddings.npy    — float32 array [N_users, out_dim]
  results/features/gnn_user_id_map.csv   — columns: node_idx, user_id
  results/models/gnn_best.pt             — best GNN checkpoint
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader

from models.bitoguard_gnn import BitoGuardClassifier, BitoGuardGNN
from utils.utils import (
    load_config,
    set_seed,
    compute_metrics,
    print_metrics,
    EarlyStopping,
    print_model_summary,
    create_output_dirs,
)


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        embeddings, logits = model(
            x_dict          = batch.x_dict,
            edge_index_dict = batch.edge_index_dict,
            edge_attr_dict  = getattr(batch, 'edge_attr_dict', None),
            cate_scores     = getattr(batch['user'], 'cate_scores', None),
        )

        if not hasattr(batch['user'], 'y'):
            continue

        # Only compute loss on seed nodes (batch input nodes, not neighbours)
        mask = batch['user'].input_id if hasattr(batch['user'], 'input_id') else slice(None)
        loss = model.compute_loss(logits[mask], batch['user'].y[mask])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n += 1

    return total_loss / n if n > 0 else 0.0


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []

    for batch in loader:
        batch = batch.to(device)
        _, logits = model(
            x_dict          = batch.x_dict,
            edge_index_dict = batch.edge_index_dict,
            edge_attr_dict  = getattr(batch, 'edge_attr_dict', None),
            cate_scores     = getattr(batch['user'], 'cate_scores', None),
        )
        mask = batch['user'].input_id if hasattr(batch['user'], 'input_id') else slice(None)
        all_logits.append(logits[mask].cpu())
        if hasattr(batch['user'], 'y'):
            all_labels.append(batch['user'].y[mask].cpu())

    logits = torch.cat(all_logits)
    probs  = torch.sigmoid(logits).numpy()
    preds  = (probs > 0.5).astype(int)

    if all_labels:
        labels  = torch.cat(all_labels).numpy().astype(int)
        metrics = compute_metrics(labels, preds, probs)
        return metrics, probs
    return {}, probs


# ─────────────────────────────────────────────────────────────────────────────
# Label alignment
# ─────────────────────────────────────────────────────────────────────────────

def attach_labels(
    hetero_data,
    user_id_to_idx: dict,
    labels_path: str,
    label_col: str = 'is_fraud',
) -> torch.Tensor:
    """
    Read fraud labels CSV and align to graph node ordering.
    Unlabelled users receive y = 0 (treated as negative during training).
    """
    print(f"\nLoading labels from {labels_path} ...")
    df  = pd.read_csv(labels_path)

    num_users = len(user_id_to_idx)
    y = torch.zeros(num_users, dtype=torch.float)

    matched = 0
    for _, row in df.iterrows():
        uid = row['user_id']
        if uid in user_id_to_idx:
            y[user_id_to_idx[uid]] = float(row[label_col])
            matched += 1

    print(f"  Matched {matched:,} / {len(df):,} labelled users")
    print(f"  Fraud rate: {y.mean():.1%}  ({int(y.sum())} positive / {num_users} total)")
    return y


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Module 6 — GNN Pre-training & Embedding Extraction')
    parser.add_argument('--config',     default='arm_xgboost/configs/config.yaml')
    parser.add_argument('--graph',      required=True,  help='graph.pt from Module 5')
    parser.add_argument('--labels',     default=None,   help='CSV with user_id, is_fraud columns (supervised)')
    parser.add_argument('--label_col',  default='is_fraud')
    parser.add_argument('--cate',       default=None,   help='cate_scores.csv from Module 3 (optional)')
    parser.add_argument('--out_emb',    default=None,   help='Override gnn_embeddings.npy path')
    parser.add_argument('--out_map',    default=None,   help='Override gnn_user_id_map.csv path')
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config['seed'])
    create_output_dirs(config)

    device = torch.device(
        'cuda' if torch.cuda.is_available() and config['device'] == 'cuda' else 'cpu'
    )
    print(f"Device: {device}")

    # ── Load graph ─────────────────────────────────────────────────────────
    print(f"\nLoading graph from {args.graph} ...")
    hetero_data  = torch.load(args.graph, weights_only=False)
    user_id_to_idx = hetero_data['user'].id_map   # {user_id: node_idx}
    num_users    = hetero_data['user'].num_nodes
    print(f"  user nodes  : {num_users:,}")
    print(f"  wallet nodes: {hetero_data['wallet'].num_nodes:,}")

    # ── Attach labels ──────────────────────────────────────────────────────
    if args.labels:
        y = attach_labels(hetero_data, user_id_to_idx, args.labels, args.label_col)
    else:
        print("\nNo labels provided — all nodes labelled 0 (unsupervised mode).")
        print("GNN will train on structural signal only. For best results, provide --labels.")
        y = torch.zeros(num_users, dtype=torch.float)

    hetero_data['user'].y = y

    # ── Optional: CATE scores ──────────────────────────────────────────────
    if args.cate and Path(args.cate).exists():
        cate_df = pd.read_csv(args.cate)
        cate_tensor = torch.zeros(num_users, 1, dtype=torch.float)
        for _, row in cate_df.iterrows():
            uid = row['user_id']
            if uid in user_id_to_idx:
                cate_tensor[user_id_to_idx[uid], 0] = float(row['cate_score'])
        hetero_data['user'].cate_scores = cate_tensor
        print(f"\nCATE scores attached for {len(cate_df):,} users")

    # ── Train / val / test masks ───────────────────────────────────────────
    cfg_tr = config['training']
    n_train = int(num_users * cfg_tr['train_ratio'])
    n_val   = int(num_users * cfg_tr['val_ratio'])

    train_mask = torch.zeros(num_users, dtype=torch.bool)
    val_mask   = torch.zeros(num_users, dtype=torch.bool)
    test_mask  = torch.zeros(num_users, dtype=torch.bool)
    train_mask[:n_train]              = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:]       = True

    hetero_data['user'].train_mask = train_mask
    hetero_data['user'].val_mask   = val_mask
    hetero_data['user'].test_mask  = test_mask

    # ── Model ──────────────────────────────────────────────────────────────
    cfg_gnn = config['gnn']
    node_feature_dims = {
        'user':   hetero_data['user'].x.shape[1],
        'wallet': hetero_data['wallet'].x.shape[1],
    }

    model = BitoGuardClassifier(
        node_feature_dims = node_feature_dims,
        hidden_dim        = cfg_gnn['hidden_dim'],
        out_dim           = cfg_gnn['out_dim'],
        num_heads         = cfg_gnn['num_heads'],
        dropout           = cfg_gnn['dropout'],
        use_arm_weights   = cfg_gnn['use_arm_edge_weights'],
        use_cf_gating     = cfg_gnn['use_cf_gating'],
        focal_alpha       = cfg_tr['focal_alpha'],
        focal_gamma       = cfg_tr['focal_gamma'],
    ).to(device)

    print_model_summary(model)

    # ── Data loaders ───────────────────────────────────────────────────────
    loader_kwargs = dict(
        num_neighbors = cfg_tr['num_neighbors'],
        batch_size    = cfg_tr['batch_size'],
    )
    train_loader = NeighborLoader(hetero_data, input_nodes=('user', train_mask), **loader_kwargs)
    val_loader   = NeighborLoader(hetero_data, input_nodes=('user', val_mask),   **loader_kwargs)

    optimizer     = torch.optim.Adam(model.parameters(), lr=cfg_tr['learning_rate'], weight_decay=cfg_tr['weight_decay'])
    early_stop    = EarlyStopping(patience=cfg_tr['patience'], min_delta=cfg_tr['min_delta'], mode='max')

    model_dir = Path(config['data']['output_dir']) / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = model_dir / 'gnn_best.pt'

    # ── Training loop ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("GNN PRE-TRAINING")
    print("=" * 80)

    best_pr_auc = 0.0

    for epoch in range(cfg_tr['epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics, _ = evaluate(model, val_loader, device)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            pr  = val_metrics.get('pr_auc',  0.0)
            roc = val_metrics.get('roc_auc', 0.0)
            f1  = val_metrics.get('f1_score', 0.0)
            print(f"Epoch {epoch+1:3d}/{cfg_tr['epochs']}  "
                  f"loss={train_loss:.4f}  pr_auc={pr:.4f}  roc_auc={roc:.4f}  f1={f1:.4f}")

        pr_auc = val_metrics.get('pr_auc', 0.0)
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            torch.save(model.state_dict(), best_ckpt)

        if val_metrics and early_stop(pr_auc):
            print(f"\nEarly stop at epoch {epoch + 1}  (best pr_auc={best_pr_auc:.4f})")
            break

    print(f"\nBest val PR-AUC: {best_pr_auc:.4f}")

    # ── Reload best weights ────────────────────────────────────────────────
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, weights_only=True))
        print(f"Reloaded best checkpoint from {best_ckpt}")

    # ── Extract embeddings for ALL users ──────────────────────────────────
    print("\nExtracting GNN embeddings for all users...")
    model.eval()
    hetero_data_dev = hetero_data.to(device)

    with torch.no_grad():
        embeddings, _ = model(
            x_dict          = hetero_data_dev.x_dict,
            edge_index_dict = hetero_data_dev.edge_index_dict,
            edge_attr_dict  = getattr(hetero_data_dev, 'edge_attr_dict', None),
            cate_scores     = getattr(hetero_data_dev['user'], 'cate_scores', None),
        )

    embeddings_np = embeddings.cpu().numpy()   # [N_users, out_dim]
    print(f"  Embedding shape: {embeddings_np.shape}")

    # ── Save embeddings ────────────────────────────────────────────────────
    feat_dir = Path(config['data']['output_dir']) / 'features'
    feat_dir.mkdir(parents=True, exist_ok=True)

    emb_path = args.out_emb or feat_dir / config['data']['gnn_embeddings']
    map_path = args.out_map or feat_dir / config['data']['gnn_user_id_map']

    np.save(emb_path, embeddings_np)
    print(f"  Embeddings saved → {emb_path}")

    # node_idx → user_id mapping CSV
    idx_to_uid = {v: k for k, v in user_id_to_idx.items()}
    map_df = pd.DataFrame({
        'node_idx': list(range(num_users)),
        'user_id':  [idx_to_uid[i] for i in range(num_users)],
    })
    map_df.to_csv(map_path, index=False)
    print(f"  User-ID map saved → {map_path}")

    print(f"\n{'=' * 80}")
    print("MODULE 6 COMPLETE")
    print(f"  GNN embedding dim : {embeddings_np.shape[1]}")
    print(f"  Users embedded    : {embeddings_np.shape[0]:,}")
    print(f"  Best val PR-AUC   : {best_pr_auc:.4f}")
    print(f"{'=' * 80}\n")


if __name__ == '__main__':
    main()
