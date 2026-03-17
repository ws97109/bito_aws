"""
BitoGuard Training Example
Simplified training script demonstrating how to train the BitoGuardGNN model
"""

import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
import numpy as np
import pandas as pd
from pathlib import Path

from models.bitoguard_gnn import BitoGuardClassifier
from utils.utils import load_config, set_seed, compute_metrics, EarlyStopping, print_model_summary


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        # Forward pass
        embeddings, logits = model(
            x_dict=batch.x_dict,
            edge_index_dict=batch.edge_index_dict,
            edge_attr_dict=batch.edge_attr_dict if hasattr(batch, 'edge_attr_dict') else None,
            cate_scores=batch['user'].cate_scores if hasattr(batch['user'], 'cate_scores') else None
        )

        # Compute loss (assuming batch has labels)
        if hasattr(batch['user'], 'y'):
            loss = model.compute_loss(logits, batch['user'].y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model"""
    model.eval()

    all_logits = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)

        embeddings, logits = model(
            x_dict=batch.x_dict,
            edge_index_dict=batch.edge_index_dict,
            edge_attr_dict=batch.edge_attr_dict if hasattr(batch, 'edge_attr_dict') else None,
            cate_scores=batch['user'].cate_scores if hasattr(batch['user'], 'cate_scores') else None
        )

        all_logits.append(logits.cpu())
        if hasattr(batch['user'], 'y'):
            all_labels.append(batch['user'].y.cpu())

    all_logits = torch.cat(all_logits)
    all_probs = torch.sigmoid(all_logits).numpy()

    if len(all_labels) > 0:
        all_labels = torch.cat(all_labels).numpy()
        all_preds = (all_probs > 0.5).astype(int)

        metrics = compute_metrics(all_labels, all_preds, all_probs)
        return metrics, all_probs
    else:
        return {}, all_probs


def main():
    # Load configuration
    config = load_config('configs/config.yaml')
    set_seed(config['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() and config['device'] == 'cuda' else 'cpu')
    print(f"Using device: {device}\n")

    # Load heterogeneous graph
    print("Loading heterogeneous graph...")
    graph_path = Path(config['data']['output_dir']) / 'graphs' / config['data']['hetero_graph']
    hetero_data = torch.load(graph_path, weights_only=False)

    print(f"Graph loaded:")
    print(f"  User nodes: {hetero_data['user'].num_nodes}")
    print(f"  Wallet nodes: {hetero_data['wallet'].num_nodes}")
    print(f"  Edge types: {list(hetero_data.edge_types)}\n")

    # Add labels (dummy labels for demonstration)
    # In practice, load real fraud labels
    num_users = hetero_data['user'].num_nodes
    hetero_data['user'].y = torch.randint(0, 2, (num_users,), dtype=torch.float)

    # Add CATE scores (if available)
    cate_path = Path(config['data']['output_dir']) / 'features' / config['data']['cate_scores']
    if cate_path.exists():
        cate_df = pd.read_csv(cate_path)
        cate_scores = torch.tensor(cate_df['cate_score'].values, dtype=torch.float).unsqueeze(-1)
        hetero_data['user'].cate_scores = cate_scores
        print("CATE scores loaded\n")

    # Create data splits
    num_train = int(num_users * config['training']['train_ratio'])
    num_val = int(num_users * config['training']['val_ratio'])

    train_mask = torch.zeros(num_users, dtype=torch.bool)
    val_mask = torch.zeros(num_users, dtype=torch.bool)
    test_mask = torch.zeros(num_users, dtype=torch.bool)

    train_mask[:num_train] = True
    val_mask[num_train:num_train + num_val] = True
    test_mask[num_train + num_val:] = True

    hetero_data['user'].train_mask = train_mask
    hetero_data['user'].val_mask = val_mask
    hetero_data['user'].test_mask = test_mask

    # Initialize model
    print("Initializing BitoGuardGNN model...\n")
    node_feature_dims = {
        'user': hetero_data['user'].x.shape[1],
        'wallet': hetero_data['wallet'].x.shape[1]
    }

    model = BitoGuardClassifier(
        node_feature_dims=node_feature_dims,
        hidden_dim=config['gnn']['hidden_dim'],
        out_dim=config['gnn']['out_dim'],
        num_heads=config['gnn']['num_heads'],
        dropout=config['gnn']['dropout'],
        use_arm_weights=config['gnn']['use_arm_edge_weights'],
        use_cf_gating=config['gnn']['use_cf_gating'],
        focal_alpha=config['training']['focal_alpha'],
        focal_gamma=config['training']['focal_gamma']
    )

    model = model.to(device)
    print_model_summary(model)

    # Create data loaders
    train_loader = NeighborLoader(
        hetero_data,
        num_neighbors=config['training']['num_neighbors'],
        batch_size=config['training']['batch_size'],
        input_nodes=('user', train_mask),
    )

    val_loader = NeighborLoader(
        hetero_data,
        num_neighbors=config['training']['num_neighbors'],
        batch_size=config['training']['batch_size'],
        input_nodes=('user', val_mask),
    )

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['training']['patience'],
        min_delta=config['training']['min_delta'],
        mode='max'
    )

    # Training loop
    print("Starting training...\n")
    print("=" * 80)

    best_val_metric = 0

    for epoch in range(config['training']['epochs']):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_metrics, _ = evaluate(model, val_loader, device)

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{config['training']['epochs']}")
            print(f"  Train Loss: {train_loss:.4f}")
            if val_metrics:
                print(f"  Val PR-AUC: {val_metrics.get('pr_auc', 0):.4f}")
                print(f"  Val ROC-AUC: {val_metrics.get('roc_auc', 0):.4f}")
                print(f"  Val F1: {val_metrics.get('f1_score', 0):.4f}")

        # Early stopping check
        if val_metrics and 'pr_auc' in val_metrics:
            if early_stopping(val_metrics['pr_auc']):
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

            if val_metrics['pr_auc'] > best_val_metric:
                best_val_metric = val_metrics['pr_auc']
                # Save best model
                model_path = Path(config['data']['output_dir']) / 'models' / 'best_model.pt'
                model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), model_path)

    print("=" * 80)
    print(f"\nTraining complete!")
    print(f"Best validation PR-AUC: {best_val_metric:.4f}\n")

    # Save final embeddings
    print("Generating final embeddings...")
    model.eval()
    with torch.no_grad():
        hetero_data = hetero_data.to(device)
        embeddings, _ = model(
            x_dict=hetero_data.x_dict,
            edge_index_dict=hetero_data.edge_index_dict
        )

    embeddings_path = Path(config['data']['output_dir']) / 'features' / config['data']['gnn_embeddings']
    np.save(embeddings_path, embeddings.cpu().numpy())
    print(f"Embeddings saved to: {embeddings_path}")


if __name__ == "__main__":
    main()
