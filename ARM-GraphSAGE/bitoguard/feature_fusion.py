"""
Feature Fusion: Merge tabular features, ARM features, and CATE scores
Output: node_features.csv (complete feature matrix for GNN)
"""

import pandas as pd
import argparse
from pathlib import Path
from utils.utils import load_config


def merge_all_features(
    tabular_features: pd.DataFrame,
    arm_features: pd.DataFrame,
    cate_scores: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge all feature sources into single node feature matrix

    Args:
        tabular_features: From Module 1 (bitoguard_feature_pipeline.py)
        arm_features: From Module 2 (arm_mining.py)
        cate_scores: From Module 3 (causal_forest.py)

    Returns:
        Complete node feature matrix
    """
    print("\n" + "="*80)
    print("FEATURE FUSION")
    print("="*80)

    print(f"\nInput features:")
    print(f"  Tabular features: {tabular_features.shape}")
    print(f"  ARM features: {arm_features.shape}")
    print(f"  CATE scores: {cate_scores.shape}")

    # Merge on user_id
    node_features = tabular_features.copy()

    node_features = node_features.merge(arm_features, on='user_id', how='left')
    node_features = node_features.merge(cate_scores, on='user_id', how='left')

    # Fill NaN values
    node_features = node_features.fillna(0)

    print(f"\nMerged feature matrix: {node_features.shape}")
    print(f"  Total features: {node_features.shape[1] - 1} (excluding user_id)")

    print(f"\nFeature breakdown:")
    tabular_count = tabular_features.shape[1] - 1
    arm_count = arm_features.shape[1] - 1
    cate_count = cate_scores.shape[1] - 1

    print(f"  Tabular: {tabular_count}")
    print(f"  ARM: {arm_count}")
    print(f"  CATE: {cate_count}")
    print(f"  Total: {tabular_count + arm_count + cate_count}")

    print(f"\n{'='*80}")
    print("FEATURE FUSION COMPLETE")
    print(f"{'='*80}\n")

    return node_features


def main():
    parser = argparse.ArgumentParser(description='BitoGuard Feature Fusion')
    parser.add_argument('--config', type=str, default='bitoguard/configs/config.yaml')
    parser.add_argument('--tabular', type=str, required=True, help='user_features.csv')
    parser.add_argument('--arm', type=str, required=True, help='arm_features.csv')
    parser.add_argument('--cate', type=str, required=True, help='cate_scores.csv')
    parser.add_argument('--output', type=str, default=None, help='Output path')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Load all features
    print("Loading feature files...")
    tabular = pd.read_csv(args.tabular)
    arm = pd.read_csv(args.arm)
    cate = pd.read_csv(args.cate)

    # Merge features
    node_features = merge_all_features(tabular, arm, cate)

    # Save output
    output_dir = Path(config['data']['output_dir']) / 'features'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = args.output if args.output else output_dir / config['data']['node_features']
    node_features.to_csv(output_path, index=False)

    print(f"Node features saved to: {output_path}")


if __name__ == "__main__":
    main()
