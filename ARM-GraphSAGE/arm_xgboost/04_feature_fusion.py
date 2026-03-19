"""
Module 4: Feature Fusion
Merges tabular, ARM, and CATE features into a single node feature matrix
that will be used as GNN node features.

Input : user_features.csv, arm_features.csv, cate_scores.csv
Output: results/features/node_features.csv
"""

import pandas as pd
from pathlib import Path
import argparse

from utils.utils import load_config, create_output_dirs


def merge_all_features(
    tabular:  pd.DataFrame,
    arm:      pd.DataFrame,
    cate:     pd.DataFrame,
) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("MODULE 4: FEATURE FUSION")
    print("=" * 80)
    print(f"\n  Tabular : {tabular.shape}")
    print(f"  ARM     : {arm.shape}")
    print(f"  CATE    : {cate.shape}")

    node_features = (
        tabular
        .merge(arm,  on='user_id', how='left')
        .merge(cate, on='user_id', how='left')
        .fillna(0)
    )

    tab_n  = tabular.shape[1]  - 1
    arm_n  = arm.shape[1]      - 1
    cate_n = cate.shape[1]     - 1

    print(f"\nMerged node feature matrix: {node_features.shape}")
    print(f"  Tabular features : {tab_n}")
    print(f"  ARM features     : {arm_n}")
    print(f"  CATE features    : {cate_n}")
    print(f"  Total features   : {tab_n + arm_n + cate_n}")
    print(f"\nFEATURE FUSION COMPLETE\n{'=' * 80}\n")

    return node_features


def main():
    parser = argparse.ArgumentParser(description='Module 4 — Feature Fusion')
    parser.add_argument('--config',  default='arm_xgboost/configs/config.yaml')
    parser.add_argument('--tabular', required=True, help='user_features.csv')
    parser.add_argument('--arm',     required=True, help='arm_features.csv')
    parser.add_argument('--cate',    required=True, help='cate_scores.csv')
    parser.add_argument('--output',  default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    create_output_dirs(config)

    tabular = pd.read_csv(args.tabular)
    arm     = pd.read_csv(args.arm)
    cate    = pd.read_csv(args.cate)

    node_features = merge_all_features(tabular, arm, cate)

    out_dir  = Path(config['data']['output_dir']) / 'features'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output or out_dir / config['data']['node_features']
    node_features.to_csv(out_path, index=False)

    print(f"Saved → {out_path}")


if __name__ == '__main__':
    main()
