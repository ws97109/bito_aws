"""
Create dummy CATE scores (skip Causal Forest step)
"""

import pandas as pd
import argparse
from pathlib import Path
from utils.utils import load_config


def create_dummy_cate_scores(user_features_path, output_path):
    """Create dummy CATE scores for all users"""
    print("Creating dummy CATE scores (skipping Causal Forest)...")

    # Load user features
    user_features = pd.read_csv(user_features_path)

    # Create dummy CATE scores (all zeros)
    cate_df = pd.DataFrame({
        'user_id': user_features['user_id'],
        'cate_score': 0.0  # Dummy value
    })

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cate_df.to_csv(output_path, index=False)

    print(f"✓ Created dummy CATE scores for {len(cate_df)} users")
    print(f"✓ Saved to: {output_path}")

    return cate_df


def main():
    parser = argparse.ArgumentParser(description='Create dummy CATE scores')
    parser.add_argument('--features', type=str, required=True, help='user_features.csv path')
    parser.add_argument('--output', type=str, default='results/features/cate_scores.csv')

    args = parser.parse_args()

    create_dummy_cate_scores(args.features, args.output)


if __name__ == "__main__":
    main()
