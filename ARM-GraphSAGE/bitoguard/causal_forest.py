"""
Module 3: Causal Forest for Treatment Effect Estimation
Computes CATE (Conditional Average Treatment Effect) scores

Output: cate_scores.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from typing import Dict

# Note: econml may not be installed by default
# Install with: pip install econml
try:
    from econml.dml import CausalForestDML
    ECONML_AVAILABLE = True
except ImportError:
    print("Warning: econml not installed. Install with: pip install econml")
    ECONML_AVAILABLE = False

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from utils.utils import load_config


class CausalForestEstimator:
    """Estimate causal treatment effects using Causal Forest"""

    def __init__(self, config: Dict):
        self.config = config
        self.cf_config = config['causal_forest']

    def define_treatment(self, user_features: pd.DataFrame) -> pd.Series:
        """
        Define treatment variable: Recent USDT trading volume surge

        Treatment T = 1 if recent_volume > threshold * historical_mean
        """
        print("\nDefining treatment variable...")

        # For demonstration, use trade_count as proxy for trading activity
        # In practice, this would use temporal windowing on actual transaction data

        df = user_features.copy()

        # Simple treatment definition: high trading speed
        treatment_threshold = self.cf_config['treatment_threshold']

        if 'trade_speed_score' in df.columns:
            historical_mean = df['trade_speed_score'].mean()
            treatment = (df['trade_speed_score'] > treatment_threshold * historical_mean).astype(int)
        else:
            # Fallback: random treatment for demonstration
            print("  Warning: trade_speed_score not found, using random treatment")
            treatment = np.random.binomial(1, 0.3, size=len(df))

        print(f"  Treatment: {treatment.sum()} / {len(treatment)} users ({treatment.mean():.2%})")

        return pd.Series(treatment, index=df.index)

    def define_outcome(self, user_features: pd.DataFrame, labels: pd.Series = None) -> pd.Series:
        """
        Define outcome variable

        If labels available: use fraud labels
        Otherwise: use anomaly score from features
        """
        print("\nDefining outcome variable...")

        if labels is not None:
            outcome = labels
            print(f"  Using provided fraud labels as outcome")
        else:
            # Create anomaly score from suspicious patterns
            df = user_features.copy()

            anomaly_score = 0

            if 'twd_withdraw_ratio' in df.columns:
                anomaly_score += (df['twd_withdraw_ratio'] > 0.85).astype(int)

            if 'twd_night_tx_ratio' in df.columns:
                anomaly_score += (df['twd_night_tx_ratio'] > 0.4).astype(int)

            if 'trade_buy_ratio' in df.columns:
                anomaly_score += (df['trade_buy_ratio'] > 0.9).astype(int)

            if 'twd_unique_ip_count' in df.columns:
                anomaly_score += (df['twd_unique_ip_count'] > 5).astype(int)

            outcome = (anomaly_score >= 2).astype(int)
            print(f"  Using anomaly score as outcome (threshold >= 2)")
            print(f"  Outcome: {outcome.sum()} / {len(outcome)} flagged ({outcome.mean():.2%})")

        return pd.Series(outcome, index=user_features.index)

    def prepare_covariates(self, user_features: pd.DataFrame) -> pd.DataFrame:
        """
        Select features to use as covariates (control variables)

        Covariates should include baseline characteristics to control for confounding
        """
        print("\nPreparing covariates...")

        # Select stable baseline features (exclude treatment-related features)
        covariate_cols = [
            'age', 'sex', 'career', 'income_source', 'user_source',
            'kyc_days_to_l1', 'kyc_days_to_l2',
            'twd_tx_count', 'twd_deposit_count'
        ]

        available_cols = [col for col in covariate_cols if col in user_features.columns]

        X = user_features[available_cols].copy()
        X = X.fillna(0)

        print(f"  Selected {len(available_cols)} covariates")

        return X

    def fit_causal_forest(
        self,
        Y: pd.Series,
        T: pd.Series,
        X: pd.DataFrame
    ) -> 'CausalForestDML':
        """
        Fit Causal Forest model

        Args:
            Y: Outcome variable
            T: Treatment variable
            X: Covariates (confounders)

        Returns:
            Fitted CausalForestDML model
        """
        print("\nFitting Causal Forest...")

        if not ECONML_AVAILABLE:
            raise ImportError("econml package is required. Install with: pip install econml")

        cf_model = CausalForestDML(
            model_y=GradientBoostingRegressor(n_estimators=100, random_state=self.cf_config['random_state']),
            model_t=GradientBoostingClassifier(n_estimators=100, random_state=self.cf_config['random_state']),
            n_estimators=self.cf_config['n_estimators'],
            min_samples_leaf=self.cf_config['min_samples_leaf'],
            random_state=self.cf_config['random_state']
        )

        cf_model.fit(Y=Y.values, T=T.values, X=X.values)

        print("  Causal Forest fitted successfully")

        return cf_model

    def compute_cate_scores(
        self,
        user_features: pd.DataFrame,
        labels: pd.Series = None
    ) -> pd.DataFrame:
        """
        Complete pipeline to compute CATE scores

        Args:
            user_features: Feature matrix
            labels: Optional fraud labels

        Returns:
            DataFrame with user_id and cate_score
        """
        print("\n" + "="*80)
        print("CAUSAL FOREST PIPELINE")
        print("="*80)

        # Define variables
        T = self.define_treatment(user_features)
        Y = self.define_outcome(user_features, labels)
        X = self.prepare_covariates(user_features)

        # Fit model
        cf_model = self.fit_causal_forest(Y, T, X)

        # Compute CATE
        print("\nComputing CATE scores...")
        cate_scores = cf_model.effect(X.values)

        # Create output dataframe
        cate_df = pd.DataFrame({
            'user_id': user_features['user_id'].values,
            'cate_score': cate_scores.flatten()
        })

        print(f"  Computed CATE for {len(cate_df)} users")
        print(f"  CATE statistics:")
        print(f"    Mean: {cate_df['cate_score'].mean():.4f}")
        print(f"    Std: {cate_df['cate_score'].std():.4f}")
        print(f"    Min: {cate_df['cate_score'].min():.4f}")
        print(f"    Max: {cate_df['cate_score'].max():.4f}")
        print(f"    Positive CATE: {(cate_df['cate_score'] > 0).sum()} users")

        print(f"\n{'='*80}")
        print("CAUSAL FOREST COMPLETE")
        print(f"{'='*80}\n")

        return cate_df


def main():
    parser = argparse.ArgumentParser(description='BitoGuard Causal Forest')
    parser.add_argument('--config', type=str, default='bitoguard/configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--features', type=str, required=True,
                        help='Path to user_features.csv')
    parser.add_argument('--labels', type=str, default=None,
                        help='Optional path to fraud labels CSV')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for cate_scores.csv')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Load features
    print(f"Loading user features from {args.features}...")
    user_features = pd.read_csv(args.features)

    # Load labels if provided
    labels = None
    if args.labels:
        print(f"Loading fraud labels from {args.labels}...")
        labels_df = pd.read_csv(args.labels)
        labels = labels_df['label']  # Assuming column named 'label'

    # Initialize estimator
    estimator = CausalForestEstimator(config)

    # Compute CATE scores
    cate_df = estimator.compute_cate_scores(user_features, labels)

    # Save output
    output_dir = Path(config['data']['output_dir']) / 'features'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = args.output if args.output else output_dir / config['data']['cate_scores']
    cate_df.to_csv(output_path, index=False)

    print(f"\nCATE scores saved to: {output_path}")


if __name__ == "__main__":
    main()
