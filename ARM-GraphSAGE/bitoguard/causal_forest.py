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

        # Try multiple strategies to define treatment
        treatment = None
        strategy_used = None

        # Strategy 1: Use trade_speed_score
        if 'trade_speed_score' in df.columns:
            trade_speed = df['trade_speed_score']
            if trade_speed.std() > 0:  # Check if there's variance
                treatment_threshold = self.cf_config['treatment_threshold']
                historical_mean = trade_speed.mean()
                treatment = (trade_speed > treatment_threshold * historical_mean).astype(int)
                strategy_used = "trade_speed_score"

        # Strategy 2: Use trade_buy_ratio
        if treatment is None and 'trade_buy_ratio' in df.columns:
            trade_buy_ratio = df['trade_buy_ratio']
            if trade_buy_ratio.std() > 0:
                median_ratio = trade_buy_ratio.median()
                treatment = (trade_buy_ratio > median_ratio).astype(int)
                strategy_used = "trade_buy_ratio (median split)"

        # Strategy 3: Use twd_tx_amount (high transaction amount)
        if treatment is None and 'twd_tx_amount' in df.columns:
            twd_tx_amount = df['twd_tx_amount']
            if twd_tx_amount.std() > 0:
                percentile_75 = twd_tx_amount.quantile(0.75)
                treatment = (twd_tx_amount > percentile_75).astype(int)
                strategy_used = "twd_tx_amount (75th percentile)"

        # Strategy 4: Use any numeric column with variance
        if treatment is None:
            for col in df.select_dtypes(include=[np.number]).columns:
                if col != 'user_id' and df[col].std() > 0:
                    median_val = df[col].median()
                    treatment = (df[col] > median_val).astype(int)
                    strategy_used = f"{col} (median split)"
                    break

        # Fallback: balanced random treatment
        if treatment is None:
            print("  Warning: No suitable treatment variable found, using balanced random treatment")
            treatment = np.random.binomial(1, 0.5, size=len(df))
            strategy_used = "random (50/50)"

        # Validate treatment has both classes
        unique_treatments = np.unique(treatment)
        if len(unique_treatments) < 2:
            print(f"  Warning: Treatment has only {len(unique_treatments)} class(es). Using balanced random treatment instead.")
            treatment = np.random.binomial(1, 0.5, size=len(df))
            strategy_used = "random (50/50) - fallback"

        print(f"  Treatment strategy: {strategy_used}")
        print(f"  Treatment: {treatment.sum()} / {len(treatment)} users ({treatment.mean():.2%})")
        print(f"  Treatment variance: {treatment.std():.4f}")

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
            print(f"  Outcome: {outcome.sum()} / {len(outcome)} positive ({outcome.mean():.2%})")
        else:
            # Create anomaly score from suspicious patterns
            df = user_features.copy()

            # Try multiple strategies to create a meaningful outcome
            outcome = None
            strategy_used = None

            # Strategy 1: Anomaly score (relaxed threshold)
            anomaly_score = 0
            if 'twd_withdraw_ratio' in df.columns:
                anomaly_score += (df['twd_withdraw_ratio'] > 0.7).astype(int)
            if 'twd_night_tx_ratio' in df.columns:
                anomaly_score += (df['twd_night_tx_ratio'] > 0.3).astype(int)
            if 'trade_buy_ratio' in df.columns:
                anomaly_score += (df['trade_buy_ratio'] > 0.8).astype(int)
            if 'twd_unique_ip_count' in df.columns:
                anomaly_score += (df['twd_unique_ip_count'] > 3).astype(int)

            # Use threshold >= 1 instead of >= 2 for more variance
            if isinstance(anomaly_score, (pd.Series, np.ndarray)) and len(np.unique(anomaly_score >= 1)) > 1:
                outcome = (anomaly_score >= 1).astype(int)
                strategy_used = "anomaly_score >= 1"

            # Strategy 2: High activity users (top 30%)
            if outcome is None and 'twd_tx_count' in df.columns:
                twd_tx = df['twd_tx_count']
                if twd_tx.std() > 0:
                    threshold = twd_tx.quantile(0.70)
                    outcome = (twd_tx > threshold).astype(int)
                    strategy_used = "twd_tx_count > 70th percentile"

            # Strategy 3: High withdrawal ratio (top 40%)
            if outcome is None and 'twd_withdraw_ratio' in df.columns:
                withdraw_ratio = df['twd_withdraw_ratio']
                if withdraw_ratio.std() > 0:
                    threshold = withdraw_ratio.quantile(0.60)
                    outcome = (withdraw_ratio > threshold).astype(int)
                    strategy_used = "twd_withdraw_ratio > 60th percentile"

            # Strategy 4: Any numeric feature with variance (median split)
            if outcome is None:
                for col in df.select_dtypes(include=[np.number]).columns:
                    if col != 'user_id' and df[col].std() > 0:
                        median_val = df[col].median()
                        outcome = (df[col] > median_val).astype(int)
                        strategy_used = f"{col} > median"
                        break

            # Fallback: balanced random outcome
            if outcome is None:
                print("  Warning: No suitable outcome variable found, using balanced random outcome")
                outcome = np.random.binomial(1, 0.3, size=len(df))
                strategy_used = "random (30% positive)"

            # Validate outcome has both classes
            unique_outcomes = np.unique(outcome)
            if len(unique_outcomes) < 2:
                print(f"  Warning: Outcome has only {len(unique_outcomes)} class(es). Using balanced random outcome.")
                outcome = np.random.binomial(1, 0.3, size=len(df))
                strategy_used = "random (30% positive) - fallback"

            print(f"  Outcome strategy: {strategy_used}")
            print(f"  Outcome: {outcome.sum()} / {len(outcome)} positive ({outcome.mean():.2%})")
            print(f"  Outcome variance: {outcome.std():.4f}")

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

        # Validate inputs
        print(f"  Input validation:")
        print(f"    Y shape: {Y.shape}, unique values: {len(np.unique(Y))}")
        print(f"    T shape: {T.shape}, unique values: {len(np.unique(T))}")
        print(f"    X shape: {X.shape}")

        # Check for sufficient variance
        if len(np.unique(Y)) < 2:
            raise ValueError(f"Outcome Y has only {len(np.unique(Y))} unique value(s). Need at least 2.")
        if len(np.unique(T)) < 2:
            raise ValueError(f"Treatment T has only {len(np.unique(T))} unique value(s). Need at least 2.")

        try:
            # For binary treatment, CausalForestDML expects model_t to be a Regressor
            # even though T is binary. This is because DML uses continuous residuals.
            cf_model = CausalForestDML(
                model_y=GradientBoostingRegressor(n_estimators=100, random_state=self.cf_config['random_state']),
                model_t=GradientBoostingRegressor(n_estimators=100, random_state=self.cf_config['random_state']),  # Use Regressor for binary T
                n_estimators=self.cf_config['n_estimators'],
                min_samples_leaf=self.cf_config['min_samples_leaf'],
                random_state=self.cf_config['random_state']
            )

            print("  Fitting DML model (this may take a few minutes)...")
            cf_model.fit(Y=Y.values, T=T.values, X=X.values)

            print("  Causal Forest fitted successfully")

            return cf_model

        except Exception as e:
            print(f"\n  Error during Causal Forest fitting:")
            print(f"  {type(e).__name__}: {str(e)}")
            raise

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
