"""
Module 3: Causal Forest — CATE Score Estimation
Estimates the Conditional Average Treatment Effect of abnormal trading activity.

Input : results/features/user_features.csv  (from Module 1)
Output: results/features/cate_scores.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import argparse

try:
    from econml.dml import CausalForestDML
    ECONML_AVAILABLE = True
except ImportError:
    print("WARNING: econml not installed.  pip install econml")
    ECONML_AVAILABLE = False

from sklearn.ensemble import GradientBoostingRegressor
from utils.utils import load_config, create_output_dirs


class CausalForestEstimator:
    """Estimate heterogeneous treatment effects via CausalForestDML."""

    def __init__(self, config: Dict):
        self.config  = config
        self.cf_cfg  = config['causal_forest']

    # ──────────────────────────────────────────────────────────────
    # Variable definitions
    # ──────────────────────────────────────────────────────────────

    def define_treatment(self, user_features: pd.DataFrame) -> pd.Series:
        """
        T = 1 if recent USDT trading speed > threshold × mean (proxy for surge).
        Falls back through progressively simpler strategies.
        """
        print("\nDefining treatment variable...")
        df        = user_features.copy()
        threshold = self.cf_cfg['treatment_threshold']

        strategies = [
            ('trade_speed_score',  lambda s: s > threshold * s.mean()),
            ('trade_buy_ratio',    lambda s: s > s.median()),
            ('twd_tx_count',       lambda s: s > s.quantile(0.75)),
        ]

        for col, fn in strategies:
            if col in df.columns and df[col].std() > 0:
                T = fn(df[col]).astype(int)
                if T.nunique() == 2:
                    print(f"  Treatment strategy: {col}")
                    print(f"  Treated: {T.sum():,} / {len(T):,} ({T.mean():.1%})")
                    return pd.Series(T.values, index=df.index)

        print("  WARNING: Using random balanced treatment as fallback")
        T = np.random.binomial(1, 0.5, len(df))
        return pd.Series(T, index=df.index)

    def define_outcome(
        self,
        user_features: pd.DataFrame,
        labels: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Y = fraud label if available; otherwise a composite anomaly score.
        """
        print("\nDefining outcome variable...")
        if labels is not None:
            print(f"  Using provided fraud labels — positive rate: {labels.mean():.1%}")
            return labels

        df    = user_features.copy()
        score = pd.Series(0, index=df.index)
        for col, thr in [
            ('twd_withdraw_ratio',  0.7),
            ('twd_night_tx_ratio',  0.3),
            ('trade_buy_ratio',     0.8),
            ('twd_unique_ip_count', 3),
        ]:
            if col in df.columns:
                score += (df[col] > thr).astype(int)

        Y = (score >= 1).astype(int)
        if Y.nunique() < 2:
            Y = (df[df.select_dtypes(include=[np.number]).columns[1]] > df[df.select_dtypes(include=[np.number]).columns[1]].median()).astype(int)

        print(f"  Anomaly-based outcome — positive rate: {Y.mean():.1%}")
        return pd.Series(Y.values, index=df.index)

    def prepare_covariates(self, user_features: pd.DataFrame) -> pd.DataFrame:
        """Baseline features that control for confounding."""
        baseline_cols = [
            'age', 'sex', 'career', 'income_source', 'user_source',
            'kyc_days_to_l1', 'kyc_days_to_l2',
            'twd_tx_count', 'twd_deposit_count',
        ]
        cols = [c for c in baseline_cols if c in user_features.columns]
        X    = user_features[cols].fillna(0)
        print(f"\nCovariates: {len(cols)} features selected")
        return X

    # ──────────────────────────────────────────────────────────────
    # Causal Forest fitting
    # ──────────────────────────────────────────────────────────────

    def fit_and_score(
        self,
        Y: pd.Series,
        T: pd.Series,
        X: pd.DataFrame
    ) -> np.ndarray:
        if not ECONML_AVAILABLE:
            raise ImportError("econml required — pip install econml")

        print("\nFitting CausalForestDML...")
        print(f"  Y shape={Y.shape}  T shape={T.shape}  X shape={X.shape}")

        model = CausalForestDML(
            model_y=GradientBoostingRegressor(
                n_estimators=100, random_state=self.cf_cfg['random_state']
            ),
            model_t=GradientBoostingRegressor(
                n_estimators=100, random_state=self.cf_cfg['random_state']
            ),
            n_estimators=self.cf_cfg['n_estimators'],
            min_samples_leaf=self.cf_cfg['min_samples_leaf'],
            random_state=self.cf_cfg['random_state'],
        )
        model.fit(Y=Y.values, T=T.values, X=X.values)
        cate = model.effect(X.values)
        return cate.flatten()

    # ──────────────────────────────────────────────────────────────
    # Full pipeline
    # ──────────────────────────────────────────────────────────────

    def compute_cate_scores(
        self,
        user_features: pd.DataFrame,
        labels: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        print("\n" + "=" * 80)
        print("MODULE 3: CAUSAL FOREST PIPELINE")
        print("=" * 80)

        T    = self.define_treatment(user_features)
        Y    = self.define_outcome(user_features, labels)
        X    = self.prepare_covariates(user_features)
        cate = self.fit_and_score(Y, T, X)

        result = pd.DataFrame({
            'user_id':    user_features['user_id'].values,
            'cate_score': cate,
        })

        print(f"\nCATE statistics:")
        print(f"  mean={result['cate_score'].mean():.4f}  "
              f"std={result['cate_score'].std():.4f}  "
              f"min={result['cate_score'].min():.4f}  "
              f"max={result['cate_score'].max():.4f}")
        print(f"  Positive CATE: {(result['cate_score'] > 0).sum():,} users")

        print(f"\nCAUSAL FOREST COMPLETE  users={len(result):,}")
        return result


def main():
    parser = argparse.ArgumentParser(description='Module 3 — Causal Forest')
    parser.add_argument('--config',   default='arm_xgboost/configs/config.yaml')
    parser.add_argument('--features', required=True, help='user_features.csv from Module 1')
    parser.add_argument('--labels',   default=None,  help='Optional fraud labels CSV (must have user_id, label columns)')
    parser.add_argument('--output',   default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    create_output_dirs(config)

    user_features = pd.read_csv(args.features)

    labels = None
    if args.labels:
        ldf    = pd.read_csv(args.labels)
        labels = ldf['label']

    estimator = CausalForestEstimator(config)
    cate_df   = estimator.compute_cate_scores(user_features, labels)

    out_dir = Path(config['data']['output_dir']) / 'features'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output or out_dir / config['data']['cate_scores']
    cate_df.to_csv(out_path, index=False)

    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
