"""
Module 1: Feature Engineering Pipeline
Extracts ~43-dimensional tabular features from raw data tables.

Key change vs original:
  - apply_unit_scaling() is called immediately after loading raw tables,
    converting integer-stored amounts to true decimal values (× 1e-8).
  - Column names updated to match actual raw data schema:
      usdt_twd_trading: trade_samount, twd_srate
      usdt_swap       : twd_samount, currency_samount

Output: results/features/user_features.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import argparse

from utils.utils import (
    load_config,
    apply_unit_scaling,
    log1p_transform,
    extract_time_features,
    calculate_ip_features,
    entropy,
    save_pickle,
    create_output_dirs,
)


class FeatureEngineer:
    """Build the tabular feature matrix from raw transaction tables."""

    def __init__(self, config: Dict):
        self.config   = config
        self.data_dir = Path(config['data']['raw_data_dir'])

    # ──────────────────────────────────────────────────────────────
    # Data Loading
    # ──────────────────────────────────────────────────────────────

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all raw tables and immediately apply 1e-8 unit scaling."""
        print("\nLoading raw data tables...")

        tables = ['user_info', 'twd_transfer', 'usdt_swap', 'usdt_twd_trading', 'crypto_transfer']
        data: Dict[str, pd.DataFrame] = {}

        for name in tables:
            path = self.data_dir / self.config['data'][name]
            if path.exists():
                data[name] = pd.read_csv(path)
                print(f"  {name}: {len(data[name]):,} rows")
            else:
                print(f"  WARNING: {path} not found — skipping {name}")
                data[name] = pd.DataFrame()

        # ← Apply 1e-8 scaling BEFORE any feature extraction
        data = apply_unit_scaling(data, self.config)

        return data

    # ──────────────────────────────────────────────────────────────
    # Group A: User basics
    # ──────────────────────────────────────────────────────────────

    def extract_user_basic_features(self, user_info: pd.DataFrame) -> pd.DataFrame:
        """Age, sex, career, income_source, user_source, KYC timing."""
        print("\n[Group A] User basic features...")

        df = user_info.copy()
        for col in ['confirmed_at', 'level1_finished_at', 'level2_finished_at']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        feat = pd.DataFrame()
        feat['user_id']       = df['user_id']
        feat['age']           = df.get('age', pd.Series(dtype=float)).fillna(df['age'].median() if 'age' in df else 0)
        feat['sex']           = df.get('sex', pd.Series(0, index=df.index)).fillna(0)
        feat['career']        = df.get('career', pd.Series(0, index=df.index)).fillna(0)
        feat['income_source'] = df.get('income_source', pd.Series(0, index=df.index)).fillna(0)
        feat['user_source']   = df.get('user_source', pd.Series(0, index=df.index)).fillna(0)

        if 'confirmed_at' in df.columns and 'level1_finished_at' in df.columns:
            feat['kyc_days_to_l1'] = (
                df['level1_finished_at'] - df['confirmed_at']
            ).dt.total_seconds() / 86400
            feat['kyc_days_to_l1'] = feat['kyc_days_to_l1'].fillna(-1)
        else:
            feat['kyc_days_to_l1'] = -1

        if 'level1_finished_at' in df.columns and 'level2_finished_at' in df.columns:
            feat['kyc_days_to_l2'] = (
                df['level2_finished_at'] - df['level1_finished_at']
            ).dt.total_seconds() / 86400
            feat['kyc_days_to_l2'] = feat['kyc_days_to_l2'].fillna(-1)
        else:
            feat['kyc_days_to_l2'] = -1

        print(f"  → {feat.shape[1] - 1} features for {len(feat):,} users")
        return feat

    # ──────────────────────────────────────────────────────────────
    # Group B: TWD transfers
    # ──────────────────────────────────────────────────────────────

    def extract_twd_transfer_features(self, twd_transfer: pd.DataFrame) -> pd.DataFrame:
        """
        Transaction counts, deposit/withdraw amounts & ratios,
        temporal patterns, IP patterns.

        Amount column: ori_samount (already scaled ×1e-8 → actual TWD).
        """
        print("\n[Group B] TWD transfer features...")

        if twd_transfer.empty:
            return pd.DataFrame()

        df = twd_transfer.copy()

        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
            df = pd.concat([df, extract_time_features(df['created_at'])], axis=1)

        amount_col = 'ori_samount' if 'ori_samount' in df.columns else 'amount'

        rows = []
        for uid, udf in df.groupby('user_id'):
            r = {'user_id': uid}
            r['twd_tx_count'] = len(udf)

            if 'kind' in udf.columns:
                dep = udf[udf['kind'] == 'deposit']
                wit = udf[udf['kind'] == 'withdraw']
                r['twd_deposit_count']  = len(dep)
                r['twd_withdraw_count'] = len(wit)

                if amount_col in udf.columns:
                    dep_sum = dep[amount_col].sum()
                    wit_sum = wit[amount_col].sum()
                    r['twd_total_deposit']  = float(log1p_transform(pd.Series([dep_sum])).iloc[0])
                    r['twd_total_withdraw'] = float(log1p_transform(pd.Series([wit_sum])).iloc[0])
                    total = r['twd_total_deposit'] + r['twd_total_withdraw']
                    r['twd_withdraw_ratio'] = r['twd_total_withdraw'] / total if total > 0 else 0.0
                else:
                    r['twd_total_deposit'] = r['twd_total_withdraw'] = r['twd_withdraw_ratio'] = 0.0
            else:
                r['twd_deposit_count'] = r['twd_withdraw_count'] = 0
                r['twd_total_deposit'] = r['twd_total_withdraw'] = r['twd_withdraw_ratio'] = 0.0

            r['twd_night_tx_ratio'] = float(udf['is_night'].mean())   if 'is_night'   in udf.columns else 0.0
            r['twd_weekend_ratio']  = float(udf['is_weekend'].mean()) if 'is_weekend' in udf.columns else 0.0

            if 'source_ip_hash' in udf.columns:
                r['twd_unique_ip_count'] = udf['source_ip_hash'].nunique()
            else:
                r['twd_unique_ip_count'] = 0

            if amount_col in udf.columns:
                r['twd_avg_amount'] = float(udf[amount_col].mean())
                std = udf[amount_col].std()
                r['twd_amount_std'] = float(std) if not pd.isna(std) else 0.0
            else:
                r['twd_avg_amount'] = r['twd_amount_std'] = 0.0

            rows.append(r)

        feat = pd.DataFrame(rows)

        if 'source_ip_hash' in df.columns:
            ip_feat = calculate_ip_features(df, 'user_id', 'source_ip_hash')
            feat = feat.merge(ip_feat[['user_id', 'ip_reuse_rate']], on='user_id', how='left')
            feat.rename(columns={'ip_reuse_rate': 'twd_ip_reuse_rate'}, inplace=True)
            feat['twd_ip_reuse_rate'] = feat['twd_ip_reuse_rate'].fillna(0)
        else:
            feat['twd_ip_reuse_rate'] = 0.0

        print(f"  → {feat.shape[1] - 1} features for {len(feat):,} users")
        return feat

    # ──────────────────────────────────────────────────────────────
    # Group C: USDT trading
    # ──────────────────────────────────────────────────────────────

    def extract_usdt_trading_features(self, usdt_twd_trading: pd.DataFrame) -> pd.DataFrame:
        """
        Buy/sell ratio, order type distribution, trading speed.

        Amount column: trade_samount (already scaled ×1e-8 → USDT).
        Rate column  : twd_srate    (already scaled ×1e-8 → TWD/USDT).
        """
        print("\n[Group C] USDT trading features...")

        if usdt_twd_trading.empty:
            return pd.DataFrame()

        df = usdt_twd_trading.copy()

        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

        # Prefer trade_samount; fall back to generic 'amount'
        amount_col = 'trade_samount' if 'trade_samount' in df.columns else 'amount'

        rows = []
        for uid, udf in df.groupby('user_id'):
            r = {'user_id': uid}
            r['trade_count'] = len(udf)

            if 'side' in udf.columns:
                r['trade_buy_ratio'] = float((udf['side'] == 'buy').sum()) / len(udf)
            else:
                r['trade_buy_ratio'] = 0.5

            if amount_col in udf.columns:
                r['trade_avg_amount']   = float(udf[amount_col].mean())
                r['trade_total_amount'] = float(log1p_transform(pd.Series([udf[amount_col].sum()])).iloc[0])
            else:
                r['trade_avg_amount'] = r['trade_total_amount'] = 0.0

            # TWD/USDT rate volatility
            if 'twd_srate' in udf.columns:
                r['trade_rate_std'] = float(udf['twd_srate'].std()) if len(udf) > 1 else 0.0
            else:
                r['trade_rate_std'] = 0.0

            if 'order_type' in udf.columns:
                r['trade_market_ratio'] = float((udf['order_type'] == 'market').sum()) / len(udf)
            else:
                r['trade_market_ratio'] = 0.0

            if 'source' in udf.columns:
                r['trade_source_entropy'] = entropy(udf['source'])
            else:
                r['trade_source_entropy'] = 0.0

            if 'created_at' in udf.columns:
                span_days = (udf['created_at'].max() - udf['created_at'].min()).total_seconds() / 86400
                r['trade_speed_score'] = len(udf) / span_days if span_days > 0 else 0.0
            else:
                r['trade_speed_score'] = 0.0

            rows.append(r)

        feat = pd.DataFrame(rows)
        print(f"  → {feat.shape[1] - 1} features for {len(feat):,} users")
        return feat

    # ──────────────────────────────────────────────────────────────
    # Group D: USDT swaps
    # ──────────────────────────────────────────────────────────────

    def extract_usdt_swap_features(self, usdt_swap: pd.DataFrame) -> pd.DataFrame:
        """
        Swap frequency, TWD & crypto amounts, currency diversity, swap intervals.

        Amount columns: twd_samount, currency_samount (already scaled ×1e-8).
        """
        print("\n[Group D] USDT swap features...")

        if usdt_swap.empty:
            return pd.DataFrame()

        df = usdt_swap.copy()

        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

        # Prefer named columns; fall back to generic 'amount'
        twd_col      = 'twd_samount'      if 'twd_samount'      in df.columns else 'amount'
        crypto_col   = 'currency_samount' if 'currency_samount' in df.columns else None

        rows = []
        for uid, udf in df.groupby('user_id'):
            r = {'user_id': uid}
            r['swap_count'] = len(udf)

            if twd_col in udf.columns:
                r['swap_twd_total']  = float(log1p_transform(pd.Series([udf[twd_col].sum()])).iloc[0])
                r['swap_twd_avg']    = float(udf[twd_col].mean())
            else:
                r['swap_twd_total'] = r['swap_twd_avg'] = 0.0

            if crypto_col and crypto_col in udf.columns:
                r['swap_crypto_total'] = float(log1p_transform(pd.Series([udf[crypto_col].sum()])).iloc[0])
                r['swap_crypto_avg']   = float(udf[crypto_col].mean())
            else:
                r['swap_crypto_total'] = r['swap_crypto_avg'] = 0.0

            if 'kind' in udf.columns:
                r['swap_kind_diversity'] = udf['kind'].nunique()
            else:
                r['swap_kind_diversity'] = 1

            if 'created_at' in udf.columns and len(udf) > 1:
                intervals = udf.sort_values('created_at')['created_at'].diff().dt.total_seconds() / 3600
                avg_interval = intervals.mean()
                r['swap_avg_interval_hours'] = float(avg_interval) if not pd.isna(avg_interval) else 24.0
            else:
                r['swap_avg_interval_hours'] = 24.0

            rows.append(r)

        feat = pd.DataFrame(rows)
        print(f"  → {feat.shape[1] - 1} features for {len(feat):,} users")
        return feat

    # ──────────────────────────────────────────────────────────────
    # Main pipeline
    # ──────────────────────────────────────────────────────────────

    def build_feature_matrix(self) -> pd.DataFrame:
        print("\n" + "=" * 80)
        print("MODULE 1: FEATURE ENGINEERING PIPELINE")
        print("=" * 80)

        data = self.load_data()

        basic   = self.extract_user_basic_features(data['user_info'])
        twd     = self.extract_twd_transfer_features(data['twd_transfer'])
        trading = self.extract_usdt_trading_features(data['usdt_twd_trading'])
        swap    = self.extract_usdt_swap_features(data['usdt_swap'])

        print("\nMerging all feature groups...")
        matrix = basic
        for feat_df in [twd, trading, swap]:
            if not feat_df.empty:
                matrix = matrix.merge(feat_df, on='user_id', how='left')

        matrix = matrix.fillna(0)

        print(f"\n{'=' * 80}")
        print(f"FEATURE MATRIX COMPLETE")
        print(f"  Users   : {len(matrix):,}")
        print(f"  Features: {matrix.shape[1] - 1}")
        print(f"{'=' * 80}\n")

        return matrix


def main():
    parser = argparse.ArgumentParser(description='Module 1 — Feature Engineering')
    parser.add_argument('--config', default='arm_xgboost/configs/config.yaml')
    parser.add_argument('--output', default=None, help='Override output CSV path')
    args = parser.parse_args()

    config = load_config(args.config)
    create_output_dirs(config)

    engineer = FeatureEngineer(config)
    matrix   = engineer.build_feature_matrix()

    output_path = (
        Path(args.output) if args.output
        else Path(config['data']['output_dir']) / 'features' / config['data']['user_features']
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_path, index=False)

    print(f"Saved → {output_path}  shape={matrix.shape}")


if __name__ == '__main__':
    main()
