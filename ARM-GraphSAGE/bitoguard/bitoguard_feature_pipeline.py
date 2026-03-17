"""
Module 1: Feature Engineering Pipeline
Extracts ~43 dimensional features from raw data tables

Output: user_features.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import argparse
from datetime import datetime

from utils.utils import (
    load_config,
    log1p_transform,
    extract_time_features,
    calculate_ip_features,
    entropy,
    save_pickle
)


class BitoGuardFeatureEngineer:
    """Feature engineering pipeline for BitoGuard fraud detection"""

    def __init__(self, config: Dict):
        self.config = config
        self.data_dir = Path(config['data']['raw_data_dir'])

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all raw data tables"""
        print("Loading raw data tables...")

        data = {}
        for table_name in ['user_info', 'twd_transfer', 'usdt_swap', 'usdt_twd_trading', 'crypto_transfer']:
            file_path = self.data_dir / self.config['data'][table_name]
            if file_path.exists():
                data[table_name] = pd.read_csv(file_path)
                print(f"  Loaded {table_name}: {len(data[table_name])} rows")
            else:
                print(f"  Warning: {file_path} not found, skipping {table_name}")
                data[table_name] = pd.DataFrame()

        return data

    def extract_user_basic_features(self, user_info: pd.DataFrame) -> pd.DataFrame:
        """
        Group A: User basic information features

        Features:
        - age
        - career (encoded)
        - income_source (encoded)
        - kyc_days_to_l1
        - kyc_days_to_l2
        - user_source (encoded)
        """
        print("\nExtracting Group A: User basic features...")

        df = user_info.copy()

        # Convert timestamps
        for col in ['confirmed_at', 'level1_finished_at', 'level2_finished_at']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        features = pd.DataFrame()
        features['user_id'] = df['user_id']

        # Basic attributes
        features['age'] = df['age'].fillna(df['age'].median())
        features['sex'] = df['sex'].fillna(0)
        features['career'] = df['career'].fillna(0)
        features['income_source'] = df['income_source'].fillna(0)
        features['user_source'] = df['user_source'].fillna(0)

        # KYC timing features
        if 'confirmed_at' in df.columns and 'level1_finished_at' in df.columns:
            features['kyc_days_to_l1'] = (
                df['level1_finished_at'] - df['confirmed_at']
            ).dt.total_seconds() / (24 * 3600)
            features['kyc_days_to_l1'] = features['kyc_days_to_l1'].fillna(-1)
        else:
            features['kyc_days_to_l1'] = -1

        if 'level1_finished_at' in df.columns and 'level2_finished_at' in df.columns:
            features['kyc_days_to_l2'] = (
                df['level2_finished_at'] - df['level1_finished_at']
            ).dt.total_seconds() / (24 * 3600)
            features['kyc_days_to_l2'] = features['kyc_days_to_l2'].fillna(-1)
        else:
            features['kyc_days_to_l2'] = -1

        print(f"  Extracted {features.shape[1]-1} basic features for {len(features)} users")

        return features

    def extract_twd_transfer_features(self, twd_transfer: pd.DataFrame) -> pd.DataFrame:
        """
        Group B: TWD transfer behavior features

        Features include transaction counts, amounts, ratios, temporal patterns, IP patterns
        """
        print("\nExtracting Group B: TWD transfer features...")

        if twd_transfer.empty:
            return pd.DataFrame()

        df = twd_transfer.copy()

        # Convert timestamps
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
            time_features = extract_time_features(df['created_at'])
            df = pd.concat([df, time_features], axis=1)

        # Amount transformation
        if 'ori_samount' in df.columns:
            df['amount_log'] = log1p_transform(df['ori_samount'])

        features_list = []

        # Group by user
        for user_id, user_df in df.groupby('user_id'):
            user_features = {'user_id': user_id}

            # Transaction counts
            user_features['twd_tx_count'] = len(user_df)

            if 'kind' in user_df.columns:
                deposit_df = user_df[user_df['kind'] == 'deposit']
                withdraw_df = user_df[user_df['kind'] == 'withdraw']

                user_features['twd_deposit_count'] = len(deposit_df)
                user_features['twd_withdraw_count'] = len(withdraw_df)

                # Amounts
                if 'ori_samount' in user_df.columns:
                    user_features['twd_total_deposit'] = log1p_transform(
                        pd.Series([deposit_df['ori_samount'].sum()])
                    ).iloc[0]
                    user_features['twd_total_withdraw'] = log1p_transform(
                        pd.Series([withdraw_df['ori_samount'].sum()])
                    ).iloc[0]

                    total = user_features['twd_total_deposit'] + user_features['twd_total_withdraw']
                    user_features['twd_withdraw_ratio'] = (
                        user_features['twd_total_withdraw'] / total if total > 0 else 0
                    )
            else:
                user_features['twd_deposit_count'] = 0
                user_features['twd_withdraw_count'] = 0
                user_features['twd_total_deposit'] = 0
                user_features['twd_total_withdraw'] = 0
                user_features['twd_withdraw_ratio'] = 0

            # Temporal patterns
            if 'is_night' in user_df.columns:
                user_features['twd_night_tx_ratio'] = user_df['is_night'].mean()
            else:
                user_features['twd_night_tx_ratio'] = 0

            if 'is_weekend' in user_df.columns:
                user_features['twd_weekend_ratio'] = user_df['is_weekend'].mean()
            else:
                user_features['twd_weekend_ratio'] = 0

            # IP patterns
            if 'source_ip_hash' in user_df.columns:
                user_features['twd_unique_ip_count'] = user_df['source_ip_hash'].nunique()
            else:
                user_features['twd_unique_ip_count'] = 0

            # Amount statistics
            if 'ori_samount' in user_df.columns:
                user_features['twd_avg_amount'] = user_df['ori_samount'].mean()
                user_features['twd_amount_std'] = user_df['ori_samount'].std()
                user_features['twd_amount_std'] = user_features['twd_amount_std'] if not pd.isna(
                    user_features['twd_amount_std']) else 0
            else:
                user_features['twd_avg_amount'] = 0
                user_features['twd_amount_std'] = 0

            features_list.append(user_features)

        features = pd.DataFrame(features_list)

        # Calculate IP reuse rate (multi-user IP sharing)
        if 'source_ip_hash' in df.columns:
            ip_features = calculate_ip_features(df, 'user_id', 'source_ip_hash')
            features = features.merge(ip_features[['user_id', 'ip_reuse_rate']], on='user_id', how='left')
            features['twd_ip_reuse_rate'] = features['ip_reuse_rate'].fillna(0)
            features.drop(columns=['ip_reuse_rate'], inplace=True)
        else:
            features['twd_ip_reuse_rate'] = 0

        print(f"  Extracted {features.shape[1]-1} TWD transfer features for {len(features)} users")

        return features

    def extract_usdt_trading_features(self, usdt_twd_trading: pd.DataFrame) -> pd.DataFrame:
        """
        Group C: USDT trading behavior features
        """
        print("\nExtracting Group C: USDT trading features...")

        if usdt_twd_trading.empty:
            return pd.DataFrame()

        df = usdt_twd_trading.copy()

        # Convert timestamps
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

        # Amount transformation
        if 'amount' in df.columns:
            df['amount_log'] = log1p_transform(df['amount'])

        features_list = []

        for user_id, user_df in df.groupby('user_id'):
            user_features = {'user_id': user_id}

            user_features['trade_count'] = len(user_df)

            # Buy/sell ratio
            if 'side' in user_df.columns:
                buy_count = (user_df['side'] == 'buy').sum()
                user_features['trade_buy_ratio'] = buy_count / len(user_df) if len(user_df) > 0 else 0
            else:
                user_features['trade_buy_ratio'] = 0.5

            # Amount statistics
            if 'amount' in user_df.columns:
                user_features['trade_avg_amount'] = user_df['amount'].mean()
            else:
                user_features['trade_avg_amount'] = 0

            # Market vs limit orders
            if 'order_type' in user_df.columns:
                market_count = (user_df['order_type'] == 'market').sum()
                user_features['trade_market_ratio'] = market_count / len(user_df) if len(user_df) > 0 else 0
            else:
                user_features['trade_market_ratio'] = 0

            # Source entropy
            if 'source' in user_df.columns:
                user_features['trade_source_entropy'] = entropy(user_df['source'])
            else:
                user_features['trade_source_entropy'] = 0

            # Trading speed
            if 'created_at' in user_df.columns:
                time_span_days = (user_df['created_at'].max() - user_df['created_at'].min()).total_seconds() / (
                            24 * 3600)
                user_features['trade_speed_score'] = len(user_df) / time_span_days if time_span_days > 0 else 0
            else:
                user_features['trade_speed_score'] = 0

            features_list.append(user_features)

        features = pd.DataFrame(features_list)

        print(f"  Extracted {features.shape[1]-1} USDT trading features for {len(features)} users")

        return features

    def extract_usdt_swap_features(self, usdt_swap: pd.DataFrame) -> pd.DataFrame:
        """
        Group D: USDT swap behavior features
        """
        print("\nExtracting Group D: USDT swap features...")

        if usdt_swap.empty:
            return pd.DataFrame()

        df = usdt_swap.copy()

        # Convert timestamps
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

        # Amount transformation
        if 'amount' in df.columns:
            df['amount_log'] = log1p_transform(df['amount'])

        features_list = []

        for user_id, user_df in df.groupby('user_id'):
            user_features = {'user_id': user_id}

            user_features['swap_count'] = len(user_df)

            # Amount
            if 'amount' in user_df.columns:
                user_features['swap_total_amount'] = log1p_transform(
                    pd.Series([user_df['amount'].sum()])
                ).iloc[0]
            else:
                user_features['swap_total_amount'] = 0

            # Currency diversity
            if 'kind' in user_df.columns:
                user_features['swap_kind_diversity'] = user_df['kind'].nunique()
            else:
                user_features['swap_kind_diversity'] = 1

            # Average interval
            if 'created_at' in user_df.columns and len(user_df) > 1:
                user_df_sorted = user_df.sort_values('created_at')
                intervals = user_df_sorted['created_at'].diff().dt.total_seconds() / 3600
                user_features['swap_avg_interval_hours'] = intervals.mean()
                user_features['swap_avg_interval_hours'] = user_features['swap_avg_interval_hours'] if not pd.isna(
                    user_features['swap_avg_interval_hours']) else 24
            else:
                user_features['swap_avg_interval_hours'] = 24

            features_list.append(user_features)

        features = pd.DataFrame(features_list)

        print(f"  Extracted {features.shape[1]-1} USDT swap features for {len(features)} users")

        return features

    def build_feature_matrix(self) -> pd.DataFrame:
        """Build complete feature matrix by merging all feature groups"""
        print("\n" + "=" * 80)
        print("BUILDING BITOGUARD FEATURE MATRIX")
        print("=" * 80)

        # Load data
        data = self.load_data()

        # Extract features from each group
        user_basic = self.extract_user_basic_features(data['user_info'])
        twd_features = self.extract_twd_transfer_features(data['twd_transfer'])
        trading_features = self.extract_usdt_trading_features(data['usdt_twd_trading'])
        swap_features = self.extract_usdt_swap_features(data['usdt_swap'])

        # Merge all features
        print("\nMerging all feature groups...")
        feature_matrix = user_basic

        if not twd_features.empty:
            feature_matrix = feature_matrix.merge(twd_features, on='user_id', how='left')

        if not trading_features.empty:
            feature_matrix = feature_matrix.merge(trading_features, on='user_id', how='left')

        if not swap_features.empty:
            feature_matrix = feature_matrix.merge(swap_features, on='user_id', how='left')

        # Fill NaN values with 0
        feature_matrix = feature_matrix.fillna(0)

        print(f"\n{'=' * 80}")
        print(f"FEATURE MATRIX COMPLETE")
        print(f"  Total users: {len(feature_matrix)}")
        print(f"  Total features: {feature_matrix.shape[1] - 1}")  # Exclude user_id
        print(f"{'=' * 80}\n")

        return feature_matrix


def main():
    parser = argparse.ArgumentParser(description='BitoGuard Feature Engineering Pipeline')
    parser.add_argument('--config', type=str, default='bitoguard/configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for user_features.csv')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Initialize feature engineer
    engineer = BitoGuardFeatureEngineer(config)

    # Build feature matrix
    feature_matrix = engineer.build_feature_matrix()

    # Save output
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(config['data']['output_dir']) / 'features' / config['data']['user_features']

    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_matrix.to_csv(output_path, index=False)

    print(f"Feature matrix saved to: {output_path}")
    print(f"Shape: {feature_matrix.shape}")
    print(f"\nFirst few rows:")
    print(feature_matrix.head())


if __name__ == "__main__":
    main()
