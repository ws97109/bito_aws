"""
Module 2: Association Rule Mining (ARM)
Discovers fraud behavior co-occurrence patterns using FP-Growth

Outputs:
  - arm_rules.pkl: Discovered association rules
  - arm_features.csv: ARM-derived features for each user
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

from utils.utils import load_config, save_pickle, load_pickle


class ARMMiner:
    """Association Rule Mining for fraud pattern discovery"""

    def __init__(self, config: Dict):
        self.config = config
        self.arm_config = config['arm']
        self.thresholds = self.arm_config['thresholds']

    def define_behavioral_items(self, user_features: pd.DataFrame) -> pd.DataFrame:
        """
        Convert continuous features into boolean behavioral items

        Args:
            user_features: Feature matrix from Module 1

        Returns:
            Boolean matrix where each column represents a behavioral item
        """
        print("\nDefining behavioral items...")

        df = user_features.copy()
        items = pd.DataFrame()
        items['user_id'] = df['user_id']

        # TWD transfer patterns
        items['HIGH_WITHDRAW_RATIO'] = (
            df['twd_withdraw_ratio'] > self.thresholds['high_withdraw_ratio']
        ).astype(int)

        items['NIGHT_TRANSACTIONS'] = (
            df['twd_night_tx_ratio'] > self.thresholds['night_tx_ratio']
        ).astype(int)

        items['MULTI_IP'] = (
            df['twd_unique_ip_count'] > self.thresholds['multi_ip_count']
        ).astype(int)

        items['IP_SHARED'] = (
            df['twd_ip_reuse_rate'] > self.thresholds['ip_shared_rate']
        ).astype(int)

        # Trading patterns
        items['BUY_ONLY'] = (
            df['trade_buy_ratio'] > self.thresholds['buy_only_ratio']
        ).astype(int)

        items['HIGH_SPEED_TRADE'] = (
            df['trade_speed_score'] > self.thresholds['high_speed_trade']
        ).astype(int)

        items['MARKET_ORDER_HEAVY'] = (
            df['trade_market_ratio'] > self.thresholds['market_order_ratio']
        ).astype(int)

        # Swap patterns
        items['FAST_SWAP'] = (
            df['swap_avg_interval_hours'] < self.thresholds['fast_swap_hours']
        ).astype(int)

        # KYC patterns
        items['QUICK_KYC'] = (
            df['kyc_days_to_l1'] < self.thresholds['quick_kyc_days']
        ).astype(int)

        # Demographics
        items['LOW_AGE'] = (
            df['age'] < self.thresholds['low_age']
        ).astype(int)

        # Amount patterns
        items['HIGH_AMOUNT_SINGLE'] = (
            df['twd_avg_amount'] > self.thresholds['high_amount_single']
        ).astype(int)

        items['FIXED_AMOUNT'] = (
            df['twd_amount_std'] < self.thresholds['fixed_amount_std']
        ).astype(int)

        print(f"  Defined {items.shape[1]-1} behavioral items")

        return items

    def items_to_transactions(self, items: pd.DataFrame) -> List[List[str]]:
        """
        Convert boolean item matrix to transaction format for FP-Growth

        Args:
            items: Boolean item matrix

        Returns:
            List of transactions (each transaction is a list of item names)
        """
        item_cols = [col for col in items.columns if col != 'user_id']

        transactions = []
        for idx, row in items.iterrows():
            transaction = [item for item in item_cols if row[item] == 1]
            if len(transaction) > 0:  # Only include non-empty transactions
                transactions.append(transaction)

        print(f"  Created {len(transactions)} transactions")

        return transactions

    def mine_frequent_itemsets(self, transactions: List[List[str]]) -> pd.DataFrame:
        """
        Use FP-Growth to mine frequent itemsets

        Args:
            transactions: List of transactions

        Returns:
            DataFrame of frequent itemsets
        """
        print("\nMining frequent itemsets with FP-Growth...")

        # Transform to one-hot encoded format
        te = TransactionEncoder()
        te_array = te.fit_transform(transactions)
        df_items = pd.DataFrame(te_array, columns=te.columns_)

        # Apply FP-Growth
        frequent_itemsets = fpgrowth(
            df_items,
            min_support=self.arm_config['min_support'],
            use_colnames=True
        )

        print(f"  Found {len(frequent_itemsets)} frequent itemsets")

        return frequent_itemsets

    def generate_association_rules(self, frequent_itemsets: pd.DataFrame) -> pd.DataFrame:
        """
        Generate association rules from frequent itemsets

        Args:
            frequent_itemsets: Frequent itemsets from FP-Growth

        Returns:
            DataFrame of association rules with metrics
        """
        print("\nGenerating association rules...")

        # Generate rules
        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=self.arm_config['min_confidence']
        )

        # Filter by lift
        rules = rules[rules['lift'] > self.arm_config['min_lift']]

        # Sort by confidence * lift (fraud score)
        rules['fraud_score'] = rules['confidence'] * np.log(rules['lift'])
        rules = rules.sort_values('fraud_score', ascending=False)

        print(f"  Generated {len(rules)} high-quality rules")
        print(f"  (min_confidence={self.arm_config['min_confidence']}, min_lift={self.arm_config['min_lift']})")

        return rules

    def compute_user_arm_features(
        self,
        items: pd.DataFrame,
        rules: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute ARM-derived features for each user

        Features:
          - arm_rule_hit_count: Number of rules matched
          - arm_max_confidence: Maximum confidence of matched rules
          - arm_avg_lift: Average lift of matched rules
          - arm_fraud_score: Weighted fraud risk score
        """
        print("\nComputing ARM features for each user...")

        arm_features = []

        item_cols = [col for col in items.columns if col != 'user_id']

        for idx, row in items.iterrows():
            user_id = row['user_id']
            user_items = set([item for item in item_cols if row[item] == 1])

            matched_rules = []

            # Check which rules this user matches (antecedent subset of user items)
            for _, rule in rules.iterrows():
                antecedent = set(rule['antecedents'])
                if antecedent.issubset(user_items):
                    matched_rules.append(rule)

            # Compute features
            user_feature = {'user_id': user_id}

            if len(matched_rules) > 0:
                user_feature['arm_rule_hit_count'] = len(matched_rules)
                user_feature['arm_max_confidence'] = max(r['confidence'] for r in matched_rules)
                user_feature['arm_avg_lift'] = np.mean([r['lift'] for r in matched_rules])
                user_feature['arm_fraud_score'] = sum(r['fraud_score'] for r in matched_rules)
            else:
                user_feature['arm_rule_hit_count'] = 0
                user_feature['arm_max_confidence'] = 0
                user_feature['arm_avg_lift'] = 0
                user_feature['arm_fraud_score'] = 0

            arm_features.append(user_feature)

        arm_features_df = pd.DataFrame(arm_features)

        print(f"  Computed ARM features for {len(arm_features_df)} users")

        return arm_features_df

    def display_top_rules(self, rules: pd.DataFrame, top_k: int = 10):
        """Display top-k rules for inspection"""
        print(f"\n{'='*80}")
        print(f"TOP {top_k} ASSOCIATION RULES")
        print(f"{'='*80}")

        for i, (idx, rule) in enumerate(rules.head(top_k).iterrows(), 1):
            antecedent = ', '.join(rule['antecedents'])
            consequent = ', '.join(rule['consequents'])

            print(f"\nRule {i}:")
            print(f"  {antecedent} => {consequent}")
            print(f"  Support: {rule['support']:.4f}")
            print(f"  Confidence: {rule['confidence']:.4f}")
            print(f"  Lift: {rule['lift']:.4f}")
            print(f"  Fraud Score: {rule['fraud_score']:.4f}")

        print(f"{'='*80}\n")

    def run_arm_pipeline(self, user_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete ARM pipeline

        Args:
            user_features: Feature matrix from Module 1

        Returns:
            (rules, arm_features)
        """
        print("\n" + "="*80)
        print("ARM MINING PIPELINE")
        print("="*80)

        # Step 1: Define behavioral items
        items = self.define_behavioral_items(user_features)

        # Step 2: Convert to transactions
        transactions = self.items_to_transactions(items)

        # Step 3: Mine frequent itemsets
        frequent_itemsets = self.mine_frequent_itemsets(transactions)

        # Step 4: Generate association rules
        rules = self.generate_association_rules(frequent_itemsets)

        # Step 5: Display top rules
        self.display_top_rules(rules, top_k=10)

        # Step 6: Compute user ARM features
        arm_features = self.compute_user_arm_features(items, rules)

        print(f"{'='*80}")
        print("ARM MINING COMPLETE")
        print(f"  Total rules discovered: {len(rules)}")
        print(f"  Users with ARM features: {len(arm_features)}")
        print(f"{'='*80}\n")

        return rules, arm_features


def compute_arm_edge_weights(
    rules: pd.DataFrame,
    user_items: Dict[int, set]
) -> Dict[Tuple[int, int], float]:
    """
    Compute ARM-based edge weights for graph construction

    If two users share high-confidence rules, their edge weight increases

    Args:
        rules: Association rules
        user_items: Dictionary mapping user_id to set of behavioral items

    Returns:
        Dictionary mapping (user_i, user_j) to edge weight
    """
    print("\nComputing ARM edge weights...")

    edge_weights = {}

    user_ids = list(user_items.keys())

    for i in range(len(user_ids)):
        for j in range(i + 1, len(user_ids)):
            user_i = user_ids[i]
            user_j = user_ids[j]

            items_i = user_items[user_i]
            items_j = user_items[user_j]

            # Find shared rules
            shared_rules = []
            for _, rule in rules.iterrows():
                antecedent = set(rule['antecedents'])
                if antecedent.issubset(items_i) and antecedent.issubset(items_j):
                    shared_rules.append(rule)

            # Compute edge weight
            if len(shared_rules) > 0:
                max_conf = max(r['confidence'] for r in shared_rules)
                max_lift = max(r['lift'] for r in shared_rules)
                weight = 1.0 + max_conf * np.log(max_lift)
            else:
                weight = 1.0

            edge_weights[(user_i, user_j)] = weight
            edge_weights[(user_j, user_i)] = weight

    print(f"  Computed edge weights for {len(edge_weights)} user pairs")

    return edge_weights


def main():
    parser = argparse.ArgumentParser(description='BitoGuard ARM Mining')
    parser.add_argument('--config', type=str, default='bitoguard/configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--features', type=str, required=True,
                        help='Path to user_features.csv from Module 1')
    parser.add_argument('--output_rules', type=str, default=None,
                        help='Output path for arm_rules.pkl')
    parser.add_argument('--output_features', type=str, default=None,
                        help='Output path for arm_features.csv')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Load user features
    print(f"Loading user features from {args.features}...")
    user_features = pd.read_csv(args.features)

    # Initialize ARM miner
    miner = ARMMiner(config)

    # Run ARM pipeline
    rules, arm_features = miner.run_arm_pipeline(user_features)

    # Save outputs
    output_dir = Path(config['data']['output_dir']) / 'features'
    output_dir.mkdir(parents=True, exist_ok=True)

    rules_path = args.output_rules if args.output_rules else output_dir / config['data']['arm_rules']
    features_path = args.output_features if args.output_features else output_dir / config['data']['arm_features']

    save_pickle(rules, rules_path)
    arm_features.to_csv(features_path, index=False)

    print(f"\nOutputs saved:")
    print(f"  Rules: {rules_path}")
    print(f"  Features: {features_path}")


if __name__ == "__main__":
    main()
