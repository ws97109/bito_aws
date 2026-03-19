"""
Module 2: Association Rule Mining (ARM)
Discovers fraud behaviour co-occurrence patterns with FP-Growth.

Input : results/features/user_features.csv  (from Module 1, already scaled)
Output:
  results/features/arm_rules.pkl
  results/features/arm_features.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

from utils.utils import load_config, save_pickle, load_pickle, create_output_dirs


class ARMMiner:
    """FP-Growth–based association rule miner for fraud pattern discovery."""

    def __init__(self, config: Dict):
        self.config     = config
        self.arm_cfg    = config['arm']
        self.thresholds = self.arm_cfg['thresholds']

    # ──────────────────────────────────────────────────────────────
    # Step 1: Boolean behavioural items
    # ──────────────────────────────────────────────────────────────

    def define_behavioral_items(self, user_features: pd.DataFrame) -> pd.DataFrame:
        """Convert continuous features to boolean items using domain thresholds."""
        print("\nDefining behavioural items...")

        df = user_features.copy()
        t  = self.thresholds
        items = pd.DataFrame({'user_id': df['user_id']})

        def flag(col, op, threshold, name):
            if col in df.columns:
                if op == '>':
                    items[name] = (df[col] > threshold).astype(int)
                else:
                    items[name] = (df[col] < threshold).astype(int)
            else:
                items[name] = 0

        flag('twd_withdraw_ratio',     '>',  t['high_withdraw_ratio'], 'HIGH_WITHDRAW_RATIO')
        flag('twd_night_tx_ratio',     '>',  t['night_tx_ratio'],      'NIGHT_TRANSACTIONS')
        flag('twd_unique_ip_count',    '>',  t['multi_ip_count'],      'MULTI_IP')
        flag('twd_ip_reuse_rate',      '>',  t['ip_shared_rate'],      'IP_SHARED')
        flag('trade_buy_ratio',        '>',  t['buy_only_ratio'],      'BUY_ONLY')
        flag('trade_speed_score',      '>',  t['high_speed_trade'],    'HIGH_SPEED_TRADE')
        flag('trade_market_ratio',     '>',  t['market_order_ratio'],  'MARKET_ORDER_HEAVY')
        flag('swap_avg_interval_hours','<',  t['fast_swap_hours'],     'FAST_SWAP')
        flag('kyc_days_to_l1',         '<',  t['quick_kyc_days'],      'QUICK_KYC')
        flag('age',                    '<',  t['low_age'],             'LOW_AGE')
        flag('twd_avg_amount',         '>',  t['high_amount_single'],  'HIGH_AMOUNT_SINGLE')
        flag('twd_amount_std',         '<',  t['fixed_amount_std'],    'FIXED_AMOUNT')

        print(f"  {items.shape[1] - 1} behavioural items defined")
        return items

    # ──────────────────────────────────────────────────────────────
    # Step 2 → 4: FP-Growth + rules
    # ──────────────────────────────────────────────────────────────

    def items_to_transactions(self, items: pd.DataFrame) -> List[List[str]]:
        item_cols = [c for c in items.columns if c != 'user_id']
        transactions = [
            [item for item in item_cols if row[item] == 1]
            for _, row in items.iterrows()
        ]
        transactions = [t for t in transactions if t]
        print(f"  {len(transactions):,} non-empty transactions")
        return transactions

    def mine_frequent_itemsets(self, transactions: List[List[str]]) -> pd.DataFrame:
        print("\nMining frequent itemsets (FP-Growth)...")
        te = TransactionEncoder()
        df_items = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)
        freq = fpgrowth(df_items, min_support=self.arm_cfg['min_support'], use_colnames=True)
        print(f"  {len(freq)} frequent itemsets found")
        return freq

    def generate_association_rules(self, freq_itemsets: pd.DataFrame) -> pd.DataFrame:
        print("\nGenerating association rules...")
        rules = association_rules(
            freq_itemsets,
            metric='confidence',
            min_threshold=self.arm_cfg['min_confidence']
        )
        rules = rules[rules['lift'] > self.arm_cfg['min_lift']]
        rules['fraud_score'] = rules['confidence'] * np.log(rules['lift'])
        rules = rules.sort_values('fraud_score', ascending=False).reset_index(drop=True)
        print(f"  {len(rules)} rules (conf≥{self.arm_cfg['min_confidence']}, lift>{self.arm_cfg['min_lift']})")
        return rules

    # ──────────────────────────────────────────────────────────────
    # Step 5: Per-user ARM features
    # ──────────────────────────────────────────────────────────────

    def compute_user_arm_features(
        self,
        items: pd.DataFrame,
        rules: pd.DataFrame
    ) -> pd.DataFrame:
        """
        For each user compute:
          arm_rule_hit_count  — number of matched rules
          arm_max_confidence  — highest confidence among matched rules
          arm_avg_lift        — average lift
          arm_fraud_score     — sum of fraud_score for matched rules
        """
        print("\nComputing per-user ARM features...")
        item_cols = [c for c in items.columns if c != 'user_id']

        rows = []
        for _, row in items.iterrows():
            uid        = row['user_id']
            user_items = {item for item in item_cols if row[item] == 1}

            matched = [
                rule for _, rule in rules.iterrows()
                if set(rule['antecedents']).issubset(user_items)
            ]

            if matched:
                r = {
                    'user_id':           uid,
                    'arm_rule_hit_count': len(matched),
                    'arm_max_confidence': max(m['confidence'] for m in matched),
                    'arm_avg_lift':       float(np.mean([m['lift'] for m in matched])),
                    'arm_fraud_score':    sum(m['fraud_score'] for m in matched),
                }
            else:
                r = {'user_id': uid, 'arm_rule_hit_count': 0,
                     'arm_max_confidence': 0.0, 'arm_avg_lift': 0.0, 'arm_fraud_score': 0.0}
            rows.append(r)

        df = pd.DataFrame(rows)
        print(f"  ARM features computed for {len(df):,} users")
        return df

    # ──────────────────────────────────────────────────────────────
    # Display
    # ──────────────────────────────────────────────────────────────

    def display_top_rules(self, rules: pd.DataFrame, top_k: int = 10):
        print(f"\n{'=' * 80}")
        print(f"TOP {top_k} ASSOCIATION RULES")
        print(f"{'=' * 80}")
        for i, (_, rule) in enumerate(rules.head(top_k).iterrows(), 1):
            ant = ', '.join(rule['antecedents'])
            con = ', '.join(rule['consequents'])
            print(f"\nRule {i}: {ant} => {con}")
            print(f"  support={rule['support']:.4f}  confidence={rule['confidence']:.4f}  "
                  f"lift={rule['lift']:.4f}  fraud_score={rule['fraud_score']:.4f}")
        print(f"{'=' * 80}\n")

    # ──────────────────────────────────────────────────────────────
    # Full pipeline
    # ──────────────────────────────────────────────────────────────

    def run(self, user_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print("\n" + "=" * 80)
        print("MODULE 2: ARM MINING PIPELINE")
        print("=" * 80)

        items        = self.define_behavioral_items(user_features)
        transactions = self.items_to_transactions(items)
        freq         = self.mine_frequent_itemsets(transactions)
        rules        = self.generate_association_rules(freq)
        self.display_top_rules(rules)
        arm_features = self.compute_user_arm_features(items, rules)

        print(f"\nARM COMPLETE  rules={len(rules)}  users={len(arm_features):,}")
        return rules, arm_features


def compute_arm_edge_weights(
    rules: pd.DataFrame,
    user_items: Dict[int, set]
) -> Dict[Tuple[int, int], float]:
    """
    Compute ARM-based edge weights for heterogeneous graph construction.
    Two users sharing high-confidence antecedents get a higher edge weight.
    """
    print("\nComputing ARM edge weights for graph...")
    user_ids    = list(user_items.keys())
    edge_weights: Dict[Tuple[int, int], float] = {}

    for i in range(len(user_ids)):
        for j in range(i + 1, len(user_ids)):
            ui, uj   = user_ids[i], user_ids[j]
            items_i  = user_items[ui]
            items_j  = user_items[uj]

            shared = [
                rule for _, rule in rules.iterrows()
                if set(rule['antecedents']).issubset(items_i)
                and set(rule['antecedents']).issubset(items_j)
            ]

            if shared:
                max_conf = max(r['confidence'] for r in shared)
                max_lift = max(r['lift'] for r in shared)
                w = 1.0 + max_conf * np.log(max_lift)
            else:
                w = 1.0

            edge_weights[(ui, uj)] = w
            edge_weights[(uj, ui)] = w

    print(f"  Edge weights computed for {len(edge_weights)} pairs")
    return edge_weights


def main():
    parser = argparse.ArgumentParser(description='Module 2 — ARM Mining')
    parser.add_argument('--config',   default='arm_xgboost/configs/config.yaml')
    parser.add_argument('--features', required=True, help='user_features.csv from Module 1')
    parser.add_argument('--out_rules',    default=None)
    parser.add_argument('--out_features', default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    create_output_dirs(config)

    user_features = pd.read_csv(args.features)
    print(f"Loaded {len(user_features):,} users from {args.features}")

    miner = ARMMiner(config)
    rules, arm_features = miner.run(user_features)

    out_dir = Path(config['data']['output_dir']) / 'features'
    out_dir.mkdir(parents=True, exist_ok=True)

    rules_path    = args.out_rules    or out_dir / config['data']['arm_rules']
    features_path = args.out_features or out_dir / config['data']['arm_features']

    save_pickle(rules, rules_path)
    arm_features.to_csv(features_path, index=False)

    print(f"\nSaved:\n  Rules   → {rules_path}\n  Features → {features_path}")


if __name__ == '__main__':
    main()
