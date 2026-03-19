"""
Shared utility functions for ARM-GraphSAGE → XGBoost pipeline
"""

import numpy as np
import pandas as pd
import torch
import random
import yaml
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    confusion_matrix
)


# ─────────────────────────────────────────────────────────────────────────────
# Config / IO
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_pickle(obj: Any, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Saved: {file_path}")


def load_pickle(file_path) -> Any:
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    print(f"Loaded: {file_path}")
    return obj


def create_output_dirs(config: Dict):
    output_dir = Path(config['data']['output_dir'])
    for sub in ['models', 'features', 'graphs', 'reports']:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)
    print(f"Output dirs ready under {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Unit Scaling  (× 1e-8)
# ─────────────────────────────────────────────────────────────────────────────

SCALE_FACTOR = 1e-8

# Default column map — overridden by config['unit_scaling'] if present
DEFAULT_SCALE_COLS: Dict[str, List[str]] = {
    'twd_transfer':     ['ori_samount'],
    'crypto_transfer':  ['ori_samount', 'twd_srate'],
    'usdt_twd_trading': ['trade_samount', 'twd_srate'],
    'usdt_swap':        ['twd_samount', 'currency_samount'],
}


def apply_unit_scaling(data: Dict[str, pd.DataFrame], config: Dict) -> Dict[str, pd.DataFrame]:
    """
    Multiply raw integer-stored amounts by 1e-8 to obtain true decimal values.

    Should be called once right after loading raw tables, before any feature
    extraction or graph construction.
    """
    scale_map = config.get('unit_scaling', DEFAULT_SCALE_COLS)

    print("\nApplying 1e-8 unit scaling...")
    for table, cols in scale_map.items():
        if table not in data or data[table].empty:
            continue
        for col in cols:
            if col in data[table].columns:
                data[table][col] = data[table][col] * SCALE_FACTOR
                print(f"  {table}.{col} × {SCALE_FACTOR}")

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Feature Transformations
# ─────────────────────────────────────────────────────────────────────────────

def log1p_transform(values: pd.Series) -> pd.Series:
    return np.log1p(values.clip(lower=0))


def extract_time_features(timestamps: pd.Series) -> pd.DataFrame:
    df_time = pd.DataFrame()
    df_time['hour'] = timestamps.dt.hour
    df_time['day_of_week'] = timestamps.dt.dayofweek
    df_time['is_weekend'] = timestamps.dt.dayofweek.isin([5, 6]).astype(int)
    df_time['is_night'] = timestamps.dt.hour.between(22, 6).astype(int)
    return df_time


def calculate_ip_features(
    df: pd.DataFrame,
    user_col: str = 'user_id',
    ip_col: str = 'source_ip_hash'
) -> pd.DataFrame:
    unique_ips = df.groupby(user_col)[ip_col].nunique().reset_index()
    unique_ips.columns = [user_col, 'unique_ip_count']

    ip_user_counts = df.groupby(ip_col)[user_col].nunique()
    df = df.copy()
    df['ip_user_count'] = df[ip_col].map(ip_user_counts)
    ip_reuse = df.groupby(user_col)['ip_user_count'].apply(
        lambda x: (x > 1).sum() / len(x) if len(x) > 0 else 0
    ).reset_index()
    ip_reuse.columns = [user_col, 'ip_reuse_rate']

    return unique_ips.merge(ip_reuse, on=user_col, how='left')


def entropy(series: pd.Series) -> float:
    vc = series.value_counts(normalize=True)
    return float(-np.sum(vc * np.log2(vc + 1e-9)))


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None
) -> Dict[str, float]:
    metrics = {}
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics['recall']    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        metrics['pr_auc']  = average_precision_score(y_true, y_proba)

        top_100_idx = np.argsort(y_proba)[-100:]
        metrics['precision_at_100'] = y_true[top_100_idx].sum() / len(top_100_idx)

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        idx = np.where(fpr <= 0.05)[0]
        metrics['recall_at_5pct_fpr'] = float(tpr[idx[-1]]) if len(idx) > 0 else 0.0

    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k:<25} {v:.4f}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Training Helpers
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'max'):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.counter   = 0
        self.best_score = None

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        improved = (
            score > self.best_score + self.min_delta if self.mode == 'max'
            else score < self.best_score - self.min_delta
        )
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def reset(self):
        self.counter = 0
        self.best_score = None


def print_model_summary(model: torch.nn.Module):
    print("\n" + "=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    print(model)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters    : {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print("=" * 80 + "\n")
