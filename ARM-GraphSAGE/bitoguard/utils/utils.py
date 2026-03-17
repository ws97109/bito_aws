"""
BitoGuard Utility Functions
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


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_pickle(obj: Any, file_path: str):
    """Save object as pickle file"""
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Saved pickle to {file_path}")


def load_pickle(file_path: str) -> Any:
    """Load pickle file"""
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    print(f"Loaded pickle from {file_path}")
    return obj


def time_based_split(
    df: pd.DataFrame,
    time_col: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe by time to avoid data leakage

    Args:
        df: Input dataframe
        time_col: Name of timestamp column
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing

    Returns:
        train_df, val_df, test_df
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    df_sorted = df.sort_values(time_col)
    n = len(df_sorted)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df_sorted.iloc[:train_end]
    val_df = df_sorted.iloc[train_end:val_end]
    test_df = df_sorted.iloc[val_end:]

    return train_df, val_df, test_df


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, float]:
    """
    Compute evaluation metrics for fraud detection

    Args:
        y_true: True labels
        y_pred: Predicted labels (binary)
        y_proba: Predicted probabilities (for AUC metrics)

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Basic metrics
    metrics['f1_score'] = f1_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

    # AUC metrics (require probabilities)
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        metrics['pr_auc'] = average_precision_score(y_true, y_proba)

        # Precision@100 (precision of top 100 predictions)
        top_100_idx = np.argsort(y_proba)[-100:]
        if len(top_100_idx) > 0:
            metrics['precision_at_100'] = y_true[top_100_idx].sum() / len(top_100_idx)

        # Recall at 5% FPR
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        idx = np.where(fpr <= 0.05)[0]
        if len(idx) > 0:
            metrics['recall_at_5pct_fpr'] = tpr[idx[-1]]
        else:
            metrics['recall_at_5pct_fpr'] = 0.0

    return metrics


def log1p_transform(values: pd.Series) -> pd.Series:
    """Apply log1p transformation to handle right-skewed distributions"""
    return np.log1p(values.clip(lower=0))


def extract_time_features(timestamps: pd.Series) -> pd.DataFrame:
    """
    Extract time-based features from timestamps

    Args:
        timestamps: Series of datetime objects

    Returns:
        DataFrame with time features
    """
    df_time = pd.DataFrame()

    df_time['hour'] = timestamps.dt.hour
    df_time['day_of_week'] = timestamps.dt.dayofweek
    df_time['is_weekend'] = timestamps.dt.dayofweek.isin([5, 6]).astype(int)
    df_time['is_night'] = timestamps.dt.hour.between(22, 6).astype(int)

    return df_time


def calculate_ip_features(df: pd.DataFrame, user_col: str = 'user_id', ip_col: str = 'source_ip_hash') -> pd.DataFrame:
    """
    Calculate IP-based features for fraud detection

    Args:
        df: Dataframe with user_id and IP hash
        user_col: Name of user ID column
        ip_col: Name of IP hash column

    Returns:
        DataFrame with IP features per user
    """
    # Unique IPs per user
    unique_ips = df.groupby(user_col)[ip_col].nunique().reset_index()
    unique_ips.columns = [user_col, 'unique_ip_count']

    # IP reuse rate (how many users share the same IP)
    ip_user_counts = df.groupby(ip_col)[user_col].nunique()
    df['ip_user_count'] = df[ip_col].map(ip_user_counts)
    ip_reuse = df.groupby(user_col)['ip_user_count'].apply(
        lambda x: (x > 1).sum() / len(x) if len(x) > 0 else 0
    ).reset_index()
    ip_reuse.columns = [user_col, 'ip_reuse_rate']

    # Merge features
    ip_features = unique_ips.merge(ip_reuse, on=user_col, how='left')

    return ip_features


def entropy(series: pd.Series) -> float:
    """Calculate Shannon entropy of a series"""
    value_counts = series.value_counts(normalize=True)
    return -np.sum(value_counts * np.log2(value_counts + 1e-9))


class EarlyStopping:
    """Early stopping callback for training"""

    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'max'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop

        Args:
            score: Current metric value

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False

    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


def print_model_summary(model: torch.nn.Module):
    """Print model architecture and parameter count"""
    print("\n" + "="*80)
    print("MODEL SUMMARY")
    print("="*80)
    print(model)
    print("="*80)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("="*80 + "\n")


def create_output_dirs(config: Dict):
    """Create output directories from config"""
    output_dir = Path(config['data']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (output_dir / 'models').mkdir(exist_ok=True)
    (output_dir / 'features').mkdir(exist_ok=True)
    (output_dir / 'graphs').mkdir(exist_ok=True)
    (output_dir / 'reports').mkdir(exist_ok=True)

    print(f"Created output directories in {output_dir}")
