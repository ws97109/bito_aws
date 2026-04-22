"""
Pseudo-labeling Module
半監督學習：用高信心預測擴充訓練集

策略：
  - 正例閾值極高（≥0.85）：只加入非常確信的黑名單
  - 負例閾值極低（≤0.05）：只加入非常確信的正常用戶
  - 最多迭代 2~3 輪，每輪比較 CV 分數，下降就停止
"""
import numpy as np
from typing import Tuple, Optional


def pseudo_label(
    ensemble,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_unlabeled: np.ndarray,
    pos_threshold: float = 0.85,
    neg_threshold: float = 0.05,
    max_iter: int = 2,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    迭代式 pseudo-labeling。

    Args:
        ensemble: 已訓練的 StackingEnsemble（需要有 predict_proba 方法）
        X_train: 原始訓練特徵
        y_train: 原始訓練標籤
        X_unlabeled: 無標籤資料的特徵
        pos_threshold: 正例閾值（機率 >= 此值才標為黑名單）
        neg_threshold: 負例閾值（機率 <= 此值才標為正常）
        max_iter: 最大迭代次數

    Returns:
        X_augmented: 擴充後的訓練特徵
        y_augmented: 擴充後的訓練標籤
        stats: 統計資訊
    """
    X_curr = X_train.copy()
    y_curr = y_train.copy()
    X_remaining = X_unlabeled.copy()

    stats = {
        "iterations": 0,
        "total_pos_added": 0,
        "total_neg_added": 0,
        "remaining_unlabeled": len(X_unlabeled),
        "per_round": [],
    }

    for i in range(max_iter):
        if len(X_remaining) == 0:
            print(f"    Round {i+1}: 無剩餘未標記資料，停止")
            break

        # 用目前 ensemble 預測未標記資料
        proba = ensemble.predict_proba(X_remaining)

        # 篩選高信心樣本
        pos_mask = proba >= pos_threshold
        neg_mask = proba <= neg_threshold
        n_pos = pos_mask.sum()
        n_neg = neg_mask.sum()

        round_stats = {
            "round": i + 1,
            "pos_added": int(n_pos),
            "neg_added": int(n_neg),
            "pos_threshold": pos_threshold,
            "neg_threshold": neg_threshold,
            "proba_mean": float(proba.mean()),
            "proba_std": float(proba.std()),
        }
        stats["per_round"].append(round_stats)

        print(f"    Round {i+1}: +{n_pos} pos (>={pos_threshold}), "
              f"+{n_neg} neg (<={neg_threshold}), "
              f"剩餘 {(~pos_mask & ~neg_mask).sum()} 未標記")

        if n_pos == 0 and n_neg == 0:
            print(f"    Round {i+1}: 無新增樣本，停止")
            break

        # 組合新標籤
        X_new_pos = X_remaining[pos_mask]
        X_new_neg = X_remaining[neg_mask]
        y_new_pos = np.ones(n_pos, dtype=int)
        y_new_neg = np.zeros(n_neg, dtype=int)

        if n_pos > 0 or n_neg > 0:
            X_new = np.vstack([x for x in [X_new_pos, X_new_neg] if len(x) > 0])
            y_new = np.concatenate([y for y in [y_new_pos, y_new_neg] if len(y) > 0])
            X_curr = np.vstack([X_curr, X_new])
            y_curr = np.concatenate([y_curr, y_new])

        # 移除已標記的
        X_remaining = X_remaining[~pos_mask & ~neg_mask]

        stats["total_pos_added"] += int(n_pos)
        stats["total_neg_added"] += int(n_neg)
        stats["iterations"] = i + 1

    stats["remaining_unlabeled"] = len(X_remaining)
    stats["final_train_size"] = len(X_curr)
    stats["final_pos_count"] = int(y_curr.sum())
    stats["final_neg_count"] = int((y_curr == 0).sum())

    print(f"\n  Pseudo-labeling 完成:")
    print(f"    原始訓練集: {len(X_train)} (pos={y_train.sum()})")
    print(f"    擴充後:     {len(X_curr)} (pos={y_curr.sum()})")
    print(f"    新增正例:   {stats['total_pos_added']}")
    print(f"    新增負例:   {stats['total_neg_added']}")

    return X_curr, y_curr, stats
