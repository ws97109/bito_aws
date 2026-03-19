"""
Feature Selection Module
特徵篩選：零方差移除 → 高相關性去重 → LightGBM 重要性篩選
"""
import pandas as pd
import numpy as np
from typing import List, Tuple


def remove_zero_variance(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """移除零方差特徵（對所有樣本都是同一個值）"""
    stds = X.std()
    zero_var_cols = stds[stds == 0].index.tolist()
    if zero_var_cols:
        X = X.drop(columns=zero_var_cols)
    return X, zero_var_cols


def remove_high_correlation(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.95,
) -> Tuple[pd.DataFrame, List[Tuple[str, str, float]]]:
    """
    移除高相關性特徵對中與 target 相關性較低的那個。
    回傳：篩選後的 DataFrame, 移除紀錄 [(dropped, kept, corr)]
    """
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # 找出所有高相關性配對
    pairs = []
    for col in upper.columns:
        for row in upper.index:
            val = upper.loc[row, col]
            if pd.notna(val) and val > threshold:
                pairs.append((row, col, val))

    # 按相關性排序，從最高的開始處理
    pairs.sort(key=lambda x: -x[2])

    dropped_cols = set()
    drop_records = []

    for a, b, corr_val in pairs:
        # 如果兩個都已經被移除，跳過
        if a in dropped_cols and b in dropped_cols:
            continue
        # 如果其中一個已被移除，跳過
        if a in dropped_cols or b in dropped_cols:
            continue

        # 比較跟 target 的相關性，移除較低的
        corr_a = abs(X[a].corr(y))
        corr_b = abs(X[b].corr(y))

        if corr_a >= corr_b:
            dropped_cols.add(b)
            drop_records.append((b, a, corr_val))
        else:
            dropped_cols.add(a)
            drop_records.append((a, b, corr_val))

    X = X.drop(columns=list(dropped_cols))
    return X, drop_records


def remove_low_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 5,
) -> Tuple[pd.DataFrame, List[str]]:
    """用 LightGBM 跑一輪，移除 importance = 0 的特徵"""
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        print("  [警告] lightgbm 未安裝，跳過重要性篩選")
        return X, []

    from sklearn.model_selection import StratifiedKFold

    importance = np.zeros(X.shape[1])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        model = LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
            random_state=42,
            verbose=-1,
        )
        model.fit(X.values[tr_idx], y[tr_idx])
        importance += model.feature_importances_

    importance /= n_splits

    # 移除 importance = 0 的特徵
    zero_imp_cols = [
        col for col, imp in zip(X.columns, importance) if imp == 0
    ]

    if zero_imp_cols:
        X = X.drop(columns=zero_imp_cols)

    return X, zero_imp_cols


def select_features(
    X: pd.DataFrame,
    y: np.ndarray,
    corr_threshold: float = 0.95,
) -> Tuple[pd.DataFrame, dict]:
    """
    完整特徵篩選流程，回傳篩選後的 DataFrame 和篩選報告。
    """
    report = {"original_count": X.shape[1]}

    print(f"\n  原始特徵數: {X.shape[1]}")

    # Step 1: 零方差
    X, zero_var_cols = remove_zero_variance(X)
    report["zero_variance_removed"] = zero_var_cols
    print(f"  零方差移除: {len(zero_var_cols)} 個 {zero_var_cols}")

    # Step 2: 高相關性
    y_series = pd.Series(y, index=X.index)
    X, corr_records = remove_high_correlation(X, y_series, corr_threshold)
    report["high_corr_removed"] = [
        {"dropped": d, "kept": k, "corr": round(c, 4)}
        for d, k, c in corr_records
    ]
    print(f"  高相關移除: {len(corr_records)} 個")
    for d, k, c in corr_records:
        print(f"    移除 {d} (保留 {k}, corr={c:.4f})")

    # Step 3: LightGBM 重要性
    X, zero_imp_cols = remove_low_importance(X, y)
    report["zero_importance_removed"] = zero_imp_cols
    print(f"  零重要移除: {len(zero_imp_cols)} 個 {zero_imp_cols}")

    report["final_count"] = X.shape[1]
    report["final_features"] = X.columns.tolist()
    print(f"\n  最終特徵數: {X.shape[1]}")

    return X, report
