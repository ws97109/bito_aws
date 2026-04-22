"""Quick benchmark: TabPFN alone on our 5-fold OOF. Helps decide whether to
plumb it into the full stacking ensemble."""
import os
import sys
import time

import numpy as np
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE))

from run_improved_v2 import prepare  # noqa: E402
from anomaly_detection import add_anomaly_scores_to_splits  # noqa: E402
from tabpfn_base import tabpfn_oof_probabilities, tabpfn_fit_predict_test  # noqa: E402
from sklearn.metrics import (average_precision_score, roc_auc_score,
                              f1_score, precision_score, recall_score)
from improved_evaluation import threshold_max_fbeta


def main():
    data_dir = os.path.join(ROOT, "adjust_data", "train")
    X, y = prepare(data_dir)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )
    X_tr, X_te, _ = add_anomaly_scores_to_splits(X_tr, X_te)
    print(f"  tr={X_tr.shape}  te={X_te.shape}")

    t0 = time.time()
    oof, info = tabpfn_oof_probabilities(X_tr, y_tr, n_splits=5,
                                          max_rows_per_fold=5000)
    print(f"  5-fold OOF done in {time.time()-t0:.1f}s: {info}")

    t0 = time.time()
    te_proba = tabpfn_fit_predict_test(X_tr, y_tr, X_te, max_rows=5000)
    print(f"  test predict done in {time.time()-t0:.1f}s")

    print(f"\n  TabPFN alone on test set:")
    print(f"    AUC-ROC : {roc_auc_score(y_te, te_proba):.4f}")
    print(f"    AUC-PR  : {average_precision_score(y_te, te_proba):.4f}")
    # Threshold found on TRAIN OOF, applied to test
    t_f2 = threshold_max_fbeta(y_tr, oof, beta=2.0).threshold
    y_hat = (te_proba >= t_f2).astype(int)
    print(f"    @thr(max_f2 on train)={t_f2:.3f}: "
          f"F1={f1_score(y_te, y_hat):.4f}  "
          f"P={precision_score(y_te, y_hat):.4f}  "
          f"R={recall_score(y_te, y_hat):.4f}")


if __name__ == "__main__":
    main()
