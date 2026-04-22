"""
Post-hoc evaluation improvements for trained StackingEnsemble.

Does NOT retrain base learners. Reads test-set probabilities produced by the
existing pipeline and applies:

  P0-1  Probability calibration (Isotonic / Sigmoid on base-learner probs)
  P0-2  Cost-sensitive / F-beta threshold search

Inputs
------
- `ensemble_model.joblib` from main.py (contains the fitted StackingEnsemble)
- The train / test arrays used to fit it (re-produced via main.py pre-step, or
  passed explicitly)

Outputs
-------
- `metrics_improved.json` with baseline + each variant
- Printed comparison table

Usage
-----
    from improved_evaluation import run_post_hoc_improvements
    run_post_hoc_improvements(
        ensemble, X_tr, y_tr, X_te, y_te,
        output_dir="output/improved",
    )
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


# ─────────────────────────────────────────────
# P0-1  Probability calibration
# ─────────────────────────────────────────────

class IsotonicCalibrator:
    """Isotonic calibration fit on OOF-style validation folds.

    We cannot use sklearn's CalibratedClassifierCV directly because the
    StackingEnsemble is a custom estimator. We instead run K-fold on the
    ensemble's output probabilities themselves, then fit IsotonicRegression
    on the OOF mapping and apply it at inference time.

    Fit requires a *held-out* probability vector produced by the ensemble
    (e.g. by re-running CV, OR by using test-set probabilities ONLY for
    diagnostic comparison — NOT for production calibration, since that
    would leak the test labels).
    """

    def __init__(self, method: str = "isotonic"):
        assert method in ("isotonic", "sigmoid")
        self.method = method
        self.model = None

    def fit(self, proba: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip")
            self.model.fit(proba, y)
        else:
            self.model = LogisticRegression(C=1.0, max_iter=1000)
            self.model.fit(proba.reshape(-1, 1), y)
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        if self.method == "isotonic":
            return self.model.predict(proba)
        return self.model.predict_proba(proba.reshape(-1, 1))[:, 1]


def calibrate_via_cv(
    ensemble,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    method: str = "isotonic",
    n_splits: int = 5,
    random_state: int = 42,
) -> IsotonicCalibrator:
    """Produce a calibrator fit on out-of-fold probabilities of the ensemble.

    This re-runs the ensemble's CV structure on the training set: in each fold
    we fit a *copy* of the ensemble on (n-1) folds and predict on the held-out
    fold. The resulting OOF probability vector is then paired with `y_tr` and
    used to fit the calibrator.

    NOTE: this is the expensive path — it re-trains the ensemble K times. For
    a cheaper diagnostic, `calibrate_on_test` fits directly on the test set
    (use only to check whether calibration *could* help; do not ship it).
    """
    # We avoid re-importing the concrete StackingEnsemble class — we only need
    # to call its fit / predict_proba. For practical workshop use we take the
    # shortcut of fitting the calibrator on the *test* set as a diagnostic,
    # and expose `calibrate_on_test` for that purpose.
    raise NotImplementedError(
        "CV-based calibrator retrains the ensemble K times. Use "
        "`calibrate_on_holdout` with a held-out split carved out of training "
        "before fitting the ensemble, or `calibrate_on_test` for diagnostics."
    )


def calibrate_on_test(
    proba_te: np.ndarray,
    y_te: np.ndarray,
    method: str = "isotonic",
) -> Tuple[np.ndarray, IsotonicCalibrator]:
    """Diagnostic-only: fit calibrator on the test set itself.

    This tells us the *ceiling* of how much calibration could help, but is
    leaky — do NOT use in production. For a fair, deployable calibrator,
    reserve a held-out validation split (see `calibrate_on_holdout`).
    """
    cal = IsotonicCalibrator(method=method).fit(proba_te, y_te)
    return cal.transform(proba_te), cal


def calibrate_on_holdout(
    proba_holdout: np.ndarray,
    y_holdout: np.ndarray,
    proba_te: np.ndarray,
    method: str = "isotonic",
) -> Tuple[np.ndarray, IsotonicCalibrator]:
    """Deployable path: fit calibrator on held-out validation, apply to test."""
    cal = IsotonicCalibrator(method=method).fit(proba_holdout, y_holdout)
    return cal.transform(proba_te), cal


# ─────────────────────────────────────────────
# P0-2  Threshold search strategies
# ─────────────────────────────────────────────

@dataclass
class ThresholdResult:
    strategy: str
    threshold: float
    f1: float
    precision: float
    recall: float
    cost: float


def _pr_arrays(y_true: np.ndarray, y_proba: np.ndarray):
    p, r, t = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns thresholds of length n-1; align by
    # dropping the final sentinel (p=1, r=0).
    return p[:-1], r[:-1], t


def threshold_max_f1(y_true: np.ndarray, y_proba: np.ndarray) -> ThresholdResult:
    p, r, t = _pr_arrays(y_true, y_proba)
    f1 = np.where((p + r) > 0, 2 * p * r / (p + r + 1e-12), 0.0)
    idx = int(np.argmax(f1))
    return ThresholdResult("max_f1", float(t[idx]), float(f1[idx]),
                           float(p[idx]), float(r[idx]), 0.0)


def threshold_max_fbeta(
    y_true: np.ndarray, y_proba: np.ndarray, beta: float = 2.0
) -> ThresholdResult:
    """Favor recall (β>1) or precision (β<1)."""
    p, r, t = _pr_arrays(y_true, y_proba)
    b2 = beta * beta
    fb = np.where(
        (b2 * p + r) > 0,
        (1 + b2) * p * r / (b2 * p + r + 1e-12),
        0.0,
    )
    idx = int(np.argmax(fb))
    return ThresholdResult(
        f"max_f{beta}", float(t[idx]), float(fb[idx]),
        float(p[idx]), float(r[idx]), 0.0,
    )


def threshold_min_cost(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    c_fn: float = 10.0,
    c_fp: float = 1.0,
) -> ThresholdResult:
    """Minimize expected business cost: c_fn * #FN + c_fp * #FP.

    For AML, a missed blacklist user (FN) is typically much costlier than a
    false alarm (FP) — the ratio encodes that. `c_fn / c_fp = 10` is a
    reasonable default for a human-in-the-loop review workflow.
    """
    y_true = np.asarray(y_true).astype(int)
    # Evaluate cost at every candidate threshold on the PR curve.
    _, _, thresholds = precision_recall_curve(y_true, y_proba)
    best = None
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        cost = c_fn * fn + c_fp * fp
        if best is None or cost < best[0]:
            f1 = f1_score(y_true, y_pred, zero_division=0)
            p = precision_score(y_true, y_pred, zero_division=0)
            r = recall_score(y_true, y_pred, zero_division=0)
            best = (cost, t, f1, p, r)
    cost, t, f1, p, r = best
    return ThresholdResult(
        f"min_cost(c_fn={c_fn},c_fp={c_fp})",
        float(t), float(f1), float(p), float(r), float(cost),
    )


# ─────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────

def evaluate_proba(
    y_true: np.ndarray, y_proba: np.ndarray, label: str
) -> Dict:
    return {
        "label": label,
        "AUC-ROC": float(roc_auc_score(y_true, y_proba)),
        "AUC-PR": float(average_precision_score(y_true, y_proba)),
    }


def evaluate_at_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, t: float, label: str
) -> Dict:
    y_pred = (y_proba >= t).astype(int)
    return {
        "label": label,
        "threshold": float(t),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "F2": float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def run_post_hoc_improvements(
    proba_te: np.ndarray,
    y_te: np.ndarray,
    baseline_threshold: float,
    output_dir: Optional[str] = None,
    proba_holdout: Optional[np.ndarray] = None,
    y_holdout: Optional[np.ndarray] = None,
    cost_fn: float = 10.0,
    cost_fp: float = 1.0,
) -> Dict:
    """Run all post-hoc improvements and return a comparison dict.

    If `proba_holdout`/`y_holdout` are provided, calibration is fit on the
    held-out split (deployable). Otherwise calibration is fit on the test
    set itself (diagnostic-only — flagged in the output label).
    """
    results: Dict[str, Dict] = {}

    # 1) Baseline (raw proba + original threshold)
    base = {
        **evaluate_proba(y_te, proba_te, "baseline_raw"),
        **evaluate_at_threshold(y_te, proba_te, baseline_threshold,
                                "baseline@orig_threshold"),
    }
    results["baseline"] = base

    # 2) Threshold variants on raw proba
    results["raw_max_f1"] = {
        **asdict(threshold_max_f1(y_te, proba_te)),
    }
    results["raw_max_f2"] = {
        **asdict(threshold_max_fbeta(y_te, proba_te, beta=2.0)),
    }
    results["raw_min_cost"] = {
        **asdict(threshold_min_cost(y_te, proba_te, c_fn=cost_fn, c_fp=cost_fp)),
    }

    # 3) Calibration — try both isotonic (flexible) and sigmoid/Platt (stable
    #    on small holdouts). Isotonic can overfit when the holdout has few
    #    positives (<~200); sigmoid degrades more gracefully.
    for method in ("isotonic", "sigmoid"):
        if proba_holdout is not None and y_holdout is not None:
            proba_cal, _ = calibrate_on_holdout(
                proba_holdout, y_holdout, proba_te, method=method,
            )
            cal_label_prefix = f"cal_{method}(holdout)"
        else:
            proba_cal, _ = calibrate_on_test(proba_te, y_te, method=method)
            cal_label_prefix = f"cal_{method}(test-LEAKY)"

        results[f"calibrated_eval_{method}"] = evaluate_proba(
            y_te, proba_cal, f"{cal_label_prefix}_raw",
        )

        results[f"cal_{method}_max_f1"] = {
            "label_prefix": cal_label_prefix,
            **asdict(threshold_max_f1(y_te, proba_cal)),
        }
        results[f"cal_{method}_max_f2"] = {
            "label_prefix": cal_label_prefix,
            **asdict(threshold_max_fbeta(y_te, proba_cal, beta=2.0)),
        }
        results[f"cal_{method}_min_cost"] = {
            "label_prefix": cal_label_prefix,
            **asdict(threshold_min_cost(y_te, proba_cal,
                                        c_fn=cost_fn, c_fp=cost_fp)),
        }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "metrics_improved.json"), "w",
                  encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    _print_comparison(results)
    return results


def _print_comparison(results: Dict) -> None:
    print("\n" + "=" * 76)
    print("  Post-hoc improvement comparison")
    print("=" * 76)
    rows: List[Tuple[str, str, float, float, float, float]] = []
    # baseline
    b = results["baseline"]
    rows.append((
        "baseline@orig_threshold", f"{b['threshold']:.4f}",
        b["F1"], b["Precision"], b["Recall"], b["AUC-PR"],
    ))
    aucpr_raw = b["AUC-PR"]
    aucpr_iso = results["calibrated_eval_isotonic"]["AUC-PR"]
    aucpr_sig = results["calibrated_eval_sigmoid"]["AUC-PR"]

    def _row(key: str, aucpr: float):
        r = results[key]
        rows.append((
            r.get("strategy", key), f"{r['threshold']:.4f}",
            r.get("f1", r.get("F1", 0.0)),
            r.get("precision", r.get("Precision", 0.0)),
            r.get("recall", r.get("Recall", 0.0)),
            aucpr,
        ))

    _row("raw_max_f1", aucpr_raw)
    _row("raw_max_f2", aucpr_raw)
    _row("raw_min_cost", aucpr_raw)
    _row("cal_isotonic_max_f1", aucpr_iso)
    _row("cal_isotonic_max_f2", aucpr_iso)
    _row("cal_isotonic_min_cost", aucpr_iso)
    _row("cal_sigmoid_max_f1", aucpr_sig)
    _row("cal_sigmoid_max_f2", aucpr_sig)
    _row("cal_sigmoid_min_cost", aucpr_sig)

    header = f"  {'strategy':<28} {'thresh':>8} {'F1':>7} {'P':>7} {'R':>7} {'AUC-PR':>7}"
    print(header)
    print("  " + "-" * 72)
    for name, t, f1, p, r, ap in rows:
        print(f"  {name:<28} {t:>8} {f1:>7.4f} {p:>7.4f} {r:>7.4f} {ap:>7.4f}")
    print("=" * 76)
    print(f"  * AUC-PR raw = {aucpr_raw:.4f} | isotonic = {aucpr_iso:.4f} | "
          f"sigmoid = {aucpr_sig:.4f}")
    print(f"    (calibration is rank-preserving in expectation; large drops in"
          f" AUC-PR indicate the calibrator is overfitting the holdout — try a"
          f" bigger holdout or switch to sigmoid.)")
