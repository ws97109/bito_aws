"""
Fairness Audit Module — 模型公平性檢測

針對受保護屬性（性別、年齡、職業、收入來源）進行公平性指標計算，
產出報告 CSV、JSON 摘要與視覺化圖表。

指標：
  - Demographic Parity Difference (DPD)
  - Equalized Odds (TPR Gap / FPR Gap)
  - Predictive Parity (Precision Gap)
  - Disparate Impact Ratio (DIR)
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


# ─── 受保護屬性定義 ───────────────────────────

PROTECTED_ATTRS = {
    "is_female": {
        "label": "性別 (Gender)",
        "type": "binary",
        "group_names": {0: "男性 (Male)", 1: "女性 (Female)"},
    },
    "age": {
        "label": "年齡 (Age)",
        "type": "continuous",
        "bins": [0, 30, 50, 200],
        "group_names": {0: "< 30 歲", 1: "30–50 歲", 2: "> 50 歲"},
    },
    "is_high_risk_career": {
        "label": "職業風險 (Career Risk)",
        "type": "binary",
        "group_names": {0: "一般職業", 1: "高風險職業"},
    },
    "is_high_risk_income": {
        "label": "收入來源 (Income Source)",
        "type": "binary",
        "group_names": {0: "一般收入", 1: "高風險收入"},
    },
}

# ─── 判定標準 ─────────────────────────────────

THRESHOLDS = {
    "dpd":  {"pass": 0.05, "warning": 0.10},   # Demographic Parity Diff
    "tpr":  {"pass": 0.05, "warning": 0.10},   # TPR Gap
    "fpr":  {"pass": 0.05, "warning": 0.10},   # FPR Gap
    "dir_low":  0.80,                            # Disparate Impact Ratio
    "dir_high": 1.25,
    "dir_warn_low": 0.60,
    "dir_warn_high": 1.50,
}


def _judge(value, metric_type="gap"):
    """根據閾值判定 PASS / WARNING / FAIL"""
    if metric_type == "gap":
        v = abs(value)
        if v < THRESHOLDS["dpd"]["pass"]:
            return "PASS ✅"
        elif v < THRESHOLDS["dpd"]["warning"]:
            return "WARNING ⚠️"
        else:
            return "FAIL ❌"
    elif metric_type == "dir":
        if THRESHOLDS["dir_low"] <= value <= THRESHOLDS["dir_high"]:
            return "PASS ✅"
        elif THRESHOLDS["dir_warn_low"] <= value <= THRESHOLDS["dir_warn_high"]:
            return "WARNING ⚠️"
        else:
            return "FAIL ❌"


def _group_metrics(y_true, y_pred):
    """計算單一群組的 TPR, FPR, Precision, Positive Rate"""
    if len(y_true) == 0:
        return {"n": 0, "positive_rate": np.nan, "tpr": np.nan,
                "fpr": np.nan, "precision": np.nan}

    n = len(y_true)
    positive_rate = y_pred.mean()

    # 處理只有一個 class 的邊界情況
    labels = np.unique(y_true)
    if len(labels) < 2:
        # 只有正類或只有負類
        if labels[0] == 1:
            tpr = y_pred[y_true == 1].mean() if (y_true == 1).sum() > 0 else np.nan
            fpr = np.nan
        else:
            tpr = np.nan
            fpr = y_pred[y_true == 0].mean() if (y_true == 0).sum() > 0 else np.nan
        precision = (y_true[y_pred == 1].mean()
                     if (y_pred == 1).sum() > 0 else np.nan)
        return {"n": n, "positive_rate": positive_rate, "tpr": tpr,
                "fpr": fpr, "precision": precision}

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan

    return {
        "n": n,
        "n_positive": int(tp + fn),
        "n_negative": int(tn + fp),
        "true_positive_rate": float(y_true.mean()),
        "positive_rate": float(positive_rate),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "precision": float(precision),
    }


def audit_attribute(feat_col, y_true, y_pred, attr_config):
    """
    對單一受保護屬性進行公平性檢測。

    Parameters
    ----------
    feat_col : array-like  受保護屬性的特徵值
    y_true   : array-like  真實標籤
    y_pred   : array-like  預測標籤
    attr_config : dict     屬性設定（來自 PROTECTED_ATTRS）

    Returns
    -------
    dict  包含各群組指標與公平性判定
    """
    feat_col = np.asarray(feat_col)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # 分群
    if attr_config["type"] == "binary":
        groups = feat_col.astype(int)
    else:
        groups = np.digitize(feat_col, attr_config["bins"][1:-1])

    unique_groups = sorted(np.unique(groups))
    group_names = attr_config["group_names"]

    # 各群組指標
    group_results = {}
    for g in unique_groups:
        mask = groups == g
        name = group_names.get(g, f"Group {g}")
        group_results[name] = _group_metrics(y_true[mask], y_pred[mask])

    # 公平性指標（兩兩比較，取最大差距）
    positive_rates = [r["positive_rate"] for r in group_results.values()
                      if not np.isnan(r["positive_rate"])]
    tprs = [r["tpr"] for r in group_results.values()
            if not np.isnan(r.get("tpr", np.nan))]
    fprs = [r["fpr"] for r in group_results.values()
            if not np.isnan(r.get("fpr", np.nan))]
    precisions = [r["precision"] for r in group_results.values()
                  if not np.isnan(r.get("precision", np.nan))]

    dpd = max(positive_rates) - min(positive_rates) if len(positive_rates) >= 2 else np.nan
    tpr_gap = max(tprs) - min(tprs) if len(tprs) >= 2 else np.nan
    fpr_gap = max(fprs) - min(fprs) if len(fprs) >= 2 else np.nan
    precision_gap = max(precisions) - min(precisions) if len(precisions) >= 2 else np.nan

    # Disparate Impact Ratio (min_rate / max_rate)
    if len(positive_rates) >= 2 and max(positive_rates) > 0:
        dir_ratio = min(positive_rates) / max(positive_rates)
    else:
        dir_ratio = np.nan

    fairness_metrics = {
        "demographic_parity_diff": dpd,
        "tpr_gap": tpr_gap,
        "fpr_gap": fpr_gap,
        "precision_gap": precision_gap,
        "disparate_impact_ratio": dir_ratio,
    }

    # 判定
    judgments = {
        "demographic_parity": _judge(dpd, "gap"),
        "equalized_odds_tpr": _judge(tpr_gap, "gap"),
        "equalized_odds_fpr": _judge(fpr_gap, "gap"),
        "disparate_impact": _judge(dir_ratio, "dir") if not np.isnan(dir_ratio) else "N/A",
    }

    # 總判定：最嚴格的結果
    all_judgments = [v for v in judgments.values() if v != "N/A"]
    if any("FAIL" in j for j in all_judgments):
        overall = "FAIL ❌"
    elif any("WARNING" in j for j in all_judgments):
        overall = "WARNING ⚠️"
    else:
        overall = "PASS ✅"

    return {
        "attribute": attr_config["label"],
        "groups": group_results,
        "fairness_metrics": fairness_metrics,
        "judgments": judgments,
        "overall": overall,
    }


def run_fairness_audit(feat_df, y_true, y_pred, output_dir):
    """
    執行完整公平性檢測。

    Parameters
    ----------
    feat_df   : DataFrame  含受保護屬性欄位的特徵 DataFrame
    y_true    : array-like 真實標籤
    y_pred    : array-like 預測標籤（二元）
    output_dir: str        輸出目錄

    Returns
    -------
    dict  完整審計結果
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    results = {}
    rows = []  # for CSV

    for attr_name, attr_config in PROTECTED_ATTRS.items():
        if attr_name not in feat_df.columns:
            print(f"    [跳過] {attr_name} 不在特徵中")
            continue

        feat_col = feat_df[attr_name].values
        result = audit_attribute(feat_col, y_true, y_pred, attr_config)
        results[attr_name] = result

        # 印出結果
        print(f"\n  ── {result['attribute']} ──")
        print(f"  {'群組':<16} {'樣本數':>7} {'黑名單率':>8} {'預測正率':>8}"
              f" {'TPR':>7} {'FPR':>7} {'Precision':>10}")
        print(f"  {'─'*75}")
        for gname, gdata in result["groups"].items():
            n = gdata["n"]
            true_pos = gdata.get("true_positive_rate", np.nan)
            pos_rate = gdata["positive_rate"]
            tpr = gdata.get("tpr", np.nan)
            fpr = gdata.get("fpr", np.nan)
            prec = gdata.get("precision", np.nan)
            print(f"  {gname:<16} {n:>7} {true_pos:>8.4f} {pos_rate:>8.4f}"
                  f" {tpr:>7.4f} {fpr:>7.4f} {prec:>10.4f}")

        fm = result["fairness_metrics"]
        jd = result["judgments"]
        print(f"\n  指標：")
        print(f"    Demographic Parity Diff : {fm['demographic_parity_diff']:.4f}  "
              f"→ {jd['demographic_parity']}")
        print(f"    TPR Gap (Eq. Odds)      : {fm['tpr_gap']:.4f}  "
              f"→ {jd['equalized_odds_tpr']}")
        print(f"    FPR Gap (Eq. Odds)      : {fm['fpr_gap']:.4f}  "
              f"→ {jd['equalized_odds_fpr']}")
        print(f"    Disparate Impact Ratio  : {fm['disparate_impact_ratio']:.4f}  "
              f"→ {jd['disparate_impact']}")
        print(f"    ── 總判定：{result['overall']}")

        # CSV row
        rows.append({
            "attribute": result["attribute"],
            "demographic_parity_diff": fm["demographic_parity_diff"],
            "tpr_gap": fm["tpr_gap"],
            "fpr_gap": fm["fpr_gap"],
            "precision_gap": fm["precision_gap"],
            "disparate_impact_ratio": fm["disparate_impact_ratio"],
            "dpd_judgment": jd["demographic_parity"],
            "tpr_judgment": jd["equalized_odds_tpr"],
            "fpr_judgment": jd["equalized_odds_fpr"],
            "dir_judgment": jd["disparate_impact"],
            "overall": result["overall"],
        })

    # ── 輸出 CSV ──
    report_df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "fairness_report.csv")
    report_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  已儲存：fairness_report.csv")

    # ── 輸出 JSON ──
    summary = {
        "description": "模型公平性檢測報告",
        "thresholds": THRESHOLDS,
        "results": {},
    }
    for attr_name, result in results.items():
        summary["results"][attr_name] = {
            "label": result["attribute"],
            "overall": result["overall"],
            "fairness_metrics": {
                k: round(v, 6) if not np.isnan(v) else None
                for k, v in result["fairness_metrics"].items()
            },
            "judgments": result["judgments"],
            "group_stats": {
                gname: {k: round(v, 6) if isinstance(v, float) and not np.isnan(v) else v
                        for k, v in gdata.items()}
                for gname, gdata in result["groups"].items()
            },
        }

    # 建議措施
    recommendations = []
    for attr_name, result in results.items():
        if "FAIL" in result["overall"]:
            recommendations.append(
                f"[嚴重] {result['attribute']}：公平性未通過，"
                f"建議移除該特徵或加入公平性約束重新訓練。"
            )
        elif "WARNING" in result["overall"]:
            recommendations.append(
                f"[注意] {result['attribute']}：公平性指標接近閾值，"
                f"建議持續監控，並評估是否需要事後校正 (post-hoc calibration)。"
            )
    if not recommendations:
        recommendations.append("所有受保護屬性均通過公平性檢測。")
    summary["recommendations"] = recommendations

    json_path = os.path.join(output_dir, "fairness_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  已儲存：fairness_summary.json")

    # ── 視覺化 ──
    _plot_fairness_charts(results, output_dir)

    return results


def _plot_fairness_charts(results, output_dir):
    """產出公平性視覺化圖表"""
    plt.rcParams["font.family"] = ["Arial Unicode MS", "Heiti TC",
                                    "Microsoft JhengHei", "sans-serif"]
    n_attrs = len(results)
    if n_attrs == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Fairness Audit Report", fontsize=16, fontweight="bold")

    # ── Chart 1: Demographic Parity (各群組預測正率) ──
    ax = axes[0, 0]
    ax.set_title("Demographic Parity\n(Predicted Positive Rate by Group)")
    all_groups = []
    all_rates = []
    all_colors = []
    color_map = plt.cm.Set2(np.linspace(0, 1, n_attrs))
    for i, (attr_name, result) in enumerate(results.items()):
        for gname, gdata in result["groups"].items():
            all_groups.append(f"{gname}")
            all_rates.append(gdata["positive_rate"])
            all_colors.append(color_map[i])
    bars = ax.bar(range(len(all_groups)), all_rates, color=all_colors)
    ax.set_xticks(range(len(all_groups)))
    ax.set_xticklabels(all_groups, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Predicted Positive Rate")
    ax.axhline(y=np.mean(all_rates), color="red", linestyle="--",
               alpha=0.5, label="Mean")
    ax.legend(fontsize=8)

    # ── Chart 2: TPR by Group ──
    ax = axes[0, 1]
    ax.set_title("Equalized Odds — TPR\n(True Positive Rate by Group)")
    groups_tpr = []
    tprs = []
    colors_tpr = []
    for i, (attr_name, result) in enumerate(results.items()):
        for gname, gdata in result["groups"].items():
            tpr_val = gdata.get("tpr", np.nan)
            if not np.isnan(tpr_val):
                groups_tpr.append(f"{gname}")
                tprs.append(tpr_val)
                colors_tpr.append(color_map[i])
    if tprs:
        ax.bar(range(len(groups_tpr)), tprs, color=colors_tpr)
        ax.set_xticks(range(len(groups_tpr)))
        ax.set_xticklabels(groups_tpr, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("True Positive Rate")

    # ── Chart 3: FPR by Group ──
    ax = axes[1, 0]
    ax.set_title("Equalized Odds — FPR\n(False Positive Rate by Group)")
    groups_fpr = []
    fprs = []
    colors_fpr = []
    for i, (attr_name, result) in enumerate(results.items()):
        for gname, gdata in result["groups"].items():
            fpr_val = gdata.get("fpr", np.nan)
            if not np.isnan(fpr_val):
                groups_fpr.append(f"{gname}")
                fprs.append(fpr_val)
                colors_fpr.append(color_map[i])
    if fprs:
        ax.bar(range(len(groups_fpr)), fprs, color=colors_fpr)
        ax.set_xticks(range(len(groups_fpr)))
        ax.set_xticklabels(groups_fpr, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("False Positive Rate")

    # ── Chart 4: Summary Dashboard ──
    ax = axes[1, 1]
    ax.set_title("Fairness Summary")
    ax.axis("off")

    table_data = []
    for attr_name, result in results.items():
        fm = result["fairness_metrics"]
        table_data.append([
            result["attribute"],
            f"{fm['demographic_parity_diff']:.4f}",
            f"{fm['tpr_gap']:.4f}",
            f"{fm['fpr_gap']:.4f}",
            f"{fm['disparate_impact_ratio']:.4f}",
            result["overall"].replace(" ✅", "").replace(" ⚠️", "").replace(" ❌", ""),
        ])

    if table_data:
        table = ax.table(
            cellText=table_data,
            colLabels=["Attribute", "DPD", "TPR Gap", "FPR Gap", "DIR", "Result"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.5)

        # 根據判定結果上色
        for i, row in enumerate(table_data):
            result_text = row[-1]
            if "FAIL" in result_text:
                color = "#ffcccc"
            elif "WARNING" in result_text:
                color = "#fff3cd"
            else:
                color = "#d4edda"
            for j in range(len(row)):
                table[i + 1, j].set_facecolor(color)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "fairness_charts.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已儲存：fairness_charts.png")
