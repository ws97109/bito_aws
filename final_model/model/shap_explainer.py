"""
Explainability Module
多層次可解釋性：
  1. SHAP（全域 + 個體）
  2. 反事實解釋（What-If）
  3. 風險因子報告生成
  4. SSR 穩定性評測
"""
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = ["DejaVu Sans"]
from typing import List, Dict, Optional, Tuple
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")


# 完整特徵中文名稱對照（對應篩選後 47 個特徵）
FEATURE_NAME_MAP = {
    # 用戶基本
    "kyc_speed_sec":              "KYC 完成速度（秒）",
    "account_age_days":           "帳號年齡（天）",
    "age":                        "用戶年齡",
    "is_high_risk_career":        "高風險職業",
    "is_high_risk_income":        "高風險收入來源",
    "career_income_risk":         "職業×收入複合風險",
    "is_app_user":                "APP 用戶",
    # 法幣行為
    "twd_dep_count":              "法幣入金次數",
    "twd_dep_sum":                "法幣入金總額",
    "twd_dep_mean":               "法幣入金均值",
    "twd_dep_std":                "法幣入金標準差",
    "twd_dep_max":                "法幣入金最大值",
    "twd_wit_count":              "法幣提領次數",
    "twd_wit_max":                "法幣提領最大值",
    "twd_net_flow":               "法幣淨流入金額",
    "twd_withdraw_ratio":         "法幣提領比率",
    "twd_smurf_flag":             "結構化交易警示",
    "twd_wit_ip_ratio":           "法幣提領IP覆蓋率",
    # 加密貨幣行為
    "crypto_dep_count":           "虛幣入金次數",
    "crypto_wit_count":           "虛幣提領次數",
    "crypto_wit_sum":             "虛幣提領總額（台幣）",
    "crypto_wit_mean":            "虛幣提領均值",
    "crypto_wit_max":             "虛幣提領最大值",
    "crypto_currency_diversity":  "幣種多樣性",
    "crypto_protocol_diversity":  "跨鏈協定數",
    "crypto_wallet_hash_nunique": "不同錢包地址數",
    "crypto_internal_count":      "內轉次數",
    "crypto_internal_peer_count": "內轉對象數",
    "crypto_wit_ip_ratio":        "虛幣提領IP覆蓋率",
    # 交易行為
    "trading_buy_ratio":          "買單比率",
    "trading_market_order_ratio": "市價單比率",
    "trading_mean":               "掛單均值",
    "trading_max":                "掛單最大值",
    "swap_count":                 "一鍵買賣次數",
    "swap_sum":                   "一鍵買賣總額",
    # IP 時序
    "ip_unique_count":            "使用IP數",
    "ip_night_ratio":             "深夜操作比例",
    "ip_max_shared":              "IP共用用戶數",
    # 資金流速
    "fund_stay_sec":              "資金停留時間（秒）",
    # 圖特徵
    "pagerank_score":             "PageRank 分數",
    "graph_in_degree":            "圖入度",
    "graph_out_degree":           "圖出度",
    "connected_component_size":   "連通分量大小",
    "betweenness_centrality":     "介數中心性",
    # 跨表衍生
    "total_tx_count":             "總交易次數",
    "first_to_last_tx_days":      "首末交易間隔（天）",
    "weekend_tx_ratio":           "週末交易佔比",
    # 行為紅旗
    "dep_to_first_wit_hours":     "入金到首筆提領時間（小時）",
    "twd_to_crypto_out_ratio":    "法幣入金/幣出比",
    "tx_amount_cv":               "交易金額變異係數",
    "rapid_kyc_then_trade":       "KYC後48小時內交易",
    "crypto_out_in_ratio":        "加密貨幣出入比",
    "same_day_in_out_count":      "同天入出金天數",
}


# ─────────────────────────────────────────────
# 1. SHAP 分析
# ─────────────────────────────────────────────

class SHAPExplainer:

    def __init__(self, model, feature_names: List[str]):
        self.model          = model
        self.feature_names  = feature_names
        self.display_names  = [FEATURE_NAME_MAP.get(f, f) for f in feature_names]
        self.explainer      = None
        self.shap_values    = None

    def fit(self, X_background: np.ndarray, X_explain: np.ndarray):
        """建立 TreeExplainer 並計算 SHAP 值"""
        print("計算 SHAP 值 ...")
        self.explainer  = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer.shap_values(X_explain)
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]   # 正類
        self.X_explain = X_explain
        return self

    def plot_global_importance(self, top_n: int = 20, save_path: str = None):
        """全域特徵重要性（蜂群圖）"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Bar plot
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        idx = np.argsort(mean_abs)[-top_n:]
        axes[0].barh(
            [self.display_names[i] for i in idx],
            mean_abs[idx],
            color="#5B8DB8",
        )
        axes[0].set_xlabel("Mean |SHAP value|")
        axes[0].set_title("全域特徵重要性 (Top {})".format(top_n))
        axes[0].grid(axis="x", alpha=0.3)

        # Summary beeswarm
        plt.sca(axes[1])
        shap.summary_plot(
            self.shap_values[:, idx],
            self.X_explain[:, idx],
            feature_names=[self.display_names[i] for i in idx],
            show=False,
            plot_size=None,
        )
        axes[1].set_title("SHAP 分布（蜂群圖）")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"全域重要性圖已儲存：{save_path}")

    def explain_user(
        self,
        user_idx: int,
        user_id: Optional[int] = None,
        risk_score: Optional[float] = None,
    ) -> Dict:
        """單一用戶 SHAP 解釋"""
        sv  = self.shap_values[user_idx]
        xv  = self.X_explain[user_idx]

        order = np.argsort(np.abs(sv))[::-1]

        report = {
            "user_id":    user_id,
            "risk_score": risk_score,
            "factors":    [],
        }
        for i in order[:10]:
            report["factors"].append({
                "feature":      self.display_names[i],
                "feature_key":  self.feature_names[i],
                "value":        float(xv[i]),
                "shap":         float(sv[i]),
                "direction":    "增加風險" if sv[i] > 0 else "降低風險",
            })
        return report

    def plot_user_waterfall(self, user_idx: int, save_path: str = None):
        """Waterfall 圖（單一用戶）"""
        exp = shap.Explanation(
            values       = self.shap_values[user_idx],
            base_values  = self.explainer.expected_value if not isinstance(
                self.explainer.expected_value, list
            ) else self.explainer.expected_value[1],
            data         = self.X_explain[user_idx],
            feature_names= self.display_names,
        )
        shap.waterfall_plot(exp, show=False, max_display=15)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


# ─────────────────────────────────────────────
# 2. 反事實解釋（What-If）
# ─────────────────────────────────────────────

class CounterfactualExplainer:
    """
    對高風險用戶生成「降至低風險需改變哪些行為」
    逐一翻轉可改變特徵，觀察分數變化
    """

    MUTABLE_FEATURES = [
        "twd_withdraw_ratio",
        "ip_night_ratio",
        "crypto_currency_diversity",
        "kyc_speed_sec",
        "fund_stay_sec",
        "twd_smurf_flag",
        "crypto_wit_count",
    ]

    def __init__(self, model, scaler, feature_names: List[str]):
        self.model         = model
        self.scaler        = scaler
        self.feature_names = feature_names

    def generate(
        self,
        X_row: np.ndarray,
        target_score: float = 0.3,
        top_k: int = 5,
    ) -> List[Dict]:
        X_scaled  = self.scaler.transform(X_row.reshape(1, -1))
        base_prob = self.model.predict_proba(X_scaled)[0, 1]

        suggestions = []
        for feat in self.MUTABLE_FEATURES:
            if feat not in self.feature_names:
                continue
            fi = self.feature_names.index(feat)
            orig_val = X_row[fi]

            X_cf = X_row.copy()
            X_cf[fi] = 0.0

            X_cf_scaled = self.scaler.transform(X_cf.reshape(1, -1))
            cf_prob = self.model.predict_proba(X_cf_scaled)[0, 1]
            delta   = base_prob - cf_prob

            if delta > 0:
                suggestions.append({
                    "feature":      FEATURE_NAME_MAP.get(feat, feat),
                    "current":      round(float(orig_val), 4),
                    "target":       0.0,
                    "score_drop":   round(float(delta), 4),
                })

        suggestions.sort(key=lambda x: x["score_drop"], reverse=True)
        return suggestions[:top_k]


# ─────────────────────────────────────────────
# 3. 風險報告生成
# ─────────────────────────────────────────────

RISK_LEVELS = [
    (0.8, "極高風險", "建議立即凍結帳戶並啟動人工調查"),
    (0.6, "高風險",   "建議暫停交易並要求補件 KYC"),
    (0.4, "中風險",   "建議設定交易限額並加強監控"),
    (0.2, "低風險",   "正常監控，定期複查"),
    (0.0, "正常",     "無需特殊處理"),
]

def score_to_level(score: float) -> Tuple:
    for threshold, level, action in RISK_LEVELS:
        if score >= threshold:
            return level, action
    return "正常", "無需特殊處理"

def generate_user_report(
    user_id: int,
    risk_score: float,
    shap_report: Dict,
    cf_suggestions: List[Dict],
) -> str:
    level, action = score_to_level(risk_score)
    lines = [
        f"╔══════════════════════════════════════════╗",
        f"  用戶風險報告  |  User ID: {user_id}",
        f"╚══════════════════════════════════════════╝",
        f"",
        f"  風險分數   : {risk_score:.4f}",
        f"  風險等級   : {level}",
        f"  建議行動   : {action}",
        f"",
        f"  ── 主要風險因子（SHAP）──",
    ]
    for i, fac in enumerate(shap_report["factors"][:5], 1):
        arrow = "▲" if fac["shap"] > 0 else "▼"
        lines.append(
            f"  {i}. {fac['feature']:<20} = {fac['value']:>10.3f}  "
            f"{arrow} {abs(fac['shap']):.4f}"
        )
    if cf_suggestions:
        lines += ["", "  ── 可改善建議（反事實）──"]
        for sug in cf_suggestions:
            lines.append(
                f"  • 若將「{sug['feature']}」從 {sug['current']} 調整至 {sug['target']}，"
                f"風險分數可降低 {sug['score_drop']:.4f}"
            )
    return "\n".join(lines)


# ─────────────────────────────────────────────
# 4. SSR 穩定性評測
# ─────────────────────────────────────────────

class SSREvaluator:
    """
    Stable Sample Ratio (SSR) — 驗證 SHAP 解釋的穩定性。
    基於 Type-aware Perturbation：依特徵語義類型做不同擾動。

    特徵類型：
      continuous    → 高斯擾動 N(0, ε·σ)
      binary        → 機率 ε 翻轉 0↔1
      label_encoded → 機率 ε 替換為另一有效值
      ordinal       → 機率 ε 移動 ±1 級
    """

    # 本專案特徵類型定義
    FEATURE_TYPE_MAP = {
        # binary
        "is_high_risk_career": "binary",
        "is_high_risk_income": "binary",
        "career_income_risk":  "binary",
        "is_app_user":         "binary",
        "twd_smurf_flag":      "binary",
        "is_female":           "binary",
        "rapid_kyc_then_trade": "binary",
        # ordinal（有序類別）
        "sex":                 "ordinal",
        "career":              "ordinal",
        "income_source":       "ordinal",
        "user_source":         "ordinal",
        # 其餘皆為 continuous（預設）
    }

    def __init__(
        self,
        explainer: SHAPExplainer,
        feature_names: List[str],
    ):
        self.shap_explainer = explainer
        self.feature_names  = feature_names
        # 解析每個特徵的類型
        self.feature_types = [
            self.FEATURE_TYPE_MAP.get(f, "continuous") for f in feature_names
        ]

    def _perturb_batch(
        self,
        X_batch: np.ndarray,
        epsilon: float,
        feature_stds: np.ndarray,
        valid_values: Dict,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """向量化 Type-aware 擾動（整批樣本一次處理）"""
        X_pert = X_batch.copy()
        n_samples = len(X_pert)

        for j, (fname, ftype) in enumerate(zip(self.feature_names, self.feature_types)):
            if ftype == "continuous":
                if feature_stds[j] > 0:
                    noise = rng.normal(0, epsilon * feature_stds[j], size=n_samples)
                    X_pert[:, j] = X_pert[:, j] + noise

            elif ftype == "binary":
                flip_mask = rng.random(n_samples) < epsilon
                X_pert[flip_mask, j] = 1.0 - X_pert[flip_mask, j]

            elif ftype == "label_encoded":
                replace_mask = rng.random(n_samples) < epsilon
                n_replace = replace_mask.sum()
                if n_replace > 0 and fname in valid_values:
                    vals = valid_values[fname]
                    if len(vals) > 1:
                        current = X_pert[replace_mask, j]
                        new_vals = np.empty(n_replace)
                        for i in range(n_replace):
                            candidates = [v for v in vals if v != current[i]]
                            new_vals[i] = rng.choice(candidates) if candidates else current[i]
                        X_pert[replace_mask, j] = new_vals

            elif ftype == "ordinal":
                move_mask = rng.random(n_samples) < epsilon
                n_move = move_mask.sum()
                if n_move > 0 and fname in valid_values:
                    vals = valid_values[fname]
                    if len(vals) > 1:
                        min_val, max_val = min(vals), max(vals)
                        current = X_pert[move_mask, j].copy()
                        directions = rng.choice([-1, 1], size=n_move)
                        new_vals = current + directions
                        # 邊界處理：超出範圍則反向
                        new_vals = np.where(new_vals < min_val, current + 1, new_vals)
                        new_vals = np.where(new_vals > max_val, current - 1, new_vals)
                        new_vals = np.clip(new_vals, min_val, max_val)
                        X_pert[move_mask, j] = new_vals

        return X_pert

    def _collect_valid_values(self, X: np.ndarray) -> Dict:
        """收集 label_encoded / ordinal 特徵的合法值"""
        valid_values = {}
        for j, (fname, ftype) in enumerate(zip(self.feature_names, self.feature_types)):
            if ftype in ("label_encoded", "ordinal"):
                valid_values[fname] = sorted(set(X[:, j].tolist()))
        return valid_values

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        epsilons: List[float] = None,
        top_k_list: List[int] = None,
        n_samples: int = 500,
        n_perturbations: int = 10,
        stability_threshold: float = 0.8,
        random_state: int = 42,
    ) -> Dict:
        """
        完整 SSR 評測（Type-aware Perturbation）。

        回傳：
        {
            "overall": {(epsilon, k): ssr_value, ...},
            "by_class": {
                0: {(epsilon, k): ssr_value, ...},
                1: {(epsilon, k): ssr_value, ...},
            }
        }
        """
        if epsilons is None:
            epsilons = [0.05, 0.10, 0.15, 0.20]
        if top_k_list is None:
            top_k_list = [1, 3, 5]

        rng = np.random.RandomState(random_state)

        # 抽樣
        n_samples = min(n_samples, len(X))
        sample_idx = rng.choice(len(X), n_samples, replace=False)
        X_sample = X[sample_idx]
        y_sample = y[sample_idx] if y is not None else None

        # 計算特徵標準差 & 合法值
        feature_stds = X.std(axis=0)
        feature_stds[feature_stds == 0] = 1.0
        valid_values = self._collect_valid_values(X)

        # 統計特徵類型分布
        type_counts = {}
        for ft in self.feature_types:
            type_counts[ft] = type_counts.get(ft, 0) + 1
        print(f"  特徵類型分布: {type_counts}")

        # 計算原始 SHAP 值
        print(f"  計算 {n_samples} 個樣本的原始 SHAP 值 ...")
        explainer = shap.TreeExplainer(self.shap_explainer.model)
        original_shap = explainer.shap_values(X_sample)
        if isinstance(original_shap, list):
            original_shap = original_shap[1]

        # 預計算 baseline Top-k
        baseline_topk = {}
        for k in top_k_list:
            baseline_topk[k] = [
                tuple(np.argsort(np.abs(original_shap[i]))[-k:][::-1])
                for i in range(n_samples)
            ]

        results = {"overall": {}, "by_class": {0: {}, 1: {}}}

        for eps in epsilons:
            print(f"  擾動 ε={eps:.2f} ...")

            # 每個 k 的穩定計數
            stability_counts = {k: np.zeros(n_samples) for k in top_k_list}

            for t in range(n_perturbations):
                trial_rng = np.random.RandomState(random_state + t * 1000 + int(eps * 100))
                X_perturbed = self._perturb_batch(
                    X_sample, eps, feature_stds, valid_values, trial_rng
                )
                pert_shap = explainer.shap_values(X_perturbed)
                if isinstance(pert_shap, list):
                    pert_shap = pert_shap[1]

                for k in top_k_list:
                    for i in range(n_samples):
                        pert_topk = tuple(np.argsort(np.abs(pert_shap[i]))[-k:][::-1])
                        if pert_topk == baseline_topk[k][i]:
                            stability_counts[k][i] += 1

            min_stable = int(stability_threshold * n_perturbations)
            for k in top_k_list:
                is_stable = stability_counts[k] >= min_stable
                ssr = is_stable.mean()
                results["overall"][(eps, k)] = round(float(ssr), 4)

                # 分群 SSR
                if y_sample is not None:
                    for cls in [0, 1]:
                        mask = y_sample == cls
                        if mask.sum() > 0:
                            cls_ssr = is_stable[mask].mean()
                            results["by_class"][cls][(eps, k)] = round(float(cls_ssr), 4)

                print(f"    SSR(ε={eps:.2f}, k={k}): {ssr:.4f}")

        return results

    def plot_ssr_curves(
        self,
        results: Dict,
        save_path: str = None,
    ):
        """繪製 SSR 衰減曲線"""
        overall = results["overall"]
        epsilons = sorted(set(e for e, k in overall.keys()))
        ks = sorted(set(k for e, k in overall.keys()))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 整體 SSR
        for k in ks:
            ssrs = [overall[(e, k)] for e in epsilons]
            axes[0].plot(epsilons, ssrs, marker="o", label=f"Top-{k}")
        axes[0].set_xlabel("Perturbation ε")
        axes[0].set_ylabel("SSR")
        axes[0].set_title("SSR Decay Curve (Overall)")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[0].set_ylim(0, 1.05)

        # 分群 SSR（Top-3）
        if results["by_class"][0] and results["by_class"][1]:
            target_k = 3 if 3 in ks else ks[0]
            for cls, label, color in [(0, "正常用戶", "#4CAF50"), (1, "黑名單用戶", "#F44336")]:
                ssrs = [results["by_class"][cls].get((e, target_k), 0) for e in epsilons]
                axes[1].plot(epsilons, ssrs, marker="s", label=label, color=color)
            axes[1].set_xlabel("Perturbation ε")
            axes[1].set_ylabel("SSR")
            axes[1].set_title(f"SSR by Class (Top-{target_k})")
            axes[1].legend()
            axes[1].grid(alpha=0.3)
            axes[1].set_ylim(0, 1.05)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"SSR 曲線圖已儲存：{save_path}")
