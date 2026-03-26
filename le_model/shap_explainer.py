"""
Explainability Module
論文核心：多層次可解釋性
  1. SHAP（全域 + 個體）
  2. 決策規則萃取（RuleFit）
  3. 反事實解釋（What-If）
  4. 風險因子報告生成
"""
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = ["DejaVu Sans"]
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


FEATURE_NAME_MAP = {
    "has_kyc_level2":             "完成 KYC Level 2",
    "kyc_speed_sec":              "KYC 完成速度（秒）",
    "account_age_days":           "帳號年齡（天）",
    "age":                        "用戶年齡",
    "is_high_risk_career":        "高風險職業",
    "is_high_risk_income":        "高風險收入來源",
    "career_income_risk":         "職業×收入複合風險",
    "is_app_user":                "APP 用戶",
    "twd_dep_count":              "法幣入金次數",
    "twd_dep_sum":                "法幣入金總額",
    "twd_dep_mean":               "法幣入金均值",
    "twd_dep_std":                "法幣入金標準差",
    "twd_dep_max":                "法幣入金最大值",
    "twd_wit_count":              "法幣提領次數",
    "twd_wit_sum":                "法幣提領總額",
    "twd_wit_mean":               "法幣提領均值",
    "twd_wit_max":                "法幣提領最大值",
    "twd_withdraw_ratio":         "法幣提領比率",
    "twd_smurf_flag":             "結構化交易警示",
    "twd_wit_ip_ratio":           "法幣提領IP覆蓋率",
    "crypto_dep_count":           "虛幣入金次數",
    "crypto_dep_sum":             "虛幣入金總額（台幣）",
    "crypto_wit_count":           "虛幣提領次數",
    "crypto_wit_sum":             "虛幣提領總額（台幣）",
    "crypto_external_wit_count":  "鏈上提領次數",
    "crypto_currency_diversity":  "幣種多樣性",
    "crypto_protocol_diversity":  "跨鏈協定數",
    "crypto_internal_peer_count": "內轉對象數",
    "crypto_wit_ip_ratio":        "虛幣提領IP覆蓋率",
    "trading_count":              "掛單交易次數",
    "trading_sum":                "掛單總額",
    "trading_buy_ratio":          "買單比率",
    "trading_market_order_ratio": "市價單比率",
    "swap_count":                 "一鍵買賣次數",
    "swap_sum":                   "一鍵買賣總額",
    "total_trading_volume":       "交易總量",
    "ip_unique_count":            "使用IP數",
    "ip_total_count":             "IP操作總次數",
    "ip_night_ratio":             "深夜操作比例",
    "ip_max_shared":              "IP共用用戶數",
    "fund_stay_sec":              "資金停留時間（秒）",
    "composite_risk_score":       "複合風險分數",
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

        # Summary beeswarm（調用 shap 原生）
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

        # 排序貢獻
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
    簡化版：逐一翻轉可改變特徵，觀察分數變化
    """

    MUTABLE_FEATURES = [
        "twd_withdraw_ratio",
        "ip_night_ratio",
        "crypto_currency_diversity",
        "kyc_speed_sec",
        "fund_stay_sec",
        "twd_smurf_flag",
        "crypto_external_wit_count",
    ]

    def __init__(self, model, scaler, feature_names: List[str], gnn_proba=None):
        self.model         = model
        self.scaler        = scaler
        self.feature_names = feature_names
        self.gnn_proba     = gnn_proba

    def generate(
        self,
        X_row: np.ndarray,
        target_score: float = 0.3,
        top_k: int = 5,
        gnn_prob: float = None,
    ) -> List[Dict]:
        """
        返回最有效的 top_k 個可操作建議
        每個建議格式：{feature, from_value, to_value, score_change}
        """
        # 檢查 model 是否為 ensemble（有 predict_proba 方法且需要原始特徵）
        X_input = X_row.reshape(1, -1)

        # 準備 GNN probability（如果有的話）
        gnn_proba_array = np.array([gnn_prob]) if gnn_prob is not None else self.gnn_proba

        if hasattr(self.model, 'predict_proba'):
            # Ensemble 會自己處理縮放，傳入原始特徵和GNN概率
            if hasattr(self.model, 'use_gnn') and self.model.use_gnn:
                base_prob = self.model.predict_proba(X_input, gnn_proba=gnn_proba_array)[0]
            else:
                base_prob = self.model.predict_proba(X_input)[0]
            if isinstance(base_prob, np.ndarray):
                base_prob = base_prob if len(base_prob.shape) == 0 else base_prob
            else:
                base_prob = float(base_prob)
        else:
            X_scaled = self.scaler.transform(X_input)
            base_prob = self.model.predict_proba(X_scaled)[0, 1]

        suggestions = []
        for feat in self.MUTABLE_FEATURES:
            if feat not in self.feature_names:
                continue
            fi = self.feature_names.index(feat)
            orig_val = X_row[fi]

            # 嘗試將此特徵設為正常用戶分位數（25th）
            X_cf = X_row.copy()
            X_cf[fi] = 0.0   # 理想值（簡化）

            X_cf_input = X_cf.reshape(1, -1)
            if hasattr(self.model, 'predict_proba'):
                if hasattr(self.model, 'use_gnn') and self.model.use_gnn:
                    cf_prob = self.model.predict_proba(X_cf_input, gnn_proba=gnn_proba_array)[0]
                else:
                    cf_prob = self.model.predict_proba(X_cf_input)[0]
                if isinstance(cf_prob, np.ndarray):
                    cf_prob = cf_prob if len(cf_prob.shape) == 0 else cf_prob
                else:
                    cf_prob = float(cf_prob)
            else:
                X_cf_scaled = self.scaler.transform(X_cf_input)
                cf_prob = self.model.predict_proba(X_cf_scaled)[0, 1]

            delta = base_prob - cf_prob

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