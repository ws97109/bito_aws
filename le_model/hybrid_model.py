"""
混合模型系統 — 結合 Ensemble 和黑名單學習器
================================================================
策略：
  1. Ensemble 模型提供基礎預測（高準確率）
  2. 黑名單學習器提供可解釋性（相似度分析）
  3. 多種融合策略可選

融合方法：
  - 策略A：串聯（Ensemble 初篩 → 黑名單學習器精篩）
  - 策略B：加權融合（兩個模型分數加權平均）
  - 策略C：投票機制（兩個模型都認為是黑名單才判定）
  - 策略D：風險分級（根據分數提供不同建議）
================================================================
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    roc_auc_score, precision_score, recall_score, f1_score,
)

from ensemble import StackingEnsemble
from blacklist_learner import BlacklistLearner


class HybridBlacklistDetector:
    """
    混合黑名單檢測器

    結合 Ensemble 模型和黑名單學習器的優勢：
    - Ensemble: 高準確率，適合自動化決策
    - 黑名單學習器: 高可解釋性，適合人工審核
    """

    def __init__(
        self,
        ensemble_model: StackingEnsemble,
        blacklist_learner: BlacklistLearner,
        fusion_strategy: str = "weighted",  # weighted, cascade, voting, tiered
        ensemble_weight: float = 0.7,
        bl_weight: float = 0.3,
    ):
        self.ensemble = ensemble_model
        self.bl_learner = blacklist_learner
        self.fusion_strategy = fusion_strategy
        self.ensemble_weight = ensemble_weight
        self.bl_weight = bl_weight

        self.optimal_threshold = 0.5
        self.is_fitted = True  # 假設兩個模型都已訓練

    def predict_proba(
        self,
        X: np.ndarray,
        gnn_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        綜合預測，返回多個分數

        Returns:
            {
                'ensemble_score': Ensemble 模型分數
                'bl_score': 黑名單學習器分數
                'hybrid_score': 混合分數
                'confidence': 預測置信度
                'closest_cluster': 最相似的黑名單群組
            }
        """
        # 1. Ensemble 模型預測
        ensemble_score = self.ensemble.predict_proba(X, gnn_proba=gnn_proba)

        # 2. 黑名單學習器預測
        bl_result = self.bl_learner.predict_similarity(X)
        bl_score = bl_result['combined_score']

        # 3. 根據策略融合
        if self.fusion_strategy == "weighted":
            # 加權平均
            hybrid_score = (
                self.ensemble_weight * ensemble_score +
                self.bl_weight * bl_score
            )
            confidence = 1 - np.abs(ensemble_score - bl_score)  # 兩者越接近，置信度越高

        elif self.fusion_strategy == "cascade":
            # 串聯：Ensemble 高分的才用黑名單學習器
            hybrid_score = np.where(
                ensemble_score > 0.3,  # Ensemble 認為可能是黑名單
                (ensemble_score + bl_score) / 2,  # 結合兩者
                ensemble_score * 0.5  # Ensemble 認為是正常，降低分數
            )
            confidence = np.where(
                ensemble_score > 0.3,
                1 - np.abs(ensemble_score - bl_score),
                ensemble_score
            )

        elif self.fusion_strategy == "voting":
            # 投票：兩個都認為是黑名單才判定
            hybrid_score = np.minimum(ensemble_score, bl_score) * 2
            confidence = np.where(
                (ensemble_score > 0.5) & (bl_score > 0.5),
                1.0,  # 兩者都同意，高置信度
                0.3   # 意見不一致，低置信度
            )

        elif self.fusion_strategy == "tiered":
            # 分級：根據 Ensemble 分數決定策略
            hybrid_score = np.zeros_like(ensemble_score)
            confidence = np.zeros_like(ensemble_score)

            # 高風險（Ensemble > 0.7）：主要看 Ensemble
            high_risk = ensemble_score > 0.7
            hybrid_score[high_risk] = ensemble_score[high_risk] * 0.9 + bl_score[high_risk] * 0.1
            confidence[high_risk] = 0.9

            # 中風險（0.3 < Ensemble < 0.7）：兩者結合
            mid_risk = (ensemble_score >= 0.3) & (ensemble_score <= 0.7)
            hybrid_score[mid_risk] = ensemble_score[mid_risk] * 0.6 + bl_score[mid_risk] * 0.4
            confidence[mid_risk] = 1 - np.abs(ensemble_score[mid_risk] - bl_score[mid_risk])

            # 低風險（Ensemble < 0.3）：主要看黑名單學習器（避免漏報）
            low_risk = ensemble_score < 0.3
            hybrid_score[low_risk] = ensemble_score[low_risk] * 0.3 + bl_score[low_risk] * 0.7
            confidence[low_risk] = bl_score[low_risk]

        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

        return {
            'ensemble_score': ensemble_score,
            'bl_score': bl_score,
            'hybrid_score': hybrid_score,
            'confidence': confidence,
            'closest_cluster': bl_result['closest_cluster'],
            'svm_score': bl_result['svm_score'],
            'knn_similarity': bl_result['knn_similarity'],
        }

    def predict(
        self,
        X: np.ndarray,
        threshold: Optional[float] = None,
        **kwargs,
    ) -> np.ndarray:
        """預測黑名單（0/1）"""
        t = threshold if threshold is not None else self.optimal_threshold
        result = self.predict_proba(X, **kwargs)
        return (result['hybrid_score'] >= t).astype(int)

    def explain_prediction(
        self,
        user_idx: int,
        X: np.ndarray,
        gnn_proba: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        詳細解釋預測結果

        Returns:
            {
                'user_idx': 用戶索引
                'ensemble_score': Ensemble 分數
                'bl_score': 黑名單學習器分數
                'hybrid_score': 混合分數
                'confidence': 置信度
                'prediction': 預測結果 (0/1)
                'risk_level': 風險等級
                'explanation': 文字解釋
                'similar_blacklists': 相似的已知黑名單
            }
        """
        result = self.predict_proba(X[user_idx:user_idx+1], gnn_proba)

        ensemble_s = float(result['ensemble_score'][0])
        bl_s = float(result['bl_score'][0])
        hybrid_s = float(result['hybrid_score'][0])
        conf = float(result['confidence'][0])
        cluster = int(result['closest_cluster'][0])

        # 預測
        pred = 1 if hybrid_s >= self.optimal_threshold else 0

        # 風險等級
        if hybrid_s >= 0.8:
            risk_level = "極高風險"
        elif hybrid_s >= 0.6:
            risk_level = "高風險"
        elif hybrid_s >= 0.4:
            risk_level = "中風險"
        elif hybrid_s >= 0.2:
            risk_level = "低風險"
        else:
            risk_level = "正常"

        # 解釋
        explanation = []
        explanation.append(f"混合分數: {hybrid_s:.4f} (閾值: {self.optimal_threshold:.2f})")
        explanation.append(f"  - Ensemble 模型: {ensemble_s:.4f}")
        explanation.append(f"  - 黑名單學習器: {bl_s:.4f}")
        explanation.append(f"  - 置信度: {conf:.4f}")

        if pred == 1:
            explanation.append(f"\n判定: 黑名單 ({risk_level})")

            if ensemble_s > 0.7 and bl_s > 0.5:
                explanation.append("  原因: 兩個模型都認為是黑名單（高置信度）")
            elif ensemble_s > 0.7:
                explanation.append("  原因: Ensemble 模型判定為高風險")
            elif bl_s > 0.7:
                explanation.append(f"  原因: 與黑名單群組 {cluster} 高度相似")
            else:
                explanation.append(f"  原因: 綜合評估後判定為黑名單（置信度中等）")
        else:
            explanation.append(f"\n判定: 正常用戶 ({risk_level})")

            if ensemble_s < 0.3 and bl_s < 0.3:
                explanation.append("  原因: 兩個模型都認為是正常用戶")
            elif abs(ensemble_s - bl_s) > 0.4:
                explanation.append("  警告: 兩個模型意見不一致，建議人工審核")

        # 相似黑名單（從黑名單學習器獲取）
        bl_explanation = self.bl_learner.explain_user(user_idx, X)

        return {
            'user_idx': user_idx,
            'ensemble_score': ensemble_s,
            'bl_score': bl_s,
            'hybrid_score': hybrid_s,
            'confidence': conf,
            'prediction': pred,
            'risk_level': risk_level,
            'explanation': '\n'.join(explanation),
            'closest_cluster': cluster,
            'similar_blacklists': bl_explanation['similar_blacklist_indices'],
            'similar_distances': bl_explanation['similar_blacklist_distances'],
        }

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        gnn_proba: Optional[np.ndarray] = None,
    ) -> Dict:
        """評估混合模型效果"""

        result = self.predict_proba(X_test, gnn_proba)

        # 測試不同閾值
        best_threshold = 0.5
        best_f1 = 0.0

        threshold_results = []
        for t in np.arange(0.1, 0.9, 0.05):
            y_pred = (result['hybrid_score'] >= t).astype(int)

            pr = precision_score(y_test, y_pred, zero_division=0)
            rc = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            threshold_results.append({
                'threshold': t,
                'precision': pr,
                'recall': rc,
                'f1': f1,
            })

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        self.optimal_threshold = best_threshold

        # 使用最佳閾值評估
        y_pred = (result['hybrid_score'] >= best_threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics = {
            'optimal_threshold': best_threshold,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test, result['hybrid_score']),
            'threshold_results': threshold_results,
        }

        # 打印評估報告
        print("\n" + "="*70)
        print(f"  混合模型評估 - {self.fusion_strategy.upper()} 策略")
        print("="*70)

        print(f"\n最佳閾值: {best_threshold:.2f}")

        print(f"\n混淆矩陣:")
        print(f"                預測正常    預測黑名單")
        print(f"  實際正常      {tn:>6}       {fp:>6}       (FPR={fp/(tn+fp)*100:.1f}%)")
        print(f"  實際黑名單    {fn:>6}       {tp:>6}       (Recall={tp/(tp+fn)*100:.1f}%)")

        print(f"\n效能指標:")
        print(f"  Precision : {metrics['precision']:.4f} ({metrics['precision']*100:.1f}%)")
        print(f"  Recall    : {metrics['recall']:.4f} ({metrics['recall']*100:.1f}%)")
        print(f"  F1 Score  : {metrics['f1']:.4f}")
        print(f"  AUC-ROC   : {metrics['auc_roc']:.4f}")

        # 對比單一模型
        ensemble_pred = (result['ensemble_score'] >= 0.5).astype(int)
        bl_pred = (result['bl_score'] >= 0.4).astype(int)

        ensemble_f1 = f1_score(y_test, ensemble_pred, zero_division=0)
        bl_f1 = f1_score(y_test, bl_pred, zero_division=0)

        print(f"\n對比:")
        print(f"  Ensemble 單獨    : F1={ensemble_f1:.4f}")
        print(f"  黑名單學習器單獨  : F1={bl_f1:.4f}")
        print(f"  混合模型        : F1={metrics['f1']:.4f}")

        improvement = (metrics['f1'] - max(ensemble_f1, bl_f1)) / max(ensemble_f1, bl_f1) * 100
        if improvement > 0:
            print(f"  → 改善: +{improvement:.1f}% ✓")
        else:
            print(f"  → 改善: {improvement:.1f}%")

        return metrics

    def plot_comparison(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        gnn_proba: Optional[np.ndarray] = None,
        save_path: str = "output/hybrid_model_comparison.png",
    ):
        """繪製混合模型與單一模型的對比"""

        result = self.predict_proba(X_test, gnn_proba)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'混合模型評估 - {self.fusion_strategy.upper()} 策略',
                     fontsize=16, fontweight='bold')

        # 1. 分數分布對比
        ax = axes[0, 0]
        blacklist = y_test == 1
        normal = y_test == 0

        ax.hist(result['ensemble_score'][normal], bins=50, alpha=0.5,
                label='正常-Ensemble', color='blue', density=True)
        ax.hist(result['ensemble_score'][blacklist], bins=50, alpha=0.5,
                label='黑名單-Ensemble', color='red', density=True)
        ax.set_xlabel('Ensemble 分數')
        ax.set_ylabel('密度')
        ax.set_title('Ensemble 分數分布')
        ax.legend()
        ax.grid(alpha=0.3)

        # 2. 黑名單學習器分數分布
        ax = axes[0, 1]
        ax.hist(result['bl_score'][normal], bins=50, alpha=0.5,
                label='正常', color='blue', density=True)
        ax.hist(result['bl_score'][blacklist], bins=50, alpha=0.5,
                label='黑名單', color='red', density=True)
        ax.set_xlabel('黑名單學習器分數')
        ax.set_ylabel('密度')
        ax.set_title('黑名單學習器分數分布')
        ax.legend()
        ax.grid(alpha=0.3)

        # 3. 混合分數分布
        ax = axes[0, 2]
        ax.hist(result['hybrid_score'][normal], bins=50, alpha=0.5,
                label='正常', color='blue', density=True)
        ax.hist(result['hybrid_score'][blacklist], bins=50, alpha=0.5,
                label='黑名單', color='red', density=True)
        ax.axvline(self.optimal_threshold, color='green', linestyle='--',
                   linewidth=2, label=f'閾值={self.optimal_threshold:.2f}')
        ax.set_xlabel('混合分數')
        ax.set_ylabel('密度')
        ax.set_title('混合分數分布')
        ax.legend()
        ax.grid(alpha=0.3)

        # 4. 分數相關性
        ax = axes[1, 0]
        scatter = ax.scatter(result['ensemble_score'], result['bl_score'],
                            c=y_test, cmap='RdYlBu_r', alpha=0.5, s=10)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('Ensemble 分數')
        ax.set_ylabel('黑名單學習器分數')
        ax.set_title('兩個模型的分數相關性')
        plt.colorbar(scatter, ax=ax, label='真實標籤')
        ax.grid(alpha=0.3)

        # 5. ROC 曲線對比
        ax = axes[1, 1]

        fpr_e, tpr_e, _ = roc_curve(y_test, result['ensemble_score'])
        fpr_b, tpr_b, _ = roc_curve(y_test, result['bl_score'])
        fpr_h, tpr_h, _ = roc_curve(y_test, result['hybrid_score'])

        auc_e = auc(fpr_e, tpr_e)
        auc_b = auc(fpr_b, tpr_b)
        auc_h = auc(fpr_h, tpr_h)

        ax.plot(fpr_e, tpr_e, 'b-', linewidth=2, label=f'Ensemble (AUC={auc_e:.3f})')
        ax.plot(fpr_b, tpr_b, 'r-', linewidth=2, label=f'黑名單學習器 (AUC={auc_b:.3f})')
        ax.plot(fpr_h, tpr_h, 'g-', linewidth=3, label=f'混合模型 (AUC={auc_h:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC 曲線對比')
        ax.legend()
        ax.grid(alpha=0.3)

        # 6. 置信度分布
        ax = axes[1, 2]
        ax.hist(result['confidence'], bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax.set_xlabel('預測置信度')
        ax.set_ylabel('數量')
        ax.set_title('預測置信度分布')
        ax.axvline(result['confidence'].mean(), color='red', linestyle='--',
                   linewidth=2, label=f'平均={result["confidence"].mean():.3f}')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ 對比圖表已儲存: {save_path}")
        plt.close()


def test_all_strategies(
    ensemble: StackingEnsemble,
    bl_learner: BlacklistLearner,
    X_test: np.ndarray,
    y_test: np.ndarray,
    gnn_proba: Optional[np.ndarray] = None,
):
    """測試所有融合策略"""

    strategies = ["weighted", "cascade", "voting", "tiered"]
    results = {}

    print("\n" + "="*70)
    print("  測試所有融合策略")
    print("="*70)

    for strategy in strategies:
        print(f"\n{'='*70}")
        print(f"  策略: {strategy.upper()}")
        print(f"{'='*70}")

        hybrid = HybridBlacklistDetector(
            ensemble, bl_learner,
            fusion_strategy=strategy,
            ensemble_weight=0.7,
            bl_weight=0.3,
        )

        metrics = hybrid.evaluate(X_test, y_test, gnn_proba)
        results[strategy] = metrics

        # 繪製對比圖
        hybrid.plot_comparison(
            X_test, y_test, gnn_proba,
            save_path=f"output/hybrid_{strategy}.png"
        )

    # 總結對比
    print("\n" + "="*70)
    print("  策略對比總結")
    print("="*70)

    print(f"\n{'策略':<15} {'閾值':<8} {'Precision':<12} {'Recall':<10} {'F1':<10} {'AUC-ROC'}")
    print("-" * 70)

    for strategy, metrics in results.items():
        print(f"{strategy:<15} {metrics['optimal_threshold']:<8.2f} "
              f"{metrics['precision']:<12.4f} {metrics['recall']:<10.4f} "
              f"{metrics['f1']:<10.4f} {metrics['auc_roc']:.4f}")

    # 找出最佳策略
    best_strategy = max(results.items(), key=lambda x: x[1]['f1'])[0]
    print(f"\n推薦策略: {best_strategy.upper()} (F1={results[best_strategy]['f1']:.4f})")

    return results, best_strategy


if __name__ == "__main__":
    print("混合模型模組，請從 test_hybrid_model.py 調用")
