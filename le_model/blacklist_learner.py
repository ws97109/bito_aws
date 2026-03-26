"""
黑名單特徵學習器 — 專注於學習黑名單行為模式
================================================================
核心概念：
  與傳統二分類不同，本模型專注於「學習黑名單長什麼樣子」，
  然後判斷新用戶與黑名單的相似度

策略組合：
  1. One-Class SVM：只用黑名單訓練，學習黑名單分布邊界
  2. 孤立森林：異常檢測，黑名單是「異常」
  3. K-Means 聚類：找出黑名單的子群組
  4. 原型網絡：學習黑名單的「典型特徵向量」
  5. 對比學習：拉近黑名單內部距離，推遠與正常用戶距離

優勢：
  - 不依賴大量正樣本標註
  - 可以發現「新型態」黑名單（與已知黑名單相似）
  - 可解釋性強（可視化與哪種黑名單類型相似）
================================================================
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    pairwise_distances, roc_auc_score,
)
import warnings
warnings.filterwarnings("ignore")


class BlacklistLearner:
    """
    黑名單特徵學習器

    工作流程：
    1. fit() - 只用黑名單樣本訓練，學習黑名單的多種模式
    2. predict_similarity() - 計算新用戶與黑名單的相似度
    3. explain() - 解釋為什麼判定為黑名單（與哪個子群相似）
    """

    def __init__(
        self,
        n_clusters: int = 5,           # 黑名單子群組數量
        contamination: float = 0.05,   # 異常檢測的容忍度
        n_neighbors: int = 10,         # K近鄰數量
    ):
        self.n_clusters = n_clusters
        self.contamination = contamination
        self.n_neighbors = n_neighbors

        # 模型組件
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # 保留 95% 方差
        self.kmeans = None
        self.one_class_svm = None
        self.isolation_forest = None
        self.knn = None

        # 黑名單樣本存儲
        self.blacklist_features = None  # 原始特徵
        self.blacklist_embeddings = None  # PCA 降維後
        self.cluster_centers = None  # 聚類中心
        self.cluster_labels = None  # 每個黑名單的群組標籤

        self.is_fitted = False

    def fit(
        self,
        X_all: np.ndarray,
        y_all: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "BlacklistLearner":
        """
        訓練黑名單學習器（只用黑名單樣本）

        Args:
            X_all: 全部樣本特徵
            y_all: 全部標籤（0=正常, 1=黑名單）
            feature_names: 特徵名稱（用於解釋）
        """
        print("\n" + "="*60)
        print("  黑名單特徵學習器 - 訓練")
        print("="*60)

        # 1. 提取黑名單樣本
        blacklist_mask = (y_all == 1)
        X_blacklist = X_all[blacklist_mask]
        n_blacklist = len(X_blacklist)

        print(f"\n[Step 1] 提取黑名單樣本")
        print(f"  總樣本數：{len(X_all):,}")
        print(f"  黑名單數：{n_blacklist:,} ({n_blacklist/len(X_all)*100:.2f}%)")

        if n_blacklist < 10:
            raise ValueError(f"黑名單樣本太少（{n_blacklist}），無法訓練！至少需要 10 個")

        # 2. 標準化 + PCA 降維
        print(f"\n[Step 2] 特徵標準化與降維")
        X_blacklist_scaled = self.scaler.fit_transform(X_blacklist)
        self.blacklist_features = X_blacklist_scaled.copy()

        # PCA（如果特徵數太多）
        if X_blacklist_scaled.shape[1] > 50:
            self.blacklist_embeddings = self.pca.fit_transform(X_blacklist_scaled)
            print(f"  PCA: {X_blacklist_scaled.shape[1]} 維 → {self.blacklist_embeddings.shape[1]} 維")
        else:
            self.blacklist_embeddings = X_blacklist_scaled
            print(f"  保持原始維度：{X_blacklist_scaled.shape[1]} 維")

        # 3. K-Means 聚類 - 發現黑名單子群組
        print(f"\n[Step 3] K-Means 聚類（k={self.n_clusters}）")
        self.kmeans = KMeans(
            n_clusters=min(self.n_clusters, n_blacklist),
            random_state=42,
            n_init=10,
        )
        self.cluster_labels = self.kmeans.fit_predict(self.blacklist_embeddings)
        self.cluster_centers = self.kmeans.cluster_centers_

        # 評估聚類品質
        if n_blacklist >= self.n_clusters + 1:
            sil_score = silhouette_score(self.blacklist_embeddings, self.cluster_labels)
            db_score = davies_bouldin_score(self.blacklist_embeddings, self.cluster_labels)
            print(f"  Silhouette Score: {sil_score:.3f} (越高越好, 範圍 [-1, 1])")
            print(f"  Davies-Bouldin Index: {db_score:.3f} (越低越好)")

        # 顯示每個群組的大小
        print(f"\n  黑名單子群組分布：")
        for i in range(self.n_clusters):
            count = (self.cluster_labels == i).sum()
            print(f"    群組 {i}: {count} 個黑名單 ({count/n_blacklist*100:.1f}%)")

        # 4. One-Class SVM - 學習黑名單的邊界
        print(f"\n[Step 4] One-Class SVM 訓練")
        self.one_class_svm = OneClassSVM(
            kernel='rbf',
            gamma='auto',
            nu=0.05,  # 允許 5% 的黑名單被視為異常值
        )
        self.one_class_svm.fit(self.blacklist_embeddings)
        in_boundary = (self.one_class_svm.predict(self.blacklist_embeddings) == 1).sum()
        print(f"  {in_boundary}/{n_blacklist} 個黑名單在學習邊界內 ({in_boundary/n_blacklist*100:.1f}%)")

        # 5. Isolation Forest - 異常檢測視角
        print(f"\n[Step 5] Isolation Forest 訓練")
        self.isolation_forest = IsolationForest(
            n_estimators=200,
            contamination=self.contamination,
            random_state=42,
        )
        self.isolation_forest.fit(X_blacklist_scaled)

        # 6. K-NN - 最近鄰相似度
        print(f"\n[Step 6] K-NN 索引建立（k={self.n_neighbors}）")
        self.knn = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, n_blacklist),
            metric='euclidean',
        )
        self.knn.fit(self.blacklist_embeddings)

        self.feature_names = feature_names
        self.is_fitted = True

        print(f"\n{'='*60}")
        print(f"  ✓ 訓練完成！已學習 {n_blacklist} 個黑名單的特徵模式")
        print(f"{'='*60}\n")

        return self

    def predict_similarity(
        self,
        X: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        計算新用戶與黑名單的相似度（多種視角）

        Returns:
            {
                'combined_score': 綜合相似度分數 [0, 1]
                'cluster_similarity': 與各群組的相似度
                'svm_score': One-Class SVM 分數
                'iso_score': Isolation Forest 異常分數
                'knn_distance': 與最近黑名單的距離
                'is_blacklist': 綜合判定（0/1）
            }
        """
        if not self.is_fitted:
            raise RuntimeError("模型未訓練，請先調用 fit()")

        # 1. 標準化 + 降維
        X_scaled = self.scaler.transform(X)
        if hasattr(self.pca, 'components_'):
            X_emb = self.pca.transform(X_scaled)
        else:
            X_emb = X_scaled

        n_samples = len(X)

        # 2. One-Class SVM 分數（距離邊界）
        svm_decision = self.one_class_svm.decision_function(X_emb)
        svm_score = 1 / (1 + np.exp(-svm_decision))  # sigmoid 轉換到 [0, 1]

        # 3. Isolation Forest 異常分數
        iso_score = -self.isolation_forest.score_samples(X_scaled)
        iso_score = (iso_score - iso_score.min()) / (iso_score.max() - iso_score.min() + 1e-9)

        # 4. K-NN 距離（與最近黑名單的平均距離）
        knn_distances, knn_indices = self.knn.kneighbors(X_emb)
        avg_knn_dist = knn_distances.mean(axis=1)
        # 距離轉換為相似度（距離越小，相似度越高）
        max_dist = np.percentile(avg_knn_dist, 95)
        knn_similarity = 1 - np.clip(avg_knn_dist / (max_dist + 1e-9), 0, 1)

        # 5. 聚類中心相似度
        cluster_distances = pairwise_distances(X_emb, self.cluster_centers, metric='euclidean')
        cluster_similarity = 1 / (1 + cluster_distances)  # 距離轉相似度
        closest_cluster = cluster_distances.argmin(axis=1)

        # 6. 綜合分數（加權平均）
        combined_score = (
            svm_score * 0.30 +
            iso_score * 0.25 +
            knn_similarity * 0.25 +
            cluster_similarity[np.arange(n_samples), closest_cluster] * 0.20
        )

        return {
            'combined_score': combined_score,
            'cluster_similarity': cluster_similarity,
            'closest_cluster': closest_cluster,
            'svm_score': svm_score,
            'iso_score': iso_score,
            'knn_distance': avg_knn_dist,
            'knn_similarity': knn_similarity,
        }

    def predict(
        self,
        X: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        預測是否為黑名單（0/1）

        Args:
            X: 特徵矩陣
            threshold: 相似度閾值（>= 則判定為黑名單）
        """
        result = self.predict_similarity(X)
        return (result['combined_score'] >= threshold).astype(int)

    def explain_user(
        self,
        user_idx: int,
        X: np.ndarray,
        top_k: int = 5,
    ) -> Dict:
        """
        解釋為什麼判定某用戶為黑名單

        Returns:
            {
                'user_idx': 用戶索引
                'combined_score': 綜合相似度
                'closest_cluster': 最相似的黑名單群組
                'similar_blacklists': 最相似的 k 個已知黑名單索引
                'feature_importance': 哪些特徵最相似
            }
        """
        result = self.predict_similarity(X[user_idx:user_idx+1])

        # 標準化 + 降維
        X_scaled = self.scaler.transform(X[user_idx:user_idx+1])
        if hasattr(self.pca, 'components_'):
            X_emb = self.pca.transform(X_scaled)
        else:
            X_emb = X_scaled

        # 找最相似的 k 個黑名單
        distances, indices = self.knn.kneighbors(X_emb, n_neighbors=min(top_k, len(self.blacklist_embeddings)))

        # 特徵差異分析
        similar_features = self.blacklist_features[indices[0]].mean(axis=0)
        feature_diff = np.abs(X_scaled[0] - similar_features)

        return {
            'user_idx': user_idx,
            'combined_score': float(result['combined_score'][0]),
            'closest_cluster': int(result['closest_cluster'][0]),
            'svm_score': float(result['svm_score'][0]),
            'iso_score': float(result['iso_score'][0]),
            'knn_similarity': float(result['knn_similarity'][0]),
            'similar_blacklist_indices': indices[0].tolist(),
            'similar_blacklist_distances': distances[0].tolist(),
            'feature_difference': feature_diff,
        }

    def visualize_blacklist_space(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_path: str = "output/blacklist_space.png",
    ):
        """
        可視化黑名單特徵空間（2D PCA）
        """
        # 測試集標準化 + 降維
        X_test_scaled = self.scaler.transform(X_test)
        if hasattr(self.pca, 'components_'):
            X_test_emb = self.pca.transform(X_test_scaled)
        else:
            X_test_emb = X_test_scaled

        # 降至 2D
        pca_2d = PCA(n_components=2)
        blacklist_2d = pca_2d.fit_transform(self.blacklist_embeddings)
        test_2d = pca_2d.transform(X_test_emb)
        centers_2d = pca_2d.transform(self.cluster_centers)

        # 繪圖
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 左圖：黑名單聚類
        ax = axes[0]
        scatter = ax.scatter(
            blacklist_2d[:, 0], blacklist_2d[:, 1],
            c=self.cluster_labels, cmap='tab10',
            alpha=0.6, s=50, edgecolors='k', linewidth=0.5,
        )
        ax.scatter(
            centers_2d[:, 0], centers_2d[:, 1],
            c='red', marker='X', s=300, edgecolors='black', linewidth=2,
            label='群組中心'
        )
        ax.set_title('黑名單聚類分布（訓練集）', fontsize=14, fontweight='bold')
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='群組')

        # 右圖：測試集分布
        ax = axes[1]
        ax.scatter(
            blacklist_2d[:, 0], blacklist_2d[:, 1],
            c='gray', alpha=0.3, s=30, label='已知黑名單'
        )
        ax.scatter(
            centers_2d[:, 0], centers_2d[:, 1],
            c='red', marker='X', s=300, edgecolors='black', linewidth=2,
            label='群組中心'
        )

        # 測試集（區分黑名單/正常）
        test_blacklist = y_test == 1
        test_normal = y_test == 0
        ax.scatter(
            test_2d[test_normal, 0], test_2d[test_normal, 1],
            c='blue', alpha=0.5, s=30, label='測試集-正常', marker='o'
        )
        ax.scatter(
            test_2d[test_blacklist, 0], test_2d[test_blacklist, 1],
            c='orange', alpha=0.7, s=50, label='測試集-黑名單', marker='^',
            edgecolors='black', linewidth=0.5
        )

        ax.set_title('測試集在黑名單空間的分布', fontsize=14, fontweight='bold')
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n  ✓ 可視化已儲存：{save_path}")
        plt.close()


# ═══════════════════════════════════════════════════════════
# 使用範例
# ═══════════════════════════════════════════════════════════

def demo_blacklist_learner(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Optional[List[str]] = None,
    output_dir: str = "output",
):
    """
    完整的黑名單學習器示範
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 1. 訓練黑名單學習器
    learner = BlacklistLearner(
        n_clusters=5,
        contamination=0.05,
        n_neighbors=10,
    )
    learner.fit(X_train, y_train, feature_names)

    # 2. 預測測試集
    print("\n" + "="*60)
    print("  測試集預測")
    print("="*60)

    result = learner.predict_similarity(X_test)
    y_pred = learner.predict(X_test, threshold=0.5)

    # 3. 評估
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

    print("\n混淆矩陣：")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("\n分類報告：")
    print(classification_report(y_test, y_pred, target_names=['正常', '黑名單']))

    auc = roc_auc_score(y_test, result['combined_score'])
    print(f"\nAUC-ROC: {auc:.4f}")

    # 4. 可視化
    learner.visualize_blacklist_space(
        X_test, y_test,
        save_path=os.path.join(output_dir, "blacklist_space.png")
    )

    # 5. 解釋高風險用戶
    print("\n" + "="*60)
    print("  高風險用戶解釋（Top 3）")
    print("="*60)

    top_risk_idx = result['combined_score'].argsort()[-3:][::-1]
    for rank, idx in enumerate(top_risk_idx, 1):
        explanation = learner.explain_user(idx, X_test)
        print(f"\n第 {rank} 高風險用戶（測試集 idx={idx}）")
        print(f"  真實標籤：{'黑名單' if y_test[idx] == 1 else '正常'}")
        print(f"  綜合相似度：{explanation['combined_score']:.4f}")
        print(f"  最相似群組：群組 {explanation['closest_cluster']}")
        print(f"  SVM 分數：{explanation['svm_score']:.4f}")
        print(f"  異常分數：{explanation['iso_score']:.4f}")
        print(f"  KNN 相似度：{explanation['knn_similarity']:.4f}")
        print(f"  最相似的黑名單：{explanation['similar_blacklist_indices'][:3]}")

    return learner, result


if __name__ == "__main__":
    print("這是黑名單學習器模組，請從 main.py 調用")
    print("\n使用方式：")
    print("  from blacklist_learner import BlacklistLearner, demo_blacklist_learner")
    print("  learner, result = demo_blacklist_learner(X_train, y_train, X_test, y_test)")
