"""
Anomaly Detection Module
非監督式異常偵測：Isolation Forest + HBOS + LOF
產出 3 個異常分數作為監督式 Ensemble 的額外特徵

注意：必須在 CV fold 內部訓練，避免資料洩漏
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.lof import LOF


class AnomalyFeatureExtractor:
    """
    在 CV fold 內部訓練三種異常偵測模型，
    產出 if_score / hbos_score / lof_score 作為新特徵。
    """

    MODELS = {
        "if":   lambda c: IForest(n_estimators=200, contamination=c, random_state=42, n_jobs=-1),
        "hbos": lambda c: HBOS(n_bins=20, contamination=c),
        "lof":  lambda c: LOF(n_neighbors=20, contamination=c),
    }

    def __init__(self, contamination: float = 0.033):
        """
        contamination: 預估異常比例，對應黑名單比例 3.2%
        """
        self.contamination = contamination
        self.scaler = None
        self.models = {}
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> "AnomalyFeatureExtractor":
        """用訓練集 fit scaler + 三個異常偵測模型"""
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.models = {}
        for name, builder in self.MODELS.items():
            model = builder(self.contamination)
            model.fit(X_scaled)
            self.models[name] = model
            print(f"    {name.upper():>4} 訓練完成")

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """產出 3 個異常分數 (n_samples, 3)"""
        if not self.is_fitted:
            raise RuntimeError("AnomalyFeatureExtractor 尚未 fit")

        X_scaled = self.scaler.transform(X)

        scores = []
        for name in ["if", "hbos", "lof"]:
            model = self.models[name]
            # decision_function: 越高越異常
            score = model.decision_function(X_scaled)
            scores.append(score)

        return np.column_stack(scores)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """fit + transform 一步完成（用於訓練集）"""
        self.fit(X)

        X_scaled = self.scaler.transform(X)
        scores = []
        for name in ["if", "hbos", "lof"]:
            model = self.models[name]
            # 訓練集用 decision_scores_（fit 時已計算）
            scores.append(model.decision_scores_)

        return np.column_stack(scores)

    @staticmethod
    def get_feature_names() -> list:
        return ["if_score", "hbos_score", "lof_score"]


def add_anomaly_scores_to_splits(
    X_train: np.ndarray,
    X_test: np.ndarray,
    contamination: float = 0.033,
) -> tuple:
    """
    便捷函數：對 train/test split 加上異常分數

    Returns:
        X_train_aug (n_train, d+3), X_test_aug (n_test, d+3), extractor
    """
    print("  [非監督異常偵測] 訓練 Isolation Forest / HBOS / LOF ...")
    extractor = AnomalyFeatureExtractor(contamination=contamination)

    train_scores = extractor.fit_transform(X_train)
    test_scores = extractor.transform(X_test)

    X_train_aug = np.hstack([X_train, train_scores])
    X_test_aug = np.hstack([X_test, test_scores])

    # 統計
    for i, name in enumerate(extractor.get_feature_names()):
        tr_mean = train_scores[:, i].mean()
        te_mean = test_scores[:, i].mean()
        print(f"    {name}: train_mean={tr_mean:.4f}, test_mean={te_mean:.4f}")

    print(f"  特徵維度：{X_train.shape[1]} → {X_train_aug.shape[1]} (+3 anomaly scores)")
    return X_train_aug, X_test_aug, extractor
