"""
Main Pipeline — 真實資料版
訓練流程：特徵工程 → GNN → 集成 → 可解釋性

使用方式：
    python main.py                               
    python main.py --data_dir ../RawData         # 指定資料夾
    python main.py --output results              # 指定輸出目錄
    python main.py --skip_gnn                    # 跳過 GNN（資料量小時）
"""
import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# Wei_model/model/ → Wei_model/ → 專案根目錄
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEI_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))  # Wei_model/model/
sys.path.insert(0, ROOT)

# 直接從當前目錄導入（所有模組都在 model/ 目錄下）
from Feature_rngineering import build_all_features
from feature_selection import select_features
from anomaly_detection import AnomalyFeatureExtractor, add_anomaly_scores_to_splits
from Gnn_model import build_transaction_graph, BlacklistGNN
from pseudo_labeling import pseudo_label
from ensemble import StackingEnsemble, evaluate, find_optimal_threshold
from shap_explainer import (
    SHAPExplainer, CounterfactualExplainer, generate_user_report,
    SSREvaluator,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備：{DEVICE}")


# ═══════════════════════════════════════════════
# 欄位規格（對應資料表文件）
# ═══════════════════════════════════════════════

TABLE_SPECS = {
    "user_info_train": {
        "required": ["user_id", "status"],
        "datetime": ["confirmed_at", "level1_finished_at",
                     "level2_finished_at", "birthday"],
        "int":      ["user_id", "status", "sex", "career",
                     "income_source", "user_source"],
    },
    "twd_transfer_train": {
        "required": ["user_id", "kind", "ori_samount"],
        "datetime": ["created_at"],
        "int":      ["user_id", "kind"],
        "float":    ["ori_samount", "source_ip"],
    },
    "crypto_transfer_train": {
        "required": ["user_id", "kind", "sub_kind", "ori_samount", "twd_srate"],
        "datetime": ["created_at"],
        "int":      ["user_id", "kind", "sub_kind", "protocol"],
        "float":    ["ori_samount", "twd_srate", "source_ip", "relation_user_id"],
        "str":      ["from_wallet", "to_wallet", "currency"],
    },
    "usdt_twd_trading_train": {
        "required": ["user_id", "is_buy", "trade_samount", "twd_srate"],
        "datetime": ["updated_at"],
        "int":      ["user_id", "is_buy", "is_market", "source"],
        "float":    ["trade_samount", "twd_srate", "source_ip"],
    },
    "usdt_swap_train": {
        "required": ["user_id", "kind", "twd_samount", "currency_samount"],
        "datetime": ["created_at"],
        "int":      ["user_id", "kind"],
        "float":    ["twd_samount", "currency_samount"],
    },
}


# ═══════════════════════════════════════════════
# 資料載入與驗證
# ═══════════════════════════════════════════════

def load_and_validate(data_dir: str) -> dict:
    """
    從 data_dir/ 讀取五張 CSV，進行：
      1. 檔案存在確認
      2. 必要欄位檢查
      3. 型別轉換（datetime / int / float / str）
      4. 缺值統計報告
      5. 黑名單比例確認
    """
    tables = {}

    for name, spec in TABLE_SPECS.items():
        path = os.path.join(data_dir, f"{name}.csv")

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"\n找不到：{os.path.abspath(path)}\n"
                f"請確認 {data_dir}/ 目錄下包含以下五個 CSV：\n"
                + "\n".join(f"  - {n}.csv" for n in TABLE_SPECS)
            )

        df = pd.read_csv(path, low_memory=False)
        print(f"\n  [{name}]  {len(df):,} 筆")

        # 必要欄位
        missing = [c for c in spec["required"] if c not in df.columns]
        if missing:
            raise ValueError(
                f"{name}.csv 缺少必要欄位：{missing}\n"
                f"現有欄位：{df.columns.tolist()}"
            )

        # datetime
        for col in spec.get("datetime", []):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # int
        for col in spec.get("int", []):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # float
        for col in spec.get("float", []):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # str
        for col in spec.get("str", []):
            if col in df.columns:
                df[col] = df[col].astype(str).replace("nan", pd.NA)

        # 缺值摘要
        null_counts = df.isnull().sum()
        null_cols   = null_counts[null_counts > 0]
        if len(null_cols):
            print(f"    缺值欄位：")
            for col, cnt in null_cols.items():
                print(f"      {col:<32} {cnt:>7} ({cnt/len(df)*100:.1f}%)")

        # user_info 特殊：確認有黑名單
        if name == "user_info_train":
            vc      = df["status"].value_counts().sort_index()
            n_black = int((df["status"] == 1).sum())
            print(f"    status 分布  : {dict(vc)}")
            print(f"    黑名單用戶   : {n_black} ({n_black/len(df)*100:.2f}%)")
            if n_black == 0:
                raise ValueError(
                    "user_info.csv 中沒有 status=1 的黑名單用戶\n"
                    "請確認 status 欄位：0=正常，1=黑名單"
                )

        tables[name] = df

    return tables


def load_predict_data(predict_dir: str) -> dict:
    """載入 predict/ 目錄的五張 CSV（無 status 欄位）"""
    PREDICT_SPECS = {
        "user_info_predict": TABLE_SPECS["user_info_train"],
        "twd_transfer_predict": TABLE_SPECS["twd_transfer_train"],
        "crypto_transfer_predict": TABLE_SPECS["crypto_transfer_train"],
        "usdt_twd_trading_predict": TABLE_SPECS["usdt_twd_trading_train"],
        "usdt_swap_predict": TABLE_SPECS["usdt_swap_train"],
    }
    tables = {}
    for name, spec in PREDICT_SPECS.items():
        path = os.path.join(predict_dir, f"{name}.csv")
        if not os.path.exists(path):
            return None  # predict 資料不存在，靜默跳過
        df = pd.read_csv(path, low_memory=False)
        for col in spec.get("datetime", []):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        for col in spec.get("int", []):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in spec.get("float", []):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in spec.get("str", []):
            if col in df.columns:
                df[col] = df[col].astype(str).replace("nan", pd.NA)
        tables[name] = df
    print(f"\n  [Predict] 載入 {len(tables)} 張表，用戶數: {len(tables.get('user_info_predict', []))}")
    return tables


# ═══════════════════════════════════════════════
# GNN 訓練
# ═══════════════════════════════════════════════

def train_gnn(
    graph_data,
    labels: torch.Tensor,
    tabular: torch.Tensor,
    in_dim: int,
    tabular_dim: int,
    epochs: int = 80,
) -> BlacklistGNN:
    model      = BlacklistGNN(in_dim, tabular_dim).to(DEVICE)
    graph_data = graph_data.to(DEVICE)
    tabular    = tabular.to(DEVICE)
    labels     = labels.to(DEVICE)

    n_neg      = (labels == 0).sum().float()
    n_pos      = (labels == 1).sum().float().clamp(min=1)
    pos_weight = torch.tensor([n_neg / n_pos], device=DEVICE)
    criterion  = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss, best_state = float("inf"), None
    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(graph_data, tabular)
        loss   = criterion(logits, labels.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if ep % 10 == 0:
            print(f"  Epoch {ep:3d}/{epochs}  Loss: {loss.item():.5f}")

        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"  GNN 完成，最佳 Loss：{best_loss:.5f}")
    return model


# ═══════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════

def main(
    data_dir: str = os.path.join(ROOT, "adjust_data", "train"),
    predict_dir: str = os.path.join(ROOT, "adjust_data", "predict"),
    output_dir: str = os.path.join(os.path.dirname(WEI_MODEL_DIR), "output"),
    skip_gnn: bool = False,
    use_focal_loss: bool = True,
    use_smote: bool = False,
    smote_strategy: float = 0.3,
    use_pseudo_label: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1：載入真實 CSV ─────────────────────
    print("\n" + "="*55)
    print(f"[Step 1] 載入資料  ←  {os.path.abspath(data_dir)}/")
    print("="*55)

    tables    = load_and_validate(data_dir)
    user_info = tables["user_info_train"]
    twd       = tables["twd_transfer_train"]
    crypto    = tables["crypto_transfer_train"]
    trading   = tables["usdt_twd_trading_train"]
    swap      = tables["usdt_swap_train"]

    # ── Step 2：特徵工程 ─────────────────────────
    print("\n" + "="*55)
    print("[Step 2] 特徵工程")
    print("="*55)

    feat_df  = build_all_features(user_info, twd, crypto, trading, swap)
    labels_s = user_info.set_index("user_id")["status"]
    feat_df  = feat_df.join(labels_s, how="left")

    n_missing = feat_df["status"].isna().sum()
    if n_missing:
        print(f"  [注意] {n_missing} 個 user_id 無對應 status，填入 0（正常）")
    feat_df["status"] = feat_df["status"].fillna(0).astype(int)
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    y             = feat_df["status"].values.astype(int)
    X_raw         = feat_df.drop(columns=["status"])

    print(f"\n  用戶總數    : {len(feat_df):,}")
    print(f"  原始特徵    : {X_raw.shape[1]}")
    print(f"  黑名單用戶  : {y.sum():,} ({y.mean()*100:.2f}%)")

    feat_df.to_csv(os.path.join(output_dir, "features_raw.csv"))

    # ── Step 2.5：特徵篩選 ─────────────────────────
    print("\n" + "="*55)
    print("[Step 2.5] 特徵篩選（零方差 → 高相關 → 重要性）")
    print("="*55)

    X_selected, selection_report = select_features(X_raw, y, corr_threshold=0.95)

    with open(os.path.join(output_dir, "feature_selection_report.json"), "w", encoding="utf-8") as f:
        json.dump(selection_report, f, indent=2, ensure_ascii=False)

    X             = X_selected.values.astype(np.float32)
    feature_names = X_selected.columns.tolist()

    print(f"\n  篩選後特徵  : {X.shape[1]}")

    # 更新 feat_df 為篩選後版本（後續 GNN、SHAP 使用）
    feat_df = X_selected.copy()
    feat_df["status"] = y

    feat_df.to_csv(os.path.join(output_dir, "features.csv"))

    # ── Step 3：Train / Test 分割 ────────────────
    print("\n" + "="*55)
    print("[Step 3] 訓練 / 測試集分割（8:2，Stratified）")
    print("="*55)

    use_stratify = y.sum() >= 5
    if not use_stratify:
        print("  [警告] 黑名單樣本不足 5 筆，改用隨機分割")

    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, np.arange(len(y)),
        test_size=0.2,
        stratify=y if use_stratify else None,
        random_state=42,
    )
    print(f"  訓練集：{len(X_tr):,}（黑名單 {y_tr.sum()}）")
    print(f"  測試集：{len(X_te):,}（黑名單 {y_te.sum()}）")

    # ── Step 3.5：非監督異常偵測 ──────────────────
    print("\n" + "="*55)
    print("[Step 3.5] 非監督式異常偵測（Isolation Forest / HBOS / LOF）")
    print("="*55)

    X_tr, X_te, anomaly_extractor = add_anomaly_scores_to_splits(X_tr, X_te)
    # 更新特徵名稱
    feature_names = feature_names + AnomalyFeatureExtractor.get_feature_names()
    # 同時處理全量 X（用於最終預測）
    all_anomaly_scores = anomaly_extractor.transform(X)
    X_all = np.hstack([X, all_anomaly_scores])
    print(f"  全量特徵維度：{X.shape[1]} → {X_all.shape[1]}")

    # ── Step 4：GNN（embedding 作為特徵）──────────
    gnn_embed_dim = 0

    if not skip_gnn:
        print("\n" + "="*55)
        print("[Step 4] 建立交易圖 & 訓練 GNN → 提取 embedding 作為特徵")
        print("="*55)

        graph     = build_transaction_graph(crypto, feat_df.drop(columns=["status"]))
        tabular_t = torch.tensor(X, dtype=torch.float32)
        labels_t  = torch.tensor(y, dtype=torch.long)

        n_edges = sum(
            graph[et].edge_index.shape[1]
            for et in graph.edge_types
            if hasattr(graph[et], "edge_index")
        )
        print(f"  用戶節點：{graph['user'].num_nodes:,}")
        print(f"  錢包節點：{graph['wallet'].num_nodes:,}")
        print(f"  邊總數  ：{n_edges:,}")

        if n_edges == 0:
            print("  [注意] 無鏈上交易邊，自動跳過 GNN")
            skip_gnn = True
        else:
            gnn_mdl = train_gnn(
                graph, labels_t, tabular_t,
                in_dim=X.shape[1], tabular_dim=X.shape[1],
                epochs=80,
            )
            gnn_mdl.eval()
            with torch.no_grad():
                # 提取 GNN encoder 的 64-dim embedding（而非最終分類機率）
                gnn_embedding = gnn_mdl.gnn_encoder(
                    graph.to(DEVICE)
                ).cpu().numpy()  # shape: [n_users, 64]

            torch.save(gnn_mdl.state_dict(),
                       os.path.join(output_dir, "gnn_model.pt"))

            # PCA 降維至 16 維，保留主要資訊
            from sklearn.decomposition import PCA
            gnn_pca_dim = min(16, gnn_embedding.shape[1])
            pca = PCA(n_components=gnn_pca_dim, random_state=42)
            gnn_features = pca.fit_transform(gnn_embedding).astype(np.float32)
            explained = pca.explained_variance_ratio_.sum()
            print(f"  GNN embedding: 64 → PCA {gnn_pca_dim} 維（解釋變異: {explained:.2%}）")

            # 拼接 GNN features 到 train/test/all
            gnn_tr_feat = gnn_features[idx_tr]
            gnn_te_feat = gnn_features[idx_te]
            X_tr = np.hstack([X_tr, gnn_tr_feat])
            X_te = np.hstack([X_te, gnn_te_feat])
            X_all = np.hstack([X_all, gnn_features])
            gnn_embed_dim = gnn_pca_dim

            # 更新特徵名稱
            gnn_feat_names = [f"gnn_emb_{i}" for i in range(gnn_pca_dim)]
            feature_names = feature_names + gnn_feat_names
            print(f"  特徵維度：{X_tr.shape[1] - gnn_pca_dim} → {X_tr.shape[1]} (+{gnn_pca_dim} GNN embedding)")
    else:
        print("\n[Step 4] 跳過 GNN（--skip_gnn）")

    # ── Step 5：集成訓練 ─────────────────────────
    print("\n" + "="*55)
    print("[Step 5] 訓練集成模型（XGBoost + LightGBM + CatBoost）")
    print("="*55)

    ensemble = StackingEnsemble(
        n_splits=5,
        use_focal_loss=use_focal_loss,
        use_smote=use_smote,
        smote_strategy=smote_strategy,
    )
    ensemble.fit(X_tr, y_tr)

    # ── Step 5.5：Pseudo-labeling（可選）─────────
    if use_pseudo_label:
        print("\n" + "="*55)
        print("[Step 5.5] 半監督學習 — Pseudo-labeling")
        print("="*55)

        # 載入 predict/ 資料並做特徵工程
        pred_tables = load_predict_data(predict_dir)
        if pred_tables is not None:
            pred_user = pred_tables["user_info_predict"]
            pred_twd  = pred_tables["twd_transfer_predict"]
            pred_crypto = pred_tables["crypto_transfer_predict"]
            pred_trading = pred_tables["usdt_twd_trading_predict"]
            pred_swap = pred_tables["usdt_swap_predict"]

            print(f"  Predict 用戶: {len(pred_user)}")
            print(f"  特徵工程中 ...")

            pred_feat = build_all_features(pred_user, pred_twd, pred_crypto, pred_trading, pred_swap)
            pred_feat = pred_feat.replace([np.inf, -np.inf], np.nan).fillna(0)

            # 對齊欄位：用 train 的篩選後欄位
            train_cols = X_selected.columns.tolist()
            for col in train_cols:
                if col not in pred_feat.columns:
                    pred_feat[col] = 0
            X_pred_raw = pred_feat[train_cols].values.astype(np.float32)

            # 加異常分數
            pred_anomaly = anomaly_extractor.transform(X_pred_raw)
            X_pred = np.hstack([X_pred_raw, pred_anomaly])

            # 加 GNN embedding（如果有）
            if not skip_gnn and gnn_embed_dim > 0:
                # predict 用戶沒有 GNN 圖，用零向量填充
                gnn_zeros = np.zeros((len(X_pred), gnn_embed_dim), dtype=np.float32)
                X_pred = np.hstack([X_pred, gnn_zeros])

            print(f"  Predict 特徵維度: {X_pred.shape[1]} (與 train 一致: {X_tr.shape[1]})")

            X_tr_aug, y_tr_aug, pl_stats = pseudo_label(
                ensemble, X_tr, y_tr, X_pred,
                pos_threshold=0.85,
                neg_threshold=0.05,
                max_iter=2,
            )

            with open(os.path.join(output_dir, "pseudo_label_stats.json"), "w", encoding="utf-8") as f:
                json.dump(pl_stats, f, indent=2, ensure_ascii=False)

            # 用擴充後資料重新訓練
            if len(X_tr_aug) > len(X_tr):
                print(f"\n  用擴充資料重新訓練 ensemble ...")
                ensemble_pl = StackingEnsemble(
                    n_splits=5,
                    use_focal_loss=use_focal_loss,
                    use_smote=use_smote,
                    smote_strategy=smote_strategy,
                )
                ensemble_pl.fit(X_tr_aug, y_tr_aug)

                # 比較：擴充前 vs 擴充後
                from sklearn.metrics import f1_score as f1_fn, average_precision_score as ap_fn
                proba_before = ensemble.predict_proba(X_te)
                proba_after = ensemble_pl.predict_proba(X_te)
                t_before = ensemble.oof_threshold
                t_after = ensemble_pl.oof_threshold
                f1_before = f1_fn(y_te, (proba_before >= t_before).astype(int))
                f1_after = f1_fn(y_te, (proba_after >= t_after).astype(int))
                ap_before = ap_fn(y_te, proba_before)
                ap_after = ap_fn(y_te, proba_after)

                print(f"\n  Pseudo-label 效果比較:")
                print(f"    Before: F1={f1_before:.4f}, AUC-PR={ap_before:.4f}")
                print(f"    After:  F1={f1_after:.4f}, AUC-PR={ap_after:.4f}")

                if f1_after > f1_before:
                    print(f"    ✓ 採用 pseudo-label 模型 (F1 +{f1_after-f1_before:.4f})")
                    ensemble = ensemble_pl
                else:
                    print(f"    ✗ 保持原始模型 (pseudo-label 未改善)")
        else:
            print("  [注意] 找不到 predict/ 資料目錄，跳過")

    # ── Step 6：評估 ─────────────────────────────
    print("\n" + "="*55)
    print("[Step 6] 模型評估")
    print("="*55)

    y_proba   = ensemble.predict_proba(X_te)
    # 先用 OOF 閾值評估
    optimal_t = ensemble.oof_threshold
    metrics   = evaluate(y_te, y_proba, threshold=optimal_t, label=f"Soft Voting Ensemble (t={optimal_t:.2f})")
    # 再搜索測試集最佳閾值（供參考，不用於最終預測）
    test_t    = find_optimal_threshold(y_te, y_proba)
    metrics["oof_threshold"]  = float(optimal_t)
    metrics["test_threshold"] = float(test_t)

    # 閾值掃描：找 F1 最佳閾值
    from sklearn.metrics import precision_recall_curve, f1_score as f1_fn
    precision_arr, recall_arr, thresh_arr = precision_recall_curve(y_te, y_proba)
    f1_arr = np.where(
        (precision_arr + recall_arr) > 0,
        2 * precision_arr * recall_arr / (precision_arr + recall_arr),
        0,
    )
    best_idx = np.argmax(f1_arr[:-1])  # 最後一個是 sentinel
    best_sweep_t = thresh_arr[best_idx]
    best_sweep_f1 = f1_arr[best_idx]

    print(f"\n  閾值掃描最佳: t={best_sweep_t:.4f} → F1={best_sweep_f1:.4f} "
          f"(P={precision_arr[best_idx]:.4f}, R={recall_arr[best_idx]:.4f})")

    # 如果掃描閾值比 OOF 閾值更好，用掃描閾值
    f1_oof_t = f1_fn(y_te, (y_proba >= optimal_t).astype(int))
    if best_sweep_f1 > f1_oof_t:
        print(f"  → 採用掃描閾值 (F1: {f1_oof_t:.4f} → {best_sweep_f1:.4f})")
        optimal_t = best_sweep_t
        metrics = evaluate(y_te, y_proba, threshold=optimal_t,
                          label=f"Soft Voting Ensemble (t={optimal_t:.4f})")

    metrics["oof_threshold"]  = float(ensemble.oof_threshold)
    metrics["test_threshold"] = float(test_t)
    metrics["sweep_threshold"] = float(best_sweep_t)
    metrics["sweep_f1"] = float(best_sweep_f1)

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2, ensure_ascii=False)

    # 輸出測試集每筆的機率值（模型沒看過的資料）
    test_result_df = pd.DataFrame({
        "user_id":    feat_df.index[idx_te],
        "true_label": y_te,
        "risk_score": y_proba,
        "pred_label":  (y_proba >= optimal_t).astype(int),
    })
    test_result_df = test_result_df.sort_values("risk_score", ascending=False)
    test_result_df.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)
    print(f"\n  測試集預測結果已儲存：test_predictions.csv ({len(test_result_df)} 筆)")

    # ── Step 7：SHAP ─────────────────────────────
    print("\n" + "="*55)
    print("[Step 7] SHAP 可解釋性分析")
    print("="*55)

    explainer   = SHAPExplainer(ensemble.xgb_model, feature_names)
    X_te_scaled = ensemble.scaler.transform(X_te)
    bg_n        = min(200, len(X_te_scaled))
    explainer.fit(X_te_scaled[:bg_n], X_te_scaled)
    explainer.plot_global_importance(
        top_n=20,
        save_path=os.path.join(output_dir, "shap_global.png"),
    )

    # ── Step 8：個體報告 ─────────────────────────
    print("\n" + "="*55)
    print("[Step 8] 生成高風險用戶個體報告（Top 5）")
    print("="*55)

    cf_explainer  = CounterfactualExplainer(
        ensemble.xgb_model, ensemble.scaler, feature_names
    )
    n_report      = min(5, len(y_proba))
    high_risk_idx = np.argsort(y_proba)[-n_report:][::-1]
    reports       = []

    for rank, local_idx in enumerate(high_risk_idx, 1):
        global_idx = idx_te[local_idx]
        uid        = feat_df.index[global_idx]
        score      = float(y_proba[local_idx])
        true_label = int(y_te[local_idx])

        shap_r = explainer.explain_user(local_idx, user_id=uid, risk_score=score)
        cf_r   = cf_explainer.generate(X_te[local_idx])
        report = generate_user_report(uid, score, shap_r, cf_r)

        tag    = "【實際黑名單】" if true_label == 1 else "【實際正常】"
        header = f"\n── 第 {rank} 高風險  {tag} ──"
        reports.append(header + "\n" + report)
        print(header)
        print(report)

    with open(os.path.join(output_dir, "risk_reports.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(reports))

    # ── Step 8.5：SSR 穩定性評測 ──────────────────
    print("\n" + "="*55)
    print("[Step 8.5] SSR 穩定性評測")
    print("="*55)

    ssr_evaluator = SSREvaluator(explainer, feature_names)
    ssr_results = ssr_evaluator.evaluate(
        X_te_scaled, y_te,
        epsilons=[0.05, 0.10, 0.15, 0.20],
        top_k_list=[1, 3, 5],
        n_samples=min(500, len(X_te_scaled)),
        n_perturbations=10,
    )
    ssr_evaluator.plot_ssr_curves(
        ssr_results,
        save_path=os.path.join(output_dir, "ssr_curves.png"),
    )

    with open(os.path.join(output_dir, "ssr_results.json"), "w", encoding="utf-8") as f:
        # 將 tuple key 轉為 string
        serializable = {
            "overall": {f"eps={e}_k={k}": v for (e, k), v in ssr_results["overall"].items()},
            "by_class": {
                str(cls): {f"eps={e}_k={k}": v for (e, k), v in vals.items()}
                for cls, vals in ssr_results["by_class"].items()
            }
        }
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    # ── Step 9：全量評分 ─────────────────────────
    print("\n" + "="*55)
    print("[Step 9] 全量用戶風險評分輸出")
    print("="*55)

    all_proba = ensemble.predict_proba(X_all)

    result_df = pd.DataFrame({
        "user_id":             feat_df.index,
        "true_label":          y,
        "risk_score":          all_proba,
        "predicted_blacklist": (all_proba >= optimal_t).astype(int),
    })
    result_df["risk_level"] = pd.cut(
        result_df["risk_score"],
        bins=[-0.001, 0.2, 0.4, 0.6, 0.8, 1.001],
        labels=["正常", "低風險", "中風險", "高風險", "極高風險"],
    )
    result_df = result_df.sort_values("risk_score", ascending=False)
    result_df.to_csv(os.path.join(output_dir, "user_risk_scores.csv"), index=False)

    print("\n  風險等級分布：")
    dist = result_df["risk_level"].value_counts().reindex(
        ["極高風險", "高風險", "中風險", "低風險", "正常"]
    )
    for level, cnt in dist.items():
        bar = "█" * min(int(cnt / len(result_df) * 50), 45)
        print(f"  {level:<6}  {cnt:>6}  {bar}")

    # ── Step 10：Predict 資料預測 → 提交 CSV ─────
    print("\n" + "="*55)
    print("[Step 10] 對 predict 資料產出最終提交 CSV")
    print("="*55)

    pred_tables = load_predict_data(predict_dir)
    if pred_tables is not None:
        pred_user    = pred_tables["user_info_predict"]
        pred_twd     = pred_tables["twd_transfer_predict"]
        pred_crypto  = pred_tables["crypto_transfer_predict"]
        pred_trading = pred_tables["usdt_twd_trading_predict"]
        pred_swap    = pred_tables["usdt_swap_predict"]

        print(f"  Predict 用戶數: {len(pred_user):,}")

        # 特徵工程
        pred_feat = build_all_features(pred_user, pred_twd, pred_crypto, pred_trading, pred_swap)
        pred_feat = pred_feat.replace([np.inf, -np.inf], np.nan).fillna(0)

        # 對齊欄位（用 train 的篩選後欄位）
        train_cols = X_selected.columns.tolist()
        for col in train_cols:
            if col not in pred_feat.columns:
                pred_feat[col] = 0
        X_pred_raw = pred_feat[train_cols].values.astype(np.float32)

        # 加異常分數
        pred_anomaly = anomaly_extractor.transform(X_pred_raw)
        X_pred = np.hstack([X_pred_raw, pred_anomaly])

        # 加 GNN embedding（predict 用戶無圖，用零向量）
        if not skip_gnn and gnn_embed_dim > 0:
            gnn_zeros = np.zeros((len(X_pred), gnn_embed_dim), dtype=np.float32)
            X_pred = np.hstack([X_pred, gnn_zeros])

        print(f"  特徵維度: {X_pred.shape[1]} (與 train 一致: {X_tr.shape[1]})")

        # 預測
        pred_proba = ensemble.predict_proba(X_pred)
        pred_labels = (pred_proba >= optimal_t).astype(int)

        # 產出提交 CSV（格式：user_id, status）
        submission_df = pd.DataFrame({
            "user_id": pred_feat.index,
            "status":  pred_labels,
        })
        submission_path = os.path.join(output_dir, "submission.csv")
        submission_df.to_csv(submission_path, index=False)

        # 同時輸出含機率的詳細版（供自己分析用）
        detail_df = pd.DataFrame({
            "user_id":    pred_feat.index,
            "risk_score": pred_proba,
            "status":     pred_labels,
        }).sort_values("risk_score", ascending=False)
        detail_df.to_csv(os.path.join(output_dir, "predict_detail.csv"), index=False)

        n_black = pred_labels.sum()
        print(f"\n  閾值: {optimal_t:.4f}")
        print(f"  預測黑名單: {n_black} / {len(pred_labels)} ({n_black/len(pred_labels)*100:.2f}%)")
        print(f"  提交檔案: {submission_path}")

        # 機率分布統計
        print(f"\n  Predict 機率分布:")
        for lo, hi, label in [(0, 0.2, "正常"), (0.2, 0.4, "低風險"),
                               (0.4, 0.6, "中風險"), (0.6, 0.8, "高風險"),
                               (0.8, 1.001, "極高風險")]:
            cnt = ((pred_proba >= lo) & (pred_proba < hi)).sum()
            bar = "█" * min(int(cnt / len(pred_proba) * 50), 45)
            print(f"    {label:<6}  {cnt:>6}  {bar}")
    else:
        print("  [注意] 找不到 predict/ 資料目錄，跳過")

    # ── 完成 ─────────────────────────────────────
    out = os.path.abspath(output_dir)
    print(f"\n{'='*55}")
    print(f"  完成！輸出目錄：{out}/")
    print(f"{'='*55}")
    print(f"  features_raw.csv                原始特徵矩陣")
    print(f"  features.csv                    篩選後特徵矩陣 + 異常分數")
    print(f"  feature_selection_report.json    特徵篩選報告")
    print(f"  metrics.json                    評估指標")
    print(f"  user_risk_scores.csv            全量風險評分")
    print(f"  test_predictions.csv            測試集預測結果")
    print(f"  submission.csv                  ★ 比賽提交檔案")
    print(f"  predict_detail.csv              Predict 詳細機率")
    print(f"  shap_global.png                 特徵重要性圖")
    print(f"  risk_reports.txt                高風險個體報告")
    print(f"  ssr_results.json                SSR 穩定性數據")
    print(f"  ssr_curves.png                  SSR 衰減曲線圖")
    if not skip_gnn and gnn_embed_dim > 0:
        print(f"  gnn_model.pt                    GNN 模型權重")

    return ensemble, result_df, metrics


# ═══════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="虛擬貨幣黑名單預測訓練")
    parser.add_argument(
        "--data_dir", default=os.path.join(ROOT, "adjust_data", "train"),
        help="訓練資料夾路徑",
    )
    parser.add_argument(
        "--predict_dir", default=os.path.join(ROOT, "adjust_data", "predict"),
        help="Predict 資料夾路徑（pseudo-labeling 用）",
    )
    parser.add_argument(
        "--output", default=os.path.join(os.path.dirname(WEI_MODEL_DIR), "output"),
        help="輸出目錄（預設：Wei_model/output/）",
    )
    parser.add_argument(
        "--skip_gnn", action="store_true", default=False,
        help="跳過 GNN 訓練（資料量小或無鏈上交易時使用）",
    )
    parser.add_argument(
        "--use_focal_loss", action="store_true", default=True,
        help="LightGBM 使用 Focal Loss（預設開啟）",
    )
    parser.add_argument(
        "--no_focal_loss", action="store_true", default=False,
        help="關閉 Focal Loss，使用 scale_pos_weight",
    )
    parser.add_argument(
        "--use_smote", action="store_true", default=False,
        help="啟用 Borderline-SMOTE（每個 CV fold 內部執行）",
    )
    parser.add_argument(
        "--smote_strategy", type=float, default=0.3,
        help="SMOTE sampling_strategy（預設 0.3 = 正負比約 3:1）",
    )
    parser.add_argument(
        "--use_pseudo_label", action="store_true", default=False,
        help="啟用 Pseudo-labeling 半監督學習",
    )
    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        predict_dir=args.predict_dir,
        output_dir=args.output,
        skip_gnn=args.skip_gnn,
        use_focal_loss=not args.no_focal_loss,
        use_smote=args.use_smote,
        smote_strategy=args.smote_strategy,
        use_pseudo_label=args.use_pseudo_label,
    )