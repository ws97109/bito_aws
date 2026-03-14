"""
Main Pipeline — 真實資料版
訓練流程：特徵工程 → GNN → 集成 → 可解釋性

使用方式：
    python main.py                               # 預設讀 
    python main.py --data_dir ../adjust_data/train         # 指定資料夾
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# 直接從當前目錄導入（所有模組都在 model/ 目錄下）
from Feature_rngineering import build_all_features
from Gnn_model import build_transaction_graph, BlacklistGNN
from ensemble import StackingEnsemble, evaluate, find_optimal_threshold
from shap_explainer import (
    SHAPExplainer, CounterfactualExplainer, generate_user_report,
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

        # 存儲時去掉 _train 後綴，方便後續使用
        key = name.replace("_train", "")
        tables[key] = df

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

def main(data_dir: str = "adjust_data/train", output_dir: str = "output", skip_gnn: bool = False):
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1：載入真實 CSV ─────────────────────
    print("\n" + "="*55)
    print(f"[Step 1] 載入資料  ←  {os.path.abspath(data_dir)}/")
    print("="*55)

    tables    = load_and_validate(data_dir)
    user_info = tables["user_info"]
    twd       = tables["twd_transfer"]
    crypto    = tables["crypto_transfer"]
    trading   = tables["usdt_twd_trading"]
    swap      = tables["usdt_swap"]

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
    X             = feat_df.drop(columns=["status"]).values.astype(np.float32)
    feature_names = feat_df.drop(columns=["status"]).columns.tolist()

    print(f"\n  用戶總數    : {len(feat_df):,}")
    print(f"  特徵維度    : {X.shape[1]}")
    print(f"  黑名單用戶  : {y.sum():,} ({y.mean()*100:.2f}%)")

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

    # ── Step 4：GNN ──────────────────────────────
    gnn_proba_all = None

    if not skip_gnn:
        print("\n" + "="*55)
        print("[Step 4] 建立交易圖 & 訓練 GNN")
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
                gnn_proba_all = gnn_mdl.predict_proba(
                    graph.to(DEVICE), tabular_t.to(DEVICE)
                ).cpu().numpy()

            torch.save(gnn_mdl.state_dict(),
                       os.path.join(output_dir, "gnn_model.pt"))
    else:
        print("\n[Step 4] 跳過 GNN（--skip_gnn）")

    gnn_tr = gnn_proba_all[idx_tr] if gnn_proba_all is not None else None
    gnn_te = gnn_proba_all[idx_te] if gnn_proba_all is not None else None

    # ── Step 5：集成訓練 ─────────────────────────
    print("\n" + "="*55)
    print("[Step 5] 訓練集成模型（XGBoost + Isolation Forest + GNN）")
    print("="*55)

    ensemble = StackingEnsemble(n_splits=5)
    ensemble.fit(X_tr, y_tr, gnn_proba=gnn_tr)

    # ── Step 6：評估 ─────────────────────────────
    print("\n" + "="*55)
    print("[Step 6] 模型評估")
    print("="*55)

    y_proba   = ensemble.predict_proba(X_te, gnn_proba=gnn_te)
    metrics   = evaluate(y_te, y_proba, label="Stacking Ensemble")
    optimal_t = find_optimal_threshold(y_te, y_proba)
    metrics["optimal_threshold"] = float(optimal_t)

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2, ensure_ascii=False)

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
        ensemble, ensemble.scaler, feature_names
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
        # 傳入對應的 GNN 概率
        gnn_prob_single = float(gnn_te[local_idx]) if gnn_te is not None else None
        cf_r   = cf_explainer.generate(X_te[local_idx], gnn_prob=gnn_prob_single)
        report = generate_user_report(uid, score, shap_r, cf_r)

        tag    = "【實際黑名單】" if true_label == 1 else "【實際正常】"
        header = f"\n── 第 {rank} 高風險  {tag} ──"
        reports.append(header + "\n" + report)
        print(header)
        print(report)

    with open(os.path.join(output_dir, "risk_reports.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(reports))

    # ── Step 9：全量評分 ─────────────────────────
    print("\n" + "="*55)
    print("[Step 9] 全量用戶風險評分輸出")
    print("="*55)

    all_proba = ensemble.predict_proba(X, gnn_proba=gnn_proba_all)

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

    # ── 完成 ─────────────────────────────────────
    out = os.path.abspath(output_dir)
    print(f"\n{'='*55}")
    print(f"  完成！輸出目錄：{out}/")
    print(f"{'='*55}")
    print(f"  features.csv          特徵矩陣")
    print(f"  metrics.json          評估指標")
    print(f"  user_risk_scores.csv  全量風險評分")
    print(f"  shap_global.png       特徵重要性圖")
    print(f"  risk_reports.txt      高風險個體報告")
    if not skip_gnn and gnn_proba_all is not None:
        print(f"  gnn_model.pt          GNN 模型權重")

    return ensemble, result_df, metrics


# ═══════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="虛擬貨幣黑名單預測訓練")
    parser.add_argument(
        "--data_dir", default="../adjust_data/train",
        help="CSV 資料夾路徑（預設：../adjust_data/train）",
    )
    parser.add_argument(
        "--output", default="output",
        help="輸出目錄（預設：output/）",
    )
    parser.add_argument(
        "--skip_gnn", action="store_true", default=False,
        help="跳過 GNN 訓練（資料量小或無鏈上交易時使用）",
    )
    args = parser.parse_args()
    main(data_dir=args.data_dir, output_dir=args.output, skip_gnn=args.skip_gnn)