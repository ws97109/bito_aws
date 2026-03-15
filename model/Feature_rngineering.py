"""
Feature Engineering Module
論文級特徵工程：從五張資料表提取多維度風險特徵
"""
import pandas as pd
import numpy as np
from typing import Optional
import ipaddress
import networkx as nx


SCALE = 1e-8  # 所有金額欄位需乘以此係數


# ─────────────────────────────────────────────
# 1. 用戶基本特徵（user_info）
# ─────────────────────────────────────────────

HIGH_RISK_CAREERS = {14, 22, 23, 29}       # 區塊鏈業、自由業、無業、珠寶銀樓
HIGH_RISK_INCOME  = {4, 8, 9}              # 理財投資、挖礦、買賣房地產（個人）


def build_user_features(user_info: pd.DataFrame) -> pd.DataFrame:
    df = user_info.copy()
    now = pd.Timestamp.now()

    df["confirmed_at"]        = pd.to_datetime(df["confirmed_at"])
    df["level1_finished_at"]  = pd.to_datetime(df["level1_finished_at"])
    df["level2_finished_at"]  = pd.to_datetime(df["level2_finished_at"])

    # KYC 完成狀態
    df["has_kyc_level2"] = df["level2_finished_at"].notna().astype(int)

    # KYC 完成速度（秒）—— 過快可能是機器人
    df["kyc_speed_sec"] = (
        df["level2_finished_at"] - df["level1_finished_at"]
    ).dt.total_seconds().clip(lower=0)

    # 帳號年齡（天）
    df["account_age_days"] = (now - df["confirmed_at"]).dt.days.clip(lower=0)

    # 年齡（資料中已是數值欄位）
    df["age"] = pd.to_numeric(df["age"], errors="coerce").clip(lower=0, upper=120)

    # 高風險職業 / 收入來源
    df["is_high_risk_career"] = df["career"].isin(HIGH_RISK_CAREERS).astype(int)
    df["is_high_risk_income"] = df["income_source"].isin(HIGH_RISK_INCOME).astype(int)

    # 職業 × 收入組合風險（同時命中）
    df["career_income_risk"] = (
        df["is_high_risk_career"] & df["is_high_risk_income"]
    ).astype(int)

    # 來源管道
    df["is_app_user"] = (df["user_source"] == 1).astype(int)

    feature_cols = [
        "user_id",
        "has_kyc_level2", "kyc_speed_sec", "account_age_days",
        "age", "is_high_risk_career", "is_high_risk_income",
        "career_income_risk", "is_app_user",
    ]
    return df[feature_cols].set_index("user_id")


# ─────────────────────────────────────────────
# 2. 法幣行為特徵（twd_transfer）
# ─────────────────────────────────────────────

def build_twd_features(twd: pd.DataFrame) -> pd.DataFrame:
    df = twd.copy()
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["amount_twd"] = df["ori_samount"] * SCALE

    deposit  = df[df["kind"] == 0]
    withdraw = df[df["kind"] == 1]

    def agg(g: pd.DataFrame, prefix: str) -> pd.DataFrame:
        return g.groupby("user_id")["amount_twd"].agg(
            **{
                f"{prefix}_count": "count",
                f"{prefix}_sum":   "sum",
                f"{prefix}_mean":  "mean",
                f"{prefix}_std":   "std",
                f"{prefix}_max":   "max",
            }
        ).fillna(0)

    dep_agg = agg(deposit,  "twd_dep")
    wit_agg = agg(withdraw, "twd_wit")
    feat = dep_agg.join(wit_agg, how="outer").fillna(0)

    # 淨流入金額（負值代表資金淨流出）
    feat["twd_net_flow"] = feat["twd_dep_sum"] - feat["twd_wit_sum"]

    # 出 / 入比率（接近 1 表示幾乎全部提出，cap 在 10 避免分母趨近 0 導致爆炸）
    feat["twd_withdraw_ratio"] = (
        feat["twd_wit_sum"] / (feat["twd_dep_sum"] + 1e-9)
    ).clip(upper=10)

    # 結構化交易偵測：金額標準差小（金額高度一致 → Smurfing）
    feat["twd_smurf_flag"] = (
        (feat["twd_dep_std"] < feat["twd_dep_mean"] * 0.05) &
        (feat["twd_dep_count"] >= 5)
    ).astype(int)

    # 提領是否有 IP（無 IP = 外部觸發，有 IP = 站內操作）
    wit_ip = withdraw.groupby("user_id")["source_ip_hash"].apply(
        lambda x: x.notna().mean()
    ).rename("twd_wit_ip_ratio")
    feat = feat.join(wit_ip, how="left").fillna(0)

    return feat


# ─────────────────────────────────────────────
# 3. 虛擬貨幣行為特徵（crypto_transfer）
# ─────────────────────────────────────────────

def build_crypto_features(crypto: pd.DataFrame) -> pd.DataFrame:
    df = crypto.copy()
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["amount_twd"] = df["ori_samount"] * SCALE * df["twd_srate"] * SCALE

    deposit  = df[df["kind"] == 0]
    withdraw = df[df["kind"] == 1]
    external = df[df["sub_kind"] == 0]  # 鏈上交易

    def agg(g: pd.DataFrame, prefix: str) -> pd.DataFrame:
        return g.groupby("user_id")["amount_twd"].agg(
            **{
                f"{prefix}_count": "count",
                f"{prefix}_sum":   "sum",
                f"{prefix}_mean":  "mean",
                f"{prefix}_max":   "max",
            }
        ).fillna(0)

    feat = agg(deposit, "crypto_dep").join(agg(withdraw, "crypto_wit"), how="outer").fillna(0)

    # 鏈上提領次數（外部轉出）
    ext_wit = external[external["kind"] == 1]
    feat["crypto_external_wit_count"] = (
        ext_wit.groupby("user_id").size().reindex(feat.index, fill_value=0)
    )

    # 幣種多樣性（越多幣種 → 可能在混淆追蹤）
    feat["crypto_currency_diversity"] = (
        df.groupby("user_id")["currency"].nunique()
          .reindex(feat.index, fill_value=0)
    )

    # 協定多樣性（跨多條鏈）
    feat["crypto_protocol_diversity"] = (
        df.groupby("user_id")["protocol"].nunique()
          .reindex(feat.index, fill_value=0)
    )

    # 不同錢包地址數（分散資金到多錢包是洗錢典型手法）
    wallets = pd.concat([
        df[["user_id", "from_wallet_hash"]].rename(columns={"from_wallet_hash": "wallet"}),
        df[["user_id", "to_wallet_hash"]].rename(columns={"to_wallet_hash": "wallet"}),
    ]).dropna(subset=["wallet"])
    feat["crypto_wallet_hash_nunique"] = (
        wallets.groupby("user_id")["wallet"].nunique()
               .reindex(feat.index, fill_value=0)
    )

    # 內轉次數與內轉對象數
    internal = df[df["sub_kind"] == 1]
    feat["crypto_internal_count"] = (
        internal.groupby("user_id").size()
                .reindex(feat.index, fill_value=0)
    )
    feat["crypto_internal_peer_count"] = (
        internal.groupby("user_id")["relation_user_id"].nunique()
                .reindex(feat.index, fill_value=0)
    )

    # 提領 IP 覆蓋率
    wit_ip = withdraw.groupby("user_id")["source_ip_hash"].apply(
        lambda x: x.notna().mean()
    ).rename("crypto_wit_ip_ratio")
    feat = feat.join(wit_ip, how="left").fillna(0)

    return feat


# ─────────────────────────────────────────────
# 4. 交易行為特徵（usdt_twd_trading + usdt_swap）
# ─────────────────────────────────────────────

def build_trading_features(
    trading: pd.DataFrame,
    swap: pd.DataFrame,
) -> pd.DataFrame:
    t = trading.copy()
    s = swap.copy()

    t["updated_at"] = pd.to_datetime(t["updated_at"])
    s["created_at"] = pd.to_datetime(s["created_at"])

    t["amount_twd"] = t["trade_samount"] * SCALE * t["twd_srate"] * SCALE
    s["amount_twd"] = s["twd_samount"] * SCALE

    # 掛單行為
    feat = t.groupby("user_id")["amount_twd"].agg(
        trading_count="count",
        trading_sum="sum",
        trading_mean="mean",
        trading_max="max",
    ).fillna(0)

    feat["trading_buy_ratio"] = (
        t.groupby("user_id")["is_buy"].mean()
         .reindex(feat.index, fill_value=0.5)
    )
    feat["trading_market_order_ratio"] = (
        t.groupby("user_id")["is_market"].mean()
         .reindex(feat.index, fill_value=0)
    )

    # 一鍵買賣
    swap_agg = s.groupby("user_id")["amount_twd"].agg(
        swap_count="count",
        swap_sum="sum",
    ).fillna(0)
    feat = feat.join(swap_agg, how="outer").fillna(0)

    # 總交易量（兩者合計）
    feat["total_trading_volume"] = feat["trading_sum"] + feat["swap_sum"]

    return feat


# ─────────────────────────────────────────────
# 5. IP 時序特徵（跨資料表整合）
# ─────────────────────────────────────────────

def int_to_ip(ip_int: float) -> Optional[str]:
    try:
        return str(ipaddress.IPv4Address(int(ip_int)))
    except Exception:
        return None


def build_ip_features(
    twd: pd.DataFrame,
    crypto: pd.DataFrame,
    trading: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    for df, ts_col in [(twd, "created_at"), (crypto, "created_at"), (trading, "updated_at")]:
        tmp = df[["user_id", "source_ip_hash", ts_col]].copy()
        tmp["ts"] = pd.to_datetime(tmp[ts_col])
        frames.append(tmp[["user_id", "source_ip_hash", "ts"]])

    all_ip = pd.concat(frames, ignore_index=True).dropna(subset=["source_ip_hash"])
    all_ip["hour"] = all_ip["ts"].dt.hour

    # 每個 user 使用的唯一 IP 數
    feat = all_ip.groupby("user_id")["source_ip_hash"].agg(
        ip_unique_count="nunique",
        ip_total_count="count",
    )

    # 深夜操作比例（0-5 點）
    feat["ip_night_ratio"] = (
        all_ip[all_ip["hour"].between(0, 5)]
        .groupby("user_id").size()
        .reindex(feat.index, fill_value=0)
        / feat["ip_total_count"].clip(lower=1)
    )

    # 同一 IP 被多少不同 user 使用（IP 共用風險）
    ip_user_count = (
        all_ip.groupby("source_ip_hash")["user_id"].nunique()
              .rename("ip_shared_user_count")
    )
    all_ip = all_ip.join(ip_user_count, on="source_ip_hash")
    feat["ip_max_shared"] = (
        all_ip.groupby("user_id")["ip_shared_user_count"].max()
              .reindex(feat.index, fill_value=1)
    )

    return feat


# ─────────────────────────────────────────────
# 6. 資金停留時間特徵
# ─────────────────────────────────────────────

def build_velocity_features(
    twd: pd.DataFrame,
    crypto: pd.DataFrame,
) -> pd.DataFrame:
    """
    資金快速移動偵測：入金後多快提領
    指標越小 → 資金停留時間越短 → 風險越高
    """
    t = twd.copy()
    c = crypto.copy()
    t["ts"] = pd.to_datetime(t["created_at"])
    c["ts"] = pd.to_datetime(c["created_at"])
    t["amount"] = t["ori_samount"] * SCALE
    c["amount"] = c["ori_samount"] * SCALE * c["twd_srate"] * SCALE

    records = []
    for df in [t, c]:
        for uid, grp in df.groupby("user_id"):
            grp = grp.sort_values("ts")
            deps = grp[grp["kind"] == 0]["ts"].values
            wits = grp[grp["kind"] == 1]["ts"].values
            if len(deps) == 0 or len(wits) == 0:
                continue
            # 最近一次入金到最近一次提領的時間差（秒）
            last_dep = deps.max()
            last_wit = wits[wits > last_dep]
            if len(last_wit) == 0:
                continue
            delta = (last_wit.min() - last_dep) / np.timedelta64(1, "s")
            records.append({"user_id": uid, "fund_stay_sec": max(delta, 0)})

    if not records:
        return pd.DataFrame(columns=["user_id", "fund_stay_sec"]).set_index("user_id")

    vel = pd.DataFrame(records).groupby("user_id")["fund_stay_sec"].min()
    return vel.to_frame()


# ─────────────────────────────────────────────
# 7. 輕量圖特徵（crypto_transfer 內轉關係）
# ─────────────────────────────────────────────

def build_graph_features(
    crypto: pd.DataFrame,
    user_info: pd.DataFrame,
) -> pd.DataFrame:
    """
    從 crypto_transfer 的內轉關係（sub_kind=1）建圖，
    提取不依賴 label 的圖結構特徵：
      - PageRank、in/out degree
      - connected_component_size（連通分量大小）
      - betweenness_centrality（介數中心性）
    注意：不使用 hop_to_blacklist（會造成 label leakage）
    """
    df = crypto.copy()
    internal = df[df["sub_kind"] == 1].dropna(subset=["relation_user_id"])
    internal["relation_user_id"] = internal["relation_user_id"].astype(int)

    all_user_ids = user_info["user_id"].unique()

    # 建立有向圖
    G = nx.DiGraph()
    G.add_nodes_from(all_user_ids)
    edges = list(zip(internal["user_id"], internal["relation_user_id"]))
    G.add_edges_from(edges)

    # PageRank
    pr = nx.pagerank(G, alpha=0.85, max_iter=100)

    # in_degree / out_degree
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    # 連通分量大小（無向圖）— 人頭戶常在同一群組
    G_undirected = G.to_undirected()
    comp_size = {}
    for comp in nx.connected_components(G_undirected):
        size = len(comp)
        for node in comp:
            comp_size[node] = size

    # 介數中心性（取樣加速，抓資金中繼站）
    # 全量計算太慢，用近似法取樣 500 個節點
    bc = nx.betweenness_centrality(G, k=min(500, len(G)), seed=42)

    feat = pd.DataFrame({
        "pagerank_score": pd.Series(pr),
        "graph_in_degree": pd.Series(in_deg),
        "graph_out_degree": pd.Series(out_deg),
        "connected_component_size": pd.Series(comp_size),
        "betweenness_centrality": pd.Series(bc),
    })
    feat.index.name = "user_id"
    feat = feat.fillna(0)

    return feat


# ─────────────────────────────────────────────
# 8. 跨表衍生特徵
# ─────────────────────────────────────────────

def build_cross_table_features(
    twd: pd.DataFrame,
    crypto: pd.DataFrame,
    trading: pd.DataFrame,
    swap: pd.DataFrame,
) -> pd.DataFrame:
    """
    跨表整合：總交易次數、首末交易間隔、週末交易比、行為加速偵測
    """
    # 收集所有交易的 (user_id, timestamp)
    frames = []
    for df, ts_col in [
        (twd, "created_at"), (crypto, "created_at"),
        (trading, "updated_at"), (swap, "created_at"),
    ]:
        tmp = df[["user_id"]].copy()
        tmp["ts"] = pd.to_datetime(df[ts_col])
        frames.append(tmp)

    all_tx = pd.concat(frames, ignore_index=True).dropna(subset=["ts"])

    # 總交易次數
    total_count = all_tx.groupby("user_id").size().rename("total_tx_count")

    # 首末交易間隔（天）
    ts_range = all_tx.groupby("user_id")["ts"].agg(["min", "max"])
    first_to_last = ((ts_range["max"] - ts_range["min"]).dt.days).rename("first_to_last_tx_days")

    # 週末交易佔比
    all_tx["is_weekend"] = all_tx["ts"].dt.dayofweek.isin([5, 6]).astype(int)
    weekend_ratio = (
        all_tx.groupby("user_id")["is_weekend"].mean()
              .rename("weekend_tx_ratio")
    )

    # 行為加速偵測：近 7 天交易頻率 / 近 30 天日均
    max_ts = all_tx["ts"].max()
    cutoff_7d = max_ts - pd.Timedelta(days=7)
    cutoff_30d = max_ts - pd.Timedelta(days=30)

    cnt_7d = (
        all_tx[all_tx["ts"] >= cutoff_7d]
        .groupby("user_id").size()
        .rename("cnt_7d")
    )
    cnt_30d = (
        all_tx[all_tx["ts"] >= cutoff_30d]
        .groupby("user_id").size()
        .rename("cnt_30d")
    )
    velocity = cnt_7d.to_frame().join(cnt_30d, how="outer").fillna(0)
    # 近 7 天日均 / 近 30 天日均
    velocity["velocity_ratio_7d_vs_30d"] = (
        (velocity["cnt_7d"] / 7) / (velocity["cnt_30d"] / 30 + 1e-9)
    )

    feat = pd.DataFrame({
        "total_tx_count": total_count,
        "first_to_last_tx_days": first_to_last,
        "weekend_tx_ratio": weekend_ratio,
        "velocity_ratio_7d_vs_30d": velocity["velocity_ratio_7d_vs_30d"],
    })
    feat.index.name = "user_id"
    feat = feat.fillna(0)

    return feat


# ─────────────────────────────────────────────
# 9. 整合所有特徵
# ─────────────────────────────────────────────

def build_all_features(
    user_info: pd.DataFrame,
    twd: pd.DataFrame,
    crypto: pd.DataFrame,
    trading: pd.DataFrame,
    swap: pd.DataFrame,
) -> pd.DataFrame:
    f_user    = build_user_features(user_info)
    f_twd     = build_twd_features(twd)
    f_crypto  = build_crypto_features(crypto)
    f_trading = build_trading_features(trading, swap)
    f_ip      = build_ip_features(twd, crypto, trading)
    f_vel     = build_velocity_features(twd, crypto)
    f_graph   = build_graph_features(crypto, user_info)
    f_cross   = build_cross_table_features(twd, crypto, trading, swap)

    feat = (
        f_user
        .join(f_twd,     how="left")
        .join(f_crypto,  how="left")
        .join(f_trading, how="left")
        .join(f_ip,      how="left")
        .join(f_vel,     how="left")
        .join(f_graph,   how="left")
        .join(f_cross,   how="left")
    ).fillna(0)

    # 複合風險分數（手工特徵交叉）
    feat["composite_risk_score"] = (
        feat["twd_withdraw_ratio"]          * 0.25 +
        feat["ip_night_ratio"]              * 0.15 +
        feat["crypto_currency_diversity"]   * 0.10 +
        feat["career_income_risk"]          * 0.20 +
        (1 - feat["has_kyc_level2"])        * 0.30
    ).clip(0, 1)

    return feat