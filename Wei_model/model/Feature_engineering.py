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

    # 性別（強信號：黑名單中女性佔比遠高於正常用戶）
    df["sex"] = pd.to_numeric(df["sex"], errors="coerce").fillna(0).astype(int)
    df["is_female"] = (df["sex"] == 2).astype(int)

    # 高風險職業 / 收入來源
    df["is_high_risk_career"] = df["career"].isin(HIGH_RISK_CAREERS).astype(int)
    df["is_high_risk_income"] = df["income_source"].isin(HIGH_RISK_INCOME).astype(int)

    # 職業 × 收入組合風險（同時命中）
    df["career_income_risk"] = (
        df["is_high_risk_career"] & df["is_high_risk_income"]
    ).astype(int)

    # 職業群組（target encoding 用不了，改用 frequency-based: 低頻職業更可疑）
    career_counts = df["career"].value_counts()
    df["career_freq"] = df["career"].map(career_counts).fillna(0)

    # 來源管道
    df["is_app_user"] = (df["user_source"] == 1).astype(int)

    # 註冊時間特徵（晚間註冊可能是機器人/代辦）
    df["reg_hour"] = df["confirmed_at"].dt.hour
    df["reg_is_night"] = df["reg_hour"].between(0, 5).astype(int)
    df["reg_is_weekend"] = df["confirmed_at"].dt.dayofweek.isin([5, 6]).astype(int)

    # KYC Level 1 到 Level 2 之間的天數
    df["kyc_gap_days"] = (
        df["level2_finished_at"] - df["level1_finished_at"]
    ).dt.days.clip(lower=0)

    # 註冊到第一次 KYC 的天數（快速完成 KYC 可疑）
    df["reg_to_kyc1_days"] = (
        df["level1_finished_at"] - df["confirmed_at"]
    ).dt.days.clip(lower=0)

    feature_cols = [
        "user_id",
        "has_kyc_level2", "kyc_speed_sec", "account_age_days",
        "age", "is_female", "is_high_risk_career", "is_high_risk_income",
        "career_income_risk", "career_freq", "is_app_user",
        "reg_hour", "reg_is_night", "reg_is_weekend",
        "kyc_gap_days", "reg_to_kyc1_days",
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
# 9. 行為紅旗特徵（高區分力）
# ─────────────────────────────────────────────

def build_red_flag_features(
    twd: pd.DataFrame,
    crypto: pd.DataFrame,
    user_info: pd.DataFrame,
) -> pd.DataFrame:
    """
    基於 AML 紅旗文獻設計的高區分力特徵：
    - deposit_to_first_withdraw_hours: 入金到第一筆提領的時間
    - twd_to_crypto_out_ratio: 法幣入金 / 加密貨幣出幣比
    - tx_amount_cv: 交易金額變異係數（Smurfing 偵測）
    - rapid_kyc_then_trade: KYC 後 48 小時內大額交易旗標
    - crypto_out_in_ratio: 加密貨幣出幣/入幣比
    - same_day_in_out_count: 同天入金又出金次數
    """
    t = twd.copy()
    c = crypto.copy()
    t["ts"] = pd.to_datetime(t["created_at"])
    c["ts"] = pd.to_datetime(c["created_at"])
    t["amount"] = t["ori_samount"] * SCALE
    c["amount"] = c["ori_samount"] * SCALE * c["twd_srate"] * SCALE

    all_user_ids = user_info["user_id"].unique()

    # ── 1. deposit_to_first_withdraw_hours ──
    # 每個用戶的「第一筆入金 → 第一筆提領」間隔
    records_dfwh = []
    for df in [t, c]:
        for uid, grp in df.groupby("user_id"):
            grp = grp.sort_values("ts")
            first_dep = grp[grp["kind"] == 0]["ts"].min()
            first_wit = grp[grp["kind"] == 1]["ts"].min()
            if pd.isna(first_dep) or pd.isna(first_wit):
                continue
            if first_wit >= first_dep:
                hours = (first_wit - first_dep) / np.timedelta64(1, "h")
                records_dfwh.append({"user_id": uid, "dep_to_first_wit_hours": hours})

    if records_dfwh:
        dfwh = pd.DataFrame(records_dfwh).groupby("user_id")["dep_to_first_wit_hours"].min()
    else:
        dfwh = pd.Series(dtype=float, name="dep_to_first_wit_hours")
    dfwh.index.name = "user_id"

    # ── 2. twd_to_crypto_out_ratio ──
    # 法幣入金總額 / 加密貨幣出幣總額（抓「法幣→幣→鏈上」洗錢鏈）
    twd_dep = t[t["kind"] == 0].groupby("user_id")["amount"].sum().rename("_twd_dep")
    crypto_wit = c[c["kind"] == 1].groupby("user_id")["amount"].sum().rename("_crypto_wit")
    ratio_df = twd_dep.to_frame().join(crypto_wit, how="outer").fillna(0)
    twd_crypto_ratio = (
        ratio_df["_twd_dep"] / (ratio_df["_crypto_wit"] + 1e-9)
    ).clip(upper=100).rename("twd_to_crypto_out_ratio")

    # ── 3. tx_amount_cv ──
    # 所有交易金額的變異係數，Smurfing 的 CV 會很低
    all_amounts = pd.concat([
        t[["user_id", "amount"]],
        c[["user_id", "amount"]],
    ], ignore_index=True)
    amount_stats = all_amounts.groupby("user_id")["amount"].agg(["mean", "std", "count"])
    tx_amount_cv = (amount_stats["std"] / (amount_stats["mean"] + 1e-9)).rename("tx_amount_cv")
    # count >= 3 才有意義，否則設為 0
    tx_amount_cv[amount_stats["count"] < 3] = 0

    # ── 4. rapid_kyc_then_trade ──
    # KYC Level 2 完成後 48 小時內就有交易的旗標
    ui = user_info[["user_id", "level2_finished_at"]].copy()
    ui["kyc2_ts"] = pd.to_datetime(ui["level2_finished_at"])
    ui = ui.set_index("user_id")

    all_tx_ts = pd.concat([
        t[["user_id", "ts"]],
        c[["user_id", "ts"]],
    ], ignore_index=True)
    first_tx = all_tx_ts.groupby("user_id")["ts"].min().rename("first_tx_ts")

    kyc_trade = ui[["kyc2_ts"]].join(first_tx, how="left")
    kyc_trade["hours_after_kyc"] = (
        (kyc_trade["first_tx_ts"] - kyc_trade["kyc2_ts"]).dt.total_seconds() / 3600
    )
    rapid_kyc = (
        (kyc_trade["hours_after_kyc"].between(0, 48))
    ).astype(int).rename("rapid_kyc_then_trade")

    # ── 5. crypto_out_in_ratio ──
    crypto_dep = c[c["kind"] == 0].groupby("user_id")["amount"].sum().rename("_c_dep")
    crypto_out = c[c["kind"] == 1].groupby("user_id")["amount"].sum().rename("_c_out")
    coi = crypto_dep.to_frame().join(crypto_out, how="outer").fillna(0)
    crypto_out_in_ratio = (
        coi["_c_out"] / (coi["_c_dep"] + 1e-9)
    ).clip(upper=10).rename("crypto_out_in_ratio")

    # ── 6. same_day_in_out_count ──
    # 同天既有入金又有出金的天數
    for df in [t, c]:
        df["date"] = df["ts"].dt.date

    def count_same_day_in_out(df):
        dep_dates = df[df["kind"] == 0].groupby("user_id")["date"].apply(set)
        wit_dates = df[df["kind"] == 1].groupby("user_id")["date"].apply(set)
        both = dep_dates.to_frame("dep").join(wit_dates.rename("wit"), how="inner")
        return both.apply(lambda r: len(r["dep"] & r["wit"]), axis=1)

    sd_twd = count_same_day_in_out(t)
    sd_crypto = count_same_day_in_out(c)
    same_day = sd_twd.add(sd_crypto, fill_value=0).rename("same_day_in_out_count")

    # ── 合併 ──
    feat = pd.DataFrame(index=pd.Index(all_user_ids, name="user_id"))
    feat = feat.join(dfwh)
    feat = feat.join(twd_crypto_ratio)
    feat = feat.join(tx_amount_cv)
    feat = feat.join(rapid_kyc)
    feat = feat.join(crypto_out_in_ratio)
    feat = feat.join(same_day)
    feat = feat.fillna(0)

    return feat


# ─────────────────────────────────────────────
# 10. 交易時序特徵（transaction interval & burst）
# ─────────────────────────────────────────────

def build_temporal_features(
    twd: pd.DataFrame,
    crypto: pd.DataFrame,
    trading: pd.DataFrame,
    swap: pd.DataFrame,
    user_info: pd.DataFrame,
) -> pd.DataFrame:
    """
    時序模式特徵：
    - tx_interval_mean/std/min: 交易間隔統計（快速連續交易是紅旗）
    - tx_burst_count: 1 小時內 >= 5 筆交易的「爆發」次數
    - amount_p90/p10_ratio: 金額 90% vs 10% 分位比（均勻金額 = Smurfing）
    - active_days: 有交易的天數
    - active_day_ratio: 有交易天數 / 帳號年齡天數
    """
    all_user_ids = user_info["user_id"].unique()

    # 收集所有交易時間和金額
    frames = []
    for df, ts_col in [
        (twd, "created_at"), (crypto, "created_at"),
        (trading, "updated_at"), (swap, "created_at"),
    ]:
        tmp = df[["user_id"]].copy()
        tmp["ts"] = pd.to_datetime(df[ts_col])
        amt_col = None
        if "ori_samount" in df.columns:
            amt_col = "ori_samount"
        elif "trade_samount" in df.columns:
            amt_col = "trade_samount"
        elif "twd_samount" in df.columns:
            amt_col = "twd_samount"
        if amt_col:
            tmp["amount"] = pd.to_numeric(df[amt_col], errors="coerce") * SCALE
        else:
            tmp["amount"] = 0
        frames.append(tmp[["user_id", "ts", "amount"]])

    all_tx = pd.concat(frames, ignore_index=True).dropna(subset=["ts"])
    all_tx = all_tx.sort_values(["user_id", "ts"])

    # 交易間隔統計
    all_tx["prev_ts"] = all_tx.groupby("user_id")["ts"].shift(1)
    all_tx["interval_sec"] = (all_tx["ts"] - all_tx["prev_ts"]).dt.total_seconds()

    interval_stats = all_tx.groupby("user_id")["interval_sec"].agg(
        tx_interval_mean="mean",
        tx_interval_std="std",
        tx_interval_min="min",
        tx_interval_median="median",
    ).fillna(0)

    # 爆發偵測：1 小時滾動窗口內 >= 5 筆
    def count_bursts(grp):
        if len(grp) < 5:
            return 0
        ts_vals = grp["ts"].values
        count = 0
        for i in range(len(ts_vals)):
            window_end = ts_vals[i] + np.timedelta64(1, "h")
            n_in_window = np.searchsorted(ts_vals, window_end, side="right") - i
            if n_in_window >= 5:
                count += 1
        return count

    burst_counts = all_tx.groupby("user_id").apply(count_bursts).rename("tx_burst_count")

    # 金額分位數比（偵測金額是否過於均勻）
    def p90_p10_ratio(x):
        p90 = x.quantile(0.9)
        p10 = x.quantile(0.1)
        return p90 / (p10 + 1e-9)

    amount_ratio = (
        all_tx[all_tx["amount"] > 0]
        .groupby("user_id")["amount"]
        .apply(p90_p10_ratio)
        .clip(upper=1000)
        .rename("amount_p90_p10_ratio")
    )

    # 活躍天數
    all_tx["date"] = all_tx["ts"].dt.date
    active_days = all_tx.groupby("user_id")["date"].nunique().rename("active_days")

    feat = pd.DataFrame(index=pd.Index(all_user_ids, name="user_id"))
    feat = feat.join(interval_stats)
    feat = feat.join(burst_counts)
    feat = feat.join(amount_ratio)
    feat = feat.join(active_days)
    feat = feat.fillna(0)

    return feat


# ─────────────────────────────────────────────
# 11. 整合所有特徵
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
    f_red     = build_red_flag_features(twd, crypto, user_info)
    f_temporal = build_temporal_features(twd, crypto, trading, swap, user_info)

    feat = (
        f_user
        .join(f_twd,      how="left")
        .join(f_crypto,   how="left")
        .join(f_trading,  how="left")
        .join(f_ip,       how="left")
        .join(f_vel,      how="left")
        .join(f_graph,    how="left")
        .join(f_cross,    how="left")
        .join(f_red,      how="left")
        .join(f_temporal, how="left")
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