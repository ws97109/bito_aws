"""
Feature Engineering Module
論文級特徵工程：從五張資料表提取多維度風險特徵
"""
import pandas as pd
import numpy as np
from typing import Optional
import ipaddress


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
    # 數據中已有 age 欄位，不需要從 birthday 計算

    # KYC 完成狀態
    df["has_kyc_level2"] = df["level2_finished_at"].notna().astype(int)

    # KYC 完成速度（秒）—— 過快可能是機器人
    df["kyc_speed_sec"] = (
        df["level2_finished_at"] - df["level1_finished_at"]
    ).dt.total_seconds().clip(lower=0)

    # 帳號年齡（天）
    df["account_age_days"] = (now - df["confirmed_at"]).dt.days.clip(lower=0)

    # 年齡 - 使用已有的 age 欄位並進行清理
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce").clip(lower=0, upper=120)
    else:
        df["age"] = 0  # 如果沒有 age 欄位，設為 0

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

    # 出 / 入比率（接近 1 表示幾乎全部提出）
    feat["twd_withdraw_ratio"] = feat["twd_wit_sum"] / (feat["twd_dep_sum"] + 1e-9)

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

    # 內轉對象數（relation_user_id 不重複數）
    internal = df[df["sub_kind"] == 1]
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
# 7. 整合所有特徵
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

    feat = (
        f_user
        .join(f_twd,     how="left")
        .join(f_crypto,  how="left")
        .join(f_trading, how="left")
        .join(f_ip,      how="left")
        .join(f_vel,     how="left")
    ).fillna(0)

    # ═══════════════════════════════════════════════════════════
    # 高階交互特徵（提升區分度）
    # ═══════════════════════════════════════════════════════════

    # 1. 快速 KYC + 快速提領（疑似洗錢模式）
    feat["quick_kyc_quick_withdraw"] = (
        (feat["kyc_speed_sec"] < 3600) &              # KYC < 1小時
        (feat["fund_stay_sec"] < 86400) &             # 資金停留 < 1天
        (feat["twd_withdraw_ratio"] > 0.8)            # 提領比例 > 80%
    ).astype(int)

    # 2. 新帳號高風險行為
    feat["new_account_high_risk"] = (
        (feat["account_age_days"] < 30) &             # 新帳號 < 30天
        (
            (feat["ip_unique_count"] > 5) |           # 多 IP
            (feat["crypto_currency_diversity"] > 3) | # 多幣種
            (feat["twd_withdraw_ratio"] > 0.7)        # 高提領
        )
    ).astype(int)

    # 3. 深夜高頻交易（異常時段）
    feat["night_high_frequency"] = (
        feat["ip_night_ratio"] *
        np.log1p(feat["twd_dep_count"] + feat["crypto_wit_count"])
    )

    # 4. 大額快速流轉
    feat["large_fast_turnover"] = (
        np.log1p(feat["swap_sum"] + feat["total_trading_volume"]) /
        np.log1p(feat["fund_stay_sec"] + 1)
    )

    # 5. 結構化交易 + 快速提領
    feat["smurf_withdraw_pattern"] = (
        feat["twd_smurf_flag"] * feat["twd_withdraw_ratio"]
    )

    # 6. 多幣種 × 多協議（複雜化追蹤）
    feat["currency_protocol_complexity"] = (
        feat["crypto_currency_diversity"] *
        feat["crypto_protocol_diversity"]
    )

    # 7. 高風險職業 × 低 KYC
    feat["risky_career_low_kyc"] = (
        (feat["is_high_risk_career"] == 1) &
        (feat["has_kyc_level2"] == 0)
    ).astype(int)

    # 8. 異常資金流速度（提領金額 / 時間）
    feat["withdraw_velocity"] = (
        feat["twd_wit_sum"] / (feat["fund_stay_sec"] + 1)
    )

    # 9. IP 多樣性 × 交易頻率
    feat["ip_diversity_frequency"] = (
        feat["ip_unique_count"] *
        np.log1p(feat["twd_dep_count"] + feat["twd_wit_count"])
    )

    # 10. 年齡與資金規模不匹配（年輕但大額）
    feat["age_volume_mismatch"] = np.where(
        feat["age"] < 25,
        np.log1p(feat["twd_dep_sum"] + feat["crypto_wit_sum"]),
        0
    )

    # 更新複合風險分數（整合新特徵）
    feat["composite_risk_score"] = (
        feat["twd_withdraw_ratio"]          * 0.15 +
        feat["ip_night_ratio"]              * 0.10 +
        feat["crypto_currency_diversity"]   * 0.08 +
        feat["career_income_risk"]          * 0.15 +
        (1 - feat["has_kyc_level2"])        * 0.20 +
        feat["quick_kyc_quick_withdraw"]    * 0.12 +
        feat["new_account_high_risk"]       * 0.10 +
        feat["smurf_withdraw_pattern"]      * 0.10
    ).clip(0, 1)

    return feat