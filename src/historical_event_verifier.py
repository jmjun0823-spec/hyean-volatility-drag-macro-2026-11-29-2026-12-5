from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def fetch_historical_etf_and_index_panel(
    start: str = "2012-01-01",
    end: str = "2026-11-27",
) -> pd.DataFrame:
    """
    yfinance에서 실제 S&P 500(^GSPC), 2x(SSO), 3x(UPRO), 성장주(QQQ),
    배당·현금흐름 방어주(SCHD), 미국채 10년물 금리(^TNX) 데이터를 내려받아
    실측 검증용 일간 패널을 생성합니다.
    """
    import yfinance as yf

    tickers = ["^GSPC", "SSO", "UPRO", "QQQ", "SCHD", "^TNX"]
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError("실측 시장 데이터(yfinance) 다운로드에 실패했습니다.")

    close = raw["Close"].dropna(how="all").ffill().dropna()
    open_p = raw["Open"].reindex(close.index).ffill().dropna()
    vol = raw["Volume"].reindex(close.index).ffill().fillna(0.0)

    tnx_diff_bps = close["^TNX"].diff().fillna(0.0) * 100.0
    price_cols = ["^GSPC", "SSO", "UPRO", "QQQ", "SCHD"]
    ret = close[price_cols].pct_change().dropna()
    tnx_diff_bps = tnx_diff_bps.loc[ret.index]

    open_gap = (open_p[price_cols].loc[ret.index] / close[price_cols].shift(1).loc[ret.index] - 1.0).fillna(0.0)
    tracking_diff_3x_bps = (ret["UPRO"] - 3.0 * ret["^GSPC"]).abs() * 10000.0
    opening_gap_bps = open_gap["UPRO"].abs() * 10000.0
    vol_ratio = (
        vol["UPRO"].loc[ret.index]
        / vol["UPRO"].loc[ret.index].rolling(20, min_periods=1).mean()
    ).fillna(1.0)

    rolling_vol_20d = ret["^GSPC"].rolling(20, min_periods=5).std(ddof=1).bfill()
    full_vol = float(ret["^GSPC"].std(ddof=1))
    vol_regime_mult = (rolling_vol_20d / max(full_vol, 1e-8)).clip(0.5, 4.0)

    return pd.DataFrame(
        {
            "date": [d.strftime("%Y-%m-%d") for d in ret.index],
            "month": [d.month for d in ret.index],
            "is_month_end": ret.index.is_month_end,
            "is_dec_first_week": [(d.month == 12 and d.day <= 7) for d in ret.index],
            "sp500_ret": ret["^GSPC"].to_numpy(),
            "sso_2x_ret": ret["SSO"].to_numpy(),
            "upro_3x_ret": ret["UPRO"].to_numpy(),
            "growth_qqq_ret": ret["QQQ"].to_numpy(),
            "defensive_schd_ret": ret["SCHD"].to_numpy(),
            "tnx_yield_change_bps": tnx_diff_bps.to_numpy(),
            "tracking_diff_3x_bps": tracking_diff_3x_bps.to_numpy(),
            "opening_gap_3x_bps": opening_gap_bps.to_numpy(),
            "volume_surge_ratio": vol_ratio.to_numpy(),
            "vol_regime_mult": vol_regime_mult.to_numpy(),
        }
    )


def verify_empirical_market_claims(
    market_df: pd.DataFrame,
    window_days: int = 5,
    chop_block_days: int = 20,
) -> Dict[str, object]:
    """
    실측 데이터에서:
    1) 미국채 10년물 금리(^TNX) 급등일(상위 10%) 배당·현금흐름(SCHD) vs 고밸류 성장(QQQ) 방어력
    2) 5거래일 및 20거래일 횡보 구간 UPRO(3x) 실측 드래그 H0 검정
    3) 고변동성 상위 10% vs 평시 하위 50% 3x 추적 괴리율(bps) 및 시초가 갭(bps) 검정
    """
    df = market_df.copy().reset_index(drop=True)
    def_col = "defensive_schd_ret" if "defensive_schd_ret" in df.columns else "defensive_xlp_ret"

    # 1) 금리 급등일(상위 10%) 기준 방어주 vs 성장주 초과수익률 검정
    if "tnx_yield_change_bps" in df.columns and df["tnx_yield_change_bps"].notna().sum() >= 30:
        rate_q90 = float(df["tnx_yield_change_bps"].dropna().quantile(0.90))
        stress = df[df["tnx_yield_change_bps"] >= rate_q90].dropna(subset=[def_col, "growth_qqq_ret"])
    else:
        vol_q90_tmp = float(df["vol_regime_mult"].quantile(0.90))
        stress = df[df["vol_regime_mult"] >= vol_q90_tmp].dropna(subset=[def_col, "growth_qqq_ret"])

    if len(stress) >= 5:
        def_spread = (stress[def_col] - stress["growth_qqq_ret"]) * 10000.0
        t_rate, p_rate = stats.ttest_1samp(def_spread, popmean=0.0, alternative="greater")
        rate_outperf_bps = float(def_spread.mean())
    else:
        t_rate, p_rate, rate_outperf_bps = float("nan"), float("nan"), float("nan")

    # 2a) 5거래일 횡보/교차등락 구간(|sp500_5d| < 2.0%) 조건부 H0 검정
    n_5d = len(df) // window_days
    tr_5d = df.iloc[: n_5d * window_days].copy()
    tr_5d["block_id"] = np.repeat(np.arange(n_5d), window_days)
    blk_5d = tr_5d.groupby("block_id", as_index=False).agg(
        sp500_5d=("sp500_ret", lambda s: float(np.prod(1.0 + s.dropna().to_numpy()) - 1.0)),
        upro_5d=("upro_3x_ret", lambda s: float(np.prod(1.0 + s.dropna().to_numpy()) - 1.0)),
    )
    blk_5d["drag_3x_5d_pct"] = (blk_5d["upro_5d"] - 3.0 * blk_5d["sp500_5d"]) * 100.0
    chop_5d = blk_5d[blk_5d["sp500_5d"].abs() < 0.02]["drag_3x_5d_pct"].dropna()
    if len(chop_5d) >= 5:
        t_chop5, p_chop5 = stats.ttest_1samp(chop_5d, popmean=0.0, alternative="less")
        chop5_mean = float(chop_5d.mean())
    else:
        t_chop5, p_chop5, chop5_mean = float("nan"), float("nan"), float("nan")

    # 2b) 20거래일 횡보 구간(|sp500_20d| < 3.0%) 조건부 H0 검정
    n_20d = len(df) // chop_block_days
    tr_20d = df.iloc[: n_20d * chop_block_days].copy()
    tr_20d["b20"] = np.repeat(np.arange(n_20d), chop_block_days)
    blk_20d = tr_20d.groupby("b20", as_index=False).agg(
        sp500_20d=("sp500_ret", lambda s: float(np.prod(1.0 + s.dropna().to_numpy()) - 1.0)),
        upro_20d=("upro_3x_ret", lambda s: float(np.prod(1.0 + s.dropna().to_numpy()) - 1.0)),
    )
    blk_20d["drag_3x_20d_pct"] = (blk_20d["upro_20d"] - 3.0 * blk_20d["sp500_20d"]) * 100.0
    sideways_20d = blk_20d[blk_20d["sp500_20d"].abs() < 0.03]["drag_3x_20d_pct"].dropna()
    if len(sideways_20d) >= 5:
        _, p_chop20 = stats.ttest_1samp(sideways_20d, popmean=0.0, alternative="less")
        sideways20_mean = float(sideways_20d.mean())
    else:
        p_chop20, sideways20_mean = float("nan"), float("nan")

    # 3) 고변동성 상위 10% vs 평시 하위 50% 3x 추적 괴리율 및 시초가 갭 검정
    vol_q90 = float(df["vol_regime_mult"].quantile(0.90))
    vol_q50 = float(df["vol_regime_mult"].quantile(0.50))
    high_vol_days = df[df["vol_regime_mult"] >= vol_q90]
    calm_days = df[df["vol_regime_mult"] <= vol_q50]

    high_div = high_vol_days["tracking_diff_3x_bps"].dropna()
    calm_div = calm_days["tracking_diff_3x_bps"].dropna()
    if len(high_div) >= 5 and len(calm_div) >= 5:
        _, p_div = stats.ttest_ind(high_div, calm_div, equal_var=False, alternative="greater")
        div_high = float(high_div.mean())
        div_calm = float(calm_div.mean())
    else:
        p_div, div_high, div_calm = float("nan"), float("nan"), float("nan")

    gap_high = float(high_vol_days["opening_gap_3x_bps"].dropna().mean())
    gap_calm = float(calm_days["opening_gap_3x_bps"].dropna().mean())

    return {
        "n_trading_days": int(len(df)),
        "rate_shock_defensive_outperf_bps": rate_outperf_bps,
        "rate_shock_t_stat": float(t_rate),
        "rate_shock_p_value": float(p_rate),
        "chop_5d_count": int(len(chop_5d)),
        "chop_5d_mean_drag_3x_pct": chop5_mean,
        "chop_5d_t_stat": float(t_chop5),
        "chop_5d_p_value": float(p_chop5),
        "sideways_20d_count": int(len(sideways_20d)),
        "sideways_20d_mean_drag_3x_pct": sideways20_mean,
        "sideways_20d_p_value": float(p_chop20),
        "high_vol_tracking_div_bps": div_high,
        "calm_tracking_div_bps": div_calm,
        "tracking_div_surge_p_val": float(p_div),
        "high_vol_opening_gap_bps": gap_high,
        "calm_opening_gap_bps": gap_calm,
    }
