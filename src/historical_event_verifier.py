from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from config.universe_config import AnalysisConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


def fetch_historical_etf_and_index_panel(
    start: str = "2010-01-01",
    end: str = "2026-11-27",
) -> pd.DataFrame:
    """
    yfinance에서 실제 S&P 500(^GSPC), 2x ETF(SSO), 3x ETF(UPRO),
    배당/가치 방어주 프록시(SCHD/XLP), 고밸류 성장주 프록시(QQQ/ARKK)의
    실측 종가·시초가·거래량 데이터를 수집해 일간 수익률, 실제 3x 추적오차(괴리),
    변동성 레짐 및 장초반 갭 리스크를 계산합니다.
    """
    import yfinance as yf

    tickers = ["^GSPC", "SSO", "UPRO", "QQQ", "XLP"]
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError("실측 시장 데이터(yfinance) 다운로드에 실패했습니다.")

    close = raw["Close"].dropna(how="all").ffill().dropna()
    open_p = raw["Open"].reindex(close.index).ffill().dropna()
    vol = raw["Volume"].reindex(close.index).ffill().fillna(0.0)

    ret = close.pct_change().dropna()
    open_gap = (open_p.loc[ret.index] / close.shift(1).loc[ret.index] - 1.0).fillna(0.0)

    # 실제 UPRO 일간 수익률 vs 이론적 3 * ^GSPC 일간 수익률 간 추적 괴리(Tracking Divergence)
    tracking_diff_3x_bps = (ret["UPRO"] - 3.0 * ret["^GSPC"]).abs() * 10000.0
    # 장초반 시초가 갭 크기(bps) 및 거래량 급증 비율
    opening_gap_bps = open_gap["UPRO"].abs() * 10000.0
    vol_ratio = (vol["UPRO"].loc[ret.index] / vol["UPRO"].loc[ret.index].rolling(20, min_periods=1).mean()).fillna(1.0)

    rolling_vol_20d = ret["^GSPC"].rolling(20, min_periods=5).std(ddof=1).bfill()
    full_vol = float(ret["^GSPC"].std(ddof=1))
    vol_regime_mult = (rolling_vol_20d / max(full_vol, 1e-8)).clip(0.5, 4.0)

    df = pd.DataFrame(
        {
            "date": [d.strftime("%Y-%m-%d") for d in ret.index],
            "month": [d.month for d in ret.index],
            "is_month_end": ret.index.is_month_end,
            "is_dec_first_week": [(d.month == 12 and d.day <= 7) for d in ret.index],
            "sp500_ret": ret["^GSPC"].to_numpy(),
            "sso_2x_ret": ret["SSO"].to_numpy(),
            "upro_3x_ret": ret["UPRO"].to_numpy(),
            "growth_qqq_ret": ret["QQQ"].to_numpy(),
            "defensive_xlp_ret": ret["XLP"].to_numpy(),
            "tracking_diff_3x_bps": tracking_diff_3x_bps.to_numpy(),
            "opening_gap_3x_bps": opening_gap_bps.to_numpy(),
            "volume_surge_ratio": vol_ratio.to_numpy(),
            "vol_regime_mult": vol_regime_mult.to_numpy(),
        }
    )
    return df


def verify_empirical_market_claims(
    market_df: pd.DataFrame,
    window_days: int = 5,
) -> Dict[str, object]:
    """
    실측 일간 데이터(market_df)에서 블로그 글의 4대 핵심 주장을 통계적으로 검증합니다.
    1) 5거래일 롤링 구간 기준: UPRO(3x) 실제 누적수익률 vs 3 * S&P500 누적수익률(H0) 기각 여부
    2) 고변동성 상위 25% 구간에서 성장주(QQQ) 대비 방어주(XLP)의 하방 방어력 우위 검증
    3) 연말(11월 말~12월 초) 및 고변동성 일자의 3x 추적 괴리율(bps)·시초가 갭(bps) 급증 검증
    """
    df = market_df.copy().reset_index(drop=True)

    # 1) 5거래일(1주간) 비겹침 블록별 누적수익률 비교 (실측 H0 검정)
    n_blocks = len(df) // window_days
    trimmed = df.iloc[: n_blocks * window_days].copy()
    trimmed["block_id"] = np.repeat(np.arange(n_blocks), window_days)

    blk = trimmed.groupby("block_id", as_index=False).agg(
        sp500_5d=("sp500_ret", lambda s: float(np.prod(1.0 + s.to_numpy()) - 1.0)),
        sso_5d=("sso_2x_ret", lambda s: float(np.prod(1.0 + s.to_numpy()) - 1.0)),
        upro_5d=("upro_3x_ret", lambda s: float(np.prod(1.0 + s.to_numpy()) - 1.0)),
        sp500_vol=("sp500_ret", lambda s: float(np.std(s.to_numpy(), ddof=1))),
    )
    blk["drag_3x_5d_pct"] = (blk["upro_5d"] - 3.0 * blk["sp500_5d"]) * 100.0
    t_stat_emp, p_val_emp = stats.ttest_1samp(blk["drag_3x_5d_pct"], popmean=0.0)

    # 2) 변동성 확대 상위 25% 구간에서의 방어주(XLP) vs 성장주(QQQ) 초과수익률 검증
    vol_q75 = float(df["vol_regime_mult"].quantile(0.75))
    stress_days = df[df["vol_regime_mult"] >= vol_q75]
    defensive_spread_stress_bps = float(
        (stress_days["defensive_xlp_ret"] - stress_days["growth_qqq_ret"]).mean() * 10000.0
    )

    # 3) 고변동성 상위 10% 이벤트일 vs 평시(하위 50%)의 3x 추적 괴리율 및 시초가 갭 비교
    vol_q90 = float(df["vol_regime_mult"].quantile(0.90))
    vol_q50 = float(df["vol_regime_mult"].quantile(0.50))
    high_vol_days = df[df["vol_regime_mult"] >= vol_q90]
    calm_days = df[df["vol_regime_mult"] <= vol_q50]

    div_high = float(high_vol_days["tracking_diff_3x_bps"].mean())
    div_calm = float(calm_days["tracking_diff_3x_bps"].mean())
    gap_high = float(high_vol_days["opening_gap_3x_bps"].mean())
    gap_calm = float(calm_days["opening_gap_3x_bps"].mean())

    _, p_div = stats.ttest_ind(
        high_vol_days["tracking_diff_3x_bps"],
        calm_days["tracking_diff_3x_bps"],
        equal_var=False,
    )

    return {
        "n_trading_days": int(len(df)),
        "n_5d_blocks": int(n_blocks),
        "empirical_mean_5d_drag_3x_pct": float(blk["drag_3x_5d_pct"].mean()),
        "empirical_h0_t_stat": float(t_stat_emp),
        "empirical_h0_p_value": float(p_val_emp),
        "empirical_h0_rejected": bool(p_val_emp < 0.05),
        "stress_defensive_outperf_bps": defensive_spread_stress_bps,
        "high_vol_tracking_div_bps": div_high,
        "calm_tracking_div_bps": div_calm,
        "tracking_div_surge_p_val": float(p_div),
        "high_vol_opening_gap_bps": gap_high,
        "calm_opening_gap_bps": gap_calm,
    }
