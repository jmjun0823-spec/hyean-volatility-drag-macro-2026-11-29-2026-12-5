from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def fetch_historical_etf_and_index_panel(
    start: str = "2010-01-01",
    end: str = "2026-11-27",
) -> pd.DataFrame:
    """
    yfinance에서 실제 ^GSPC, SSO(2x), UPRO(3x), QQQ(성장), XLP(방어) 실측 데이터를 내려받아
    일간 수익률, 3x 추적 괴리율(bps), 시초가 갭(bps), 변동성 레짐 배수를 산출합니다.
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
            "defensive_xlp_ret": ret["XLP"].to_numpy(),
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
    실측 시장 데이터에서:
    1) 5일 및 20일 블록 기준 UPRO(3x) 실측 드래그 및 횡보(|sp_cum| < 2%)·고변동성 드래그 검증
    2) 고변동성 구간(>= q75) 방어주(XLP) vs 성장주(QQQ) 일간 스프레드 실측치 검증
    3) 고변동성일(>= q90) vs 평시(<= q50) 3x 추적 괴리율(bps) 및 시초가 갭(bps) 검증
    """
    df = market_df.copy().reset_index(drop=True)

    # 1a) 5거래일 블록 검정
    n_5d = len(df) // window_days
    tr_5d = df.iloc[: n_5d * window_days].copy()
    tr_5d["block_id"] = np.repeat(np.arange(n_5d), window_days)
    blk_5d = tr_5d.groupby("block_id", as_index=False).agg(
        sp500_5d=("sp500_ret", lambda s: float(np.prod(1.0 + s.to_numpy()) - 1.0)),
        upro_5d=("upro_3x_ret", lambda s: float(np.prod(1.0 + s.to_numpy()) - 1.0)),
    )
    blk_5d["drag_3x_5d_pct"] = (blk_5d["upro_5d"] - 3.0 * blk_5d["sp500_5d"]) * 100.0
    t_5d, p_5d = stats.ttest_1samp(blk_5d["drag_3x_5d_pct"], popmean=0.0)

    # 1b) 20거래일 블록 및 횡보/고변동성 드래그 검증
    n_20d = len(df) // chop_block_days
    tr_20d = df.iloc[: n_20d * chop_block_days].copy()
    tr_20d["b20"] = np.repeat(np.arange(n_20d), chop_block_days)
    blk_20d = tr_20d.groupby("b20", as_index=False).agg(
        sp500_20d=("sp500_ret", lambda s: float(np.prod(1.0 + s.to_numpy()) - 1.0)),
        upro_20d=("upro_3x_ret", lambda s: float(np.prod(1.0 + s.to_numpy()) - 1.0)),
        vol_20d=("sp500_ret", lambda s: float(np.std(s.to_numpy(), ddof=1))),
    )
    blk_20d["drag_3x_20d_pct"] = (blk_20d["upro_20d"] - 3.0 * blk_20d["sp500_20d"]) * 100.0
    sideways_20d = blk_20d[blk_20d["sp500_20d"].abs() < 0.02]
    high_vol_20d = blk_20d[blk_20d["vol_20d"] >= blk_20d["vol_20d"].quantile(0.75)]
    low_vol_20d = blk_20d[blk_20d["vol_20d"] <= blk_20d["vol_20d"].quantile(0.25)]

    # 2) 고변동성 상위 25% 구간 XLP - QQQ 일간 스프레드 실측치
    vol_q75 = float(df["vol_regime_mult"].quantile(0.75))
    stress_days = df[df["vol_regime_mult"] >= vol_q75]
    defensive_spread_stress_bps = float(
        (stress_days["defensive_xlp_ret"] - stress_days["growth_qqq_ret"]).mean() * 10000.0
    )

    # 3) 고변동성 상위 10% vs 평시 하위 50% 3x 추적 괴리율 및 시초가 갭
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
        "n_5d_blocks": int(n_5d),
        "empirical_mean_5d_drag_3x_pct": float(blk_5d["drag_3x_5d_pct"].mean()),
        "empirical_h0_t_stat": float(t_5d),
        "empirical_h0_p_value": float(p_5d),
        "empirical_h0_rejected": bool(p_5d < 0.05),
        "empirical_mean_20d_drag_3x_pct": float(blk_20d["drag_3x_20d_pct"].mean()),
        "empirical_sideways_20d_drag_3x_pct": float(sideways_20d["drag_3x_20d_pct"].mean()) if not sideways_20d.empty else float("nan"),
        "empirical_high_vol_20d_drag_3x_pct": float(high_vol_20d["drag_3x_20d_pct"].mean()),
        "empirical_low_vol_20d_drag_3x_pct": float(low_vol_20d["drag_3x_20d_pct"].mean()),
        "stress_defensive_outperf_bps": defensive_spread_stress_bps,
        "high_vol_tracking_div_bps": div_high,
        "calm_tracking_div_bps": div_calm,
        "tracking_div_surge_p_val": float(p_div),
        "high_vol_opening_gap_bps": gap_high,
        "calm_opening_gap_bps": gap_calm,
    }
