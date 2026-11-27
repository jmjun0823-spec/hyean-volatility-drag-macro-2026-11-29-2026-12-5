from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from config.universe_config import AnalysisConfig, DEFAULT_CONFIG


def build_override_from_prices(
    close_df: pd.DataFrame,
    ticker: str,
    volume_df: Optional[pd.DataFrame] = None,
    vol_window: int = 10,
    cfg: AnalysisConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    종가 및 선택적 거래량 시계열로부터 `load_or_generate_panel(raw_override_df=...)`에
    주입할 실측 검증 패널을 생성합니다.
    """
    if ticker not in close_df.columns:
        raise KeyError(f"종가 데이터에 '{ticker}' 컬럼이 없습니다: {list(close_df.columns)}")

    close_s = pd.to_numeric(close_df[ticker], errors="coerce").dropna().sort_index()
    if len(close_s) < 2:
        raise ValueError(f"'{ticker}' 수익률 계산을 위해 최소 2개 이상의 종가 데이터가 필요합니다.")

    ret = close_s.pct_change().dropna()
    base_vol = float(ret.std(ddof=1))
    if not np.isfinite(base_vol) or base_vol <= 1e-12:
        base_vol = 0.01

    rolling_vol = ret.rolling(window=vol_window, min_periods=1).std(ddof=1).fillna(base_vol)
    vol_mult = np.clip((rolling_vol / base_vol).to_numpy(dtype=float), 0.5, 4.0)

    if volume_df is not None and ticker in volume_df.columns:
        vol_s = pd.to_numeric(volume_df[ticker], errors="coerce").reindex(ret.index).ffill().bfill()
        vol_avg = vol_s.rolling(window=vol_window, min_periods=1).mean().replace(0, np.nan)
        vol_ratio = (vol_s / vol_avg).fillna(1.0).to_numpy(dtype=float)
        imbalance = np.clip((vol_ratio - 0.7) / 1.3, 0.0, 1.0)
    else:
        imbalance = np.clip(0.15 + 0.20 * np.maximum(vol_mult - 1.0, 0.0), 0.0, 1.0)

    spread_bps = cfg.base_spread_bps + 5.0 * np.maximum(vol_mult - 1.0, 0.0)
    dates_str = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10] for d in ret.index]

    return pd.DataFrame(
        {
            "path_id": 0,
            "regime": f"market_{ticker}",
            "date": dates_str,
            "event_name": "observed_session",
            "event_category": "market_data",
            "underlying_return": np.clip(ret.to_numpy(dtype=float), -0.35, 0.35),
            "vol_multiplier": vol_mult,
            "spread_bps": spread_bps.astype(float),
            "order_imbalance": imbalance.astype(float),
        }
    )


def fetch_yfinance_override_panel(
    ticker: str = "^GSPC",
    start: Optional[str] = None,
    end: Optional[str] = None,
    cfg: AnalysisConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """yfinance에서 실측 시세 데이터를 내려받아 검증용 오버라이드 패널을 반환합니다."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance가 설치되어 있지 않습니다. `pip install yfinance` 후 실행하세요.") from exc

    s_date = start or cfg.window_start
    e_date = end or cfg.window_end
    hist = yf.download(ticker, start=s_date, end=e_date, auto_adjust=True, progress=False)
    if hist is None or hist.empty:
        raise RuntimeError(f"'{ticker}' ({s_date} ~ {e_date}) 데이터를 수신하지 못했습니다.")

    if isinstance(hist.columns, pd.MultiIndex):
        close_s = hist["Close"][ticker] if ticker in hist["Close"].columns else hist["Close"].iloc[:, 0]
        vol_s = hist["Volume"][ticker] if ("Volume" in hist.columns and ticker in hist["Volume"].columns) else None
    else:
        close_s = hist["Close"]
        vol_s = hist["Volume"] if "Volume" in hist.columns else None

    close_df = pd.DataFrame({ticker: close_s})
    volume_df = pd.DataFrame({ticker: vol_s}) if vol_s is not None else None
    return build_override_from_prices(close_df=close_df, ticker=ticker, volume_df=volume_df, cfg=cfg)
