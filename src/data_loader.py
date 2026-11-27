from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config.universe_config import AnalysisConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

REQUIRED_OVERRIDE_COLS = {"path_id", "date", "underlying_return"}


def _generate_fallback_event_panel(
    cfg: AnalysisConfig,
    seed: int = 42,
    n_paths: int = 2000,
) -> pd.DataFrame:
    """
    외부 API 미연결 또는 재현 테스트 환경에서 이벤트 주간(2026-11-29 ~ 2026-12-05)
    기초지수 수익률, 변동성 배수, 스프레드, 주문 불균형 패널을 생성합니다.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=cfg.window_start, end=cfg.window_end)
    event_map = {ew.date_label: ew for ew in cfg.event_windows}

    rows: list[dict[str, object]] = []
    for path_id in range(n_paths):
        # 50% 노이즈성 횡보/교차등락 레짐, 50% 이벤트 변동성 레짐
        regime = "noisy_chop" if path_id % 2 == 0 else "event_volatility"
        for idx, dt in enumerate(dates):
            dt_str = dt.strftime("%Y-%m-%d")
            ew = event_map.get(dt_str)
            vol_mult = ew.expected_vol_multiplier if ew else 1.0
            spread_bps = ew.spread_bps_shock if ew else cfg.base_spread_bps
            base_imb = ew.flow_imbalance_shock if ew else 0.20

            if regime == "noisy_chop":
                sign = 1.0 if idx % 2 == 0 else -1.0
                mag = abs(rng.normal(0.018 * vol_mult, 0.003))
                daily_ret = sign * mag
            else:
                drift = 0.002 if dt_str != "2026-12-04" else -0.008
                daily_ret = float(rng.normal(drift, 0.015 * vol_mult))

            imbalance = float(np.clip(rng.normal(base_imb, 0.15), 0.0, 1.0))
            rows.append(
                {
                    "path_id": int(path_id),
                    "regime": regime,
                    "date": dt_str,
                    "event_name": ew.name if ew else "non_event_session",
                    "event_category": ew.category if ew else "baseline",
                    "underlying_return": float(np.clip(daily_ret, -0.25, 0.25)),
                    "vol_multiplier": float(vol_mult),
                    "spread_bps": float(spread_bps),
                    "order_imbalance": imbalance,
                }
            )

    return pd.DataFrame(rows)


def load_or_generate_panel(
    cfg: AnalysisConfig = DEFAULT_CONFIG,
    raw_override_df: Optional[pd.DataFrame] = None,
    use_cache: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    외부 데이터 오버라이드(raw_override_df) -> Parquet/CSV 캐시 -> 폴백 생성 순으로
    일관된 스키마의 검증 패널을 반환합니다.
    """
    if raw_override_df is not None:
        missing = REQUIRED_OVERRIDE_COLS - set(raw_override_df.columns)
        if missing:
            raise ValueError(f"raw_override_df 필수 컬럼 누락: {sorted(missing)}")

        df = raw_override_df.copy()
        if "path_id" not in df.columns:
            df["path_id"] = 0
        if "regime" not in df.columns:
            df["regime"] = "override_input"
        if "event_name" not in df.columns:
            df["event_name"] = "override_event"
        if "event_category" not in df.columns:
            df["event_category"] = "override"
        if "vol_multiplier" not in df.columns:
            df["vol_multiplier"] = 1.0
        if "spread_bps" not in df.columns:
            df["spread_bps"] = cfg.base_spread_bps
        if "order_imbalance" not in df.columns:
            df["order_imbalance"] = 0.20

        df["underlying_return"] = pd.to_numeric(df["underlying_return"], errors="raise").astype(float)
        df["vol_multiplier"] = pd.to_numeric(df["vol_multiplier"], errors="coerce").fillna(1.0).astype(float)
        df["spread_bps"] = pd.to_numeric(df["spread_bps"], errors="coerce").fillna(cfg.base_spread_bps).astype(float)
        df["order_imbalance"] = pd.to_numeric(df["order_imbalance"], errors="coerce").fillna(0.20).clip(0.0, 1.0).astype(float)
        return df.sort_values(["path_id", "date"]).reset_index(drop=True)

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = cfg.data_dir / "event_volatility_panel.parquet"
    csv_path = cfg.data_dir / "event_volatility_panel.csv"

    if use_cache and parquet_path.exists():
        try:
            cached = pd.read_parquet(parquet_path)
            if REQUIRED_OVERRIDE_COLS.issubset(cached.columns):
                return cached
        except Exception as exc:
            logger.warning("Parquet 캐시 로드 실패(%s), 폴백 패널을 재생성합니다.", exc)

    panel_df = _generate_fallback_event_panel(cfg=cfg, seed=seed)
    try:
        panel_df.to_parquet(parquet_path, index=False)
    except Exception as exc:
        logger.info("Parquet 저장 생략(%s), CSV 캐시만 사용합니다.", exc)
    panel_df.to_csv(csv_path, index=False)
    return panel_df
