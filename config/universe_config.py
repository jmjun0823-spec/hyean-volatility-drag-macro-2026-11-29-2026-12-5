from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class EventWindow:
    name: str
    date_label: str
    category: str
    expected_vol_multiplier: float
    spread_bps_shock: float
    flow_imbalance_shock: float


@dataclass(frozen=True)
class AnalysisConfig:
    """혜안 매크로·변동성 드래그 검증 파이프라인 중앙 설정."""

    start_date: str = "2006-01-01"
    end_date: str = "2026-12-05"
    window_start: str = "2026-11-29"
    window_end: str = "2026-12-05"
    rolling_years: int = 10
    screening_years: int = 3
    signal_lag_days: int = 1
    risk_free_rate: float = 0.035

    # 무형자산 자본화 및 감가상각 파라미터
    rd_capitalization_rate: float = 0.80
    sga_intangibles_share: float = 0.25
    intangibles_amort_rate: float = 0.20
    tangible_deprec_rate: float = 0.08
    alpha_u: float = 0.35
    tfp_unmeasured_weight: float = 0.50

    # 스크리닝 컷오프
    min_3y_conversion: float = 1.50
    min_3y_fcf_margin: float = 0.08
    min_3y_adj_roic: float = 0.12
    min_3y_gpm_change: float = 0.00
    max_3y_debt_growth: float = 0.25

    # 레버리지 배수 및 실행 리스크 파라미터
    leverage_multipliers: Tuple[int, ...] = (1, 2, 3)
    base_nav: float = 100.0
    base_spread_bps: float = 2.0
    slippage_vol_factor: float = 18.0
    slippage_imbalance_factor: float = 12.0
    eod_rebalance_bps: float = 6.5

    # 2026년 12월 1주차 주요 이벤트 일정
    event_windows: Tuple[EventWindow, ...] = field(
        default_factory=lambda: (
            EventWindow("month_end_rebalance", "2026-11-30", "flow_rebalance", 1.30, 5.0, 0.45),
            EventWindow("kr_trade_us_ism_mfg", "2026-12-01", "trade_manufacturing", 1.40, 6.0, 0.35),
            EventWindow("kr_cpi_us_adp", "2026-12-02", "inflation_labor", 1.45, 6.5, 0.38),
            EventWindow("us_ism_svc_jobless", "2026-12-03", "services_jobless", 1.35, 5.5, 0.30),
            EventWindow("us_nfp_unemployment", "2026-12-04", "nonfarm_payrolls", 1.95, 12.0, 0.55),
        )
    )

    # 벤치마크 및 샘플 유니버스
    universe: Tuple[str, ...] = field(
        default_factory=lambda: (
            "^GSPC", "^IXIC", "^KS11", "XLK", "XLI",
            "MSFT", "NVDA", "GOOGL", "META", "AAPL",
            "CAT", "DE", "HON", "XOM", "005930.KS", "000660.KS", "035420.KS", "051910.KS",
        )
    )
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def r_analytics_dir(self) -> Path:
        return self.project_root / "r_analytics"

    @property
    def output_dir(self) -> Path:
        return self.project_root / "outputs"


DEFAULT_CONFIG = AnalysisConfig()
