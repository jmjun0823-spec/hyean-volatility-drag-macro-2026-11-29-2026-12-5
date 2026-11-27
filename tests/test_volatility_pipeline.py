from __future__ import annotations

import pandas as pd
import pytest

from config.universe_config import AnalysisConfig
from src.data_loader import load_or_generate_panel
from src.market_data_fetcher import build_override_from_prices
from src.report_builder import build_blog_summary_markdown
from src.volatility_drag_engine import (
    compute_amplitude_and_duration_matrix,
    compute_exact_chop_example,
    run_volatility_drag_verification,
)


def test_exact_chop_numbers_1x_2x_3x() -> None:
    df = compute_exact_chop_example(base_nav=100.0)
    final_rows = df[df["step"] == "Day 2 (-2% Underlying)"].set_index("leverage")

    assert final_rows.loc["1x", "nav"] == pytest.approx(99.96, abs=1e-4)
    assert final_rows.loc["1x", "cum_return_pct"] == pytest.approx(-0.04, abs=1e-4)
    assert final_rows.loc["2x", "nav"] == pytest.approx(99.84, abs=1e-4)
    assert final_rows.loc["2x", "cum_return_pct"] == pytest.approx(-0.16, abs=1e-4)
    assert final_rows.loc["3x", "nav"] == pytest.approx(99.64, abs=1e-4)
    assert final_rows.loc["3x", "cum_return_pct"] == pytest.approx(-0.36, abs=1e-4)


def test_amplitude_and_duration_monotonically_worsen_drag() -> None:
    mat = compute_amplitude_and_duration_matrix(base_nav=100.0)
    # 동일 기간(10일)에서 등락폭 1% -> 2% -> 5%로 커질수록 NAV가 단조 감소
    d10 = mat[mat["holding_days"] == 10].sort_values("amp_pct")
    assert d10["nav_3x"].is_monotonic_decreasing

    # 동일 등락폭(2%)에서 보유일수 2일 -> 10일 -> 30일로 길어질수록 NAV가 단조 감소
    a2 = mat[mat["amp_pct"] == 2.0].sort_values("holding_days")
    assert a2["nav_3x"].is_monotonic_decreasing


def test_panel_hypothesis_event_risk_and_report(tmp_path) -> None:
    # 평시 대조군(2026-11-27, non_event_session)까지 포함해 생성 후 이벤트일(2026-12-04)과 비교
    cfg = AnalysisConfig(
        window_start="2026-11-27",
        window_end="2026-12-05",
        project_root=tmp_path,
    )
    panel = load_or_generate_panel(cfg=cfg, use_cache=False, seed=2026)
    res = run_volatility_drag_verification(panel_df=panel, cfg=cfg)

    assert res.h0_rejected_3x is True
    assert res.h0_p_value_3x < 0.01
    assert res.high_vol_drag_worse_3x is True

    # 12월 4일(us_nfp_unemployment) 슬리피지가 평시(2026-11-27 baseline)보다 높아야 함
    nfp_row = res.event_risk_df[res.event_risk_df["date"] == "2026-12-04"].iloc[0]
    base_row = res.event_risk_df[res.event_risk_df["date"] == "2026-11-27"].iloc[0]
    assert nfp_row["est_slippage_bps"] > base_row["est_slippage_bps"]

    md = build_blog_summary_markdown(result=res, cfg=cfg)
    assert (tmp_path / "outputs" / "blog_verification_summary.md").exists()
    assert "변동성 드래그" in md


def test_price_override_pipeline() -> None:
    close_df = pd.DataFrame(
        {"^KS11": [2500.0, 2550.0, 2499.0, 2548.98, 2497.9999]}
    )
    override_df = build_override_from_prices(close_df=close_df, ticker="^KS11")
    assert len(override_df) == 4
    assert {"path_id", "date", "underlying_return", "vol_multiplier", "spread_bps", "order_imbalance"}.issubset(
        override_df.columns
    )
