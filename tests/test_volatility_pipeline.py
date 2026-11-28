from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.universe_config import AnalysisConfig
from src.chart_builder import generate_all_charts
from src.claim_verifiers import build_claim_verification_matrix
from src.data_loader import load_or_generate_panel
from src.historical_event_verifier import verify_empirical_market_claims
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
    d10 = mat[mat["holding_days"] == 10].sort_values("amp_pct")
    assert d10["nav_3x"].is_monotonic_decreasing

    a2 = mat[mat["amp_pct"] == 2.0].sort_values("holding_days")
    assert a2["nav_3x"].is_monotonic_decreasing


def test_path_vol_multiplier_has_variance_and_full_claims(tmp_path) -> None:
    cfg = AnalysisConfig(
        window_start="2026-11-27",
        window_end="2026-12-05",
        project_root=tmp_path,
    )
    panel = load_or_generate_panel(cfg=cfg, use_cache=False, seed=2026)

    # 경로별 max(vol_multiplier)에 분산이 존재해 R 회귀에서 NA가 발생하지 않음을 보장
    max_vols = panel.groupby("path_id")["vol_multiplier"].max()
    assert max_vols.nunique() > 10

    res = run_volatility_drag_verification(panel_df=panel, cfg=cfg)
    assert res.h0_rejected_3x is True
    assert res.h0_p_value_3x < 0.01
    assert res.high_vol_drag_worse_3x is True
    assert res.chop_drag_worse_3x is True

    # 합성 실측 패널로 조건부 검증(금리 상승일 방어주 우위 + 횡보 5일 드래그) 확인
    rng = np.random.default_rng(7)
    n_days = 500
    sp = np.where(np.arange(n_days) % 2 == 0, 0.012, -0.0115)
    upro = np.clip(3.0 * sp - 0.0003, -0.99, None)
    tnx_chg = rng.normal(0.0, 5.0, size=n_days)
    qqq = sp - 0.0004 * np.maximum(tnx_chg, 0.0)
    schd = sp - 0.0001 * np.maximum(tnx_chg, 0.0)
    vol_mult = np.where(np.arange(n_days) % 10 == 0, 2.2, 0.8)

    mock_emp = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n_days, freq="B").strftime("%Y-%m-%d"),
            "sp500_ret": sp,
            "sso_2x_ret": 2.0 * sp,
            "upro_3x_ret": upro,
            "growth_qqq_ret": qqq,
            "defensive_schd_ret": schd,
            "tnx_yield_change_bps": tnx_chg,
            "tracking_diff_3x_bps": np.where(vol_mult > 1.5, 22.0, 8.0),
            "opening_gap_3x_bps": np.where(vol_mult > 1.5, 180.0, 60.0),
            "vol_regime_mult": vol_mult,
        }
    )
    emp_stats = verify_empirical_market_claims(mock_emp)
    assert emp_stats["rate_shock_defensive_outperf_bps"] > 0
    assert emp_stats["chop_5d_mean_drag_3x_pct"] < 0
    assert emp_stats["chop_5d_p_value"] < 0.05

    matrix_df = build_claim_verification_matrix(sim_result=res, empirical_stats=emp_stats)
    assert (matrix_df["status"].isin(["VERIFIED", "MODEL_VERIFIED"])).all()

    md = build_blog_summary_markdown(result=res, cfg=cfg, panel_df=panel)
    assert (tmp_path / "outputs" / "blog_verification_summary.md").exists()
    assert "변동성 드래그" in md

    p1, p2 = generate_all_charts(result=res, cfg=cfg)
    assert p1.exists() and p2.exists()
