from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.universe_config import AnalysisConfig, DEFAULT_CONFIG
from src.volatility_drag_engine import VolatilityDragResult


def _build_python_ols_fallback_summary(panel_df: Optional[pd.DataFrame]) -> str:
    """R 미설치 환경에서도 회귀 교차 검증 요약을 생성합니다."""
    if panel_df is None or panel_df.empty:
        return "[검증 요약] 패널 데이터가 제공되지 않았습니다."

    grp = panel_df.groupby("path_id", as_index=False).agg(
        base_cum=("underlying_return", lambda s: float(np.prod(1.0 + s.to_numpy()) - 1.0)),
        lev3_cum=("underlying_return", lambda s: float(np.prod(1.0 + np.clip(3.0 * s.to_numpy(), -0.9999, None)) - 1.0)),
        realized_var=("underlying_return", lambda s: float(np.var(s.to_numpy(), ddof=1))),
        vol_mult=("vol_multiplier", "mean"),
    )
    grp["drag_3x"] = (1.0 + grp["lev3_cum"]) - (1.0 + 3.0 * grp["base_cum"])

    if len(grp) >= 4 and float(grp["realized_var"].std(ddof=1)) > 1e-15:
        slope_var, intercept, r_val, p_val, std_err = np.polyfit(
            grp["realized_var"], grp["drag_3x"], deg=1
        ), 0.0, 0.0, 0.0, 0.0
        # 단순 회귀 계수 산출
        x = grp["realized_var"].to_numpy()
        y = grp["drag_3x"].to_numpy()
        x_mean, y_mean = float(np.mean(x)), float(np.mean(y))
        denom = float(np.sum((x - x_mean) ** 2))
        b1 = float(np.sum((x - x_mean) * (y - y_mean)) / denom) if denom > 1e-18 else 0.0
        b0 = y_mean - b1 * x_mean
        return (
            "=== Python OLS Cross-Validation Fallback ===\n"
            f"Paths: {len(grp)} | Mean 3x Drag: {np.mean(y):.6f}\n"
            f"OLS(drag_3x ~ realized_var): intercept={b0:.6f}, slope_var={b1:.4f}"
        )
    return (
        "=== Python Deterministic Verification Summary ===\n"
        f"Paths: {len(grp)} | Mean 3x Drag: {grp['drag_3x'].mean():.6f}"
    )


def _read_text_safe(path: Path, panel_df: Optional[pd.DataFrame] = None) -> str:
    if path.exists():
        content = path.read_text(encoding="utf-8").strip()
        if content:
            return content
    return _build_python_ols_fallback_summary(panel_df)


def build_blog_summary_markdown(
    result: VolatilityDragResult,
    cfg: AnalysisConfig = DEFAULT_CONFIG,
    r_summary_path: Optional[Path] = None,
    panel_df: Optional[pd.DataFrame] = None,
) -> str:
    """변동성 드래그, 수급 레짐, 상관성 및 이벤트 리스크 검증 결과를 마크다운 리포트로 생성합니다."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    r_path = r_summary_path or (cfg.output_dir / "volatility_drag_r_summary.txt")
    r_text = _read_text_safe(r_path, panel_df=panel_df)

    verdict_str = (
        f"기각 (t={result.h0_t_stat_3x:.2f}, p={result.h0_p_value_3x:.2e}, Wilcoxon p={result.h0_wilcoxon_p_3x:.2e})"
        if result.h0_rejected_3x
        else f"미기각 (t={result.h0_t_stat_3x:.2f}, p={result.h0_p_value_3x:.2e}, Wilcoxon p={result.h0_wilcoxon_p_3x:.2e})"
    )

    sections = [
        f"# 주간 증시 및 변동성 드래그 정량 검증 리포트 ({cfg.window_start} ~ {cfg.window_end})",
        "",
        "## 1. 가설 검증 및 변동성 드래그 입증",
        f"- **3배 레버리지 배수 비례 귀무가설($H_0$)**: {verdict_str}",
        f"- **고변동성 레짐 드래그 심화 여부**: {result.high_vol_drag_worse_3x}",
        f"- **노이즈성 교차등락(`noisy_chop`) 드래그 심화 여부**: {result.chop_drag_worse_3x}",
        f"- **변동성 배수 vs 슬리피지 상관계수**: r={result.vol_slippage_corr:.4f} (p={result.vol_slippage_p_val:.2e})",
        f"- **수급 불균형 vs 슬리피지 상관계수**: r={result.imbalance_slippage_corr:.4f} (p={result.imbalance_slippage_p_val:.2e})",
        "",
        "### 1-1. +2% / -2% 교차 등락 수치 검증 (기준가 100)",
        result.exact_example_df.to_markdown(index=False),
        "",
        "### 1-2. 등락폭 및 기간 확대에 따른 3배 드래그 검증",
        result.amplitude_duration_df.to_markdown(index=False),
        "",
        "### 1-3. 레버리지 배수별 누적 수익률 및 변동성 드래그 요약",
        result.summary_df.to_markdown(index=False),
        "",
        "### 1-4. 레짐별 수급·변동성·드래그·슬리피지 비교",
        result.regime_comparison_df.to_markdown(index=False),
        "",
        "## 2. 이벤트 일정별 괴리율 및 슬리피지 리스크",
        result.event_risk_df.to_markdown(index=False),
        "",
        "## 3. 회귀 및 교차 검증 결과",
        "```text",
        r_text,
        "```",
    ]

    report_md = "\n".join(sections)
    (cfg.output_dir / "blog_verification_summary.md").write_text(report_md, encoding="utf-8")
    return report_md
