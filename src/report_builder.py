from __future__ import annotations

from pathlib import Path
from typing import Optional

from config.universe_config import AnalysisConfig, DEFAULT_CONFIG
from src.volatility_drag_engine import VolatilityDragResult


def _read_text_safe(path: Path) -> str:
    if not path.exists():
        return f"[R 산출물 미생성: {path.name}] R 스크립트 실행 후 다시 병합됩니다."
    content = path.read_text(encoding="utf-8").strip()
    return content if content else f"[R 산출물 비어 있음: {path.name}]"


def build_blog_summary_markdown(
    result: VolatilityDragResult,
    cfg: AnalysisConfig = DEFAULT_CONFIG,
    r_summary_path: Optional[Path] = None,
) -> str:
    """변동성 드래그 및 이벤트 리스크 검증 결과를 마크다운 리포트로 생성합니다."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    r_path = r_summary_path or (cfg.output_dir / "volatility_drag_r_summary.txt")
    r_text = _read_text_safe(r_path)

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
        "## 2. 이벤트 일정별 괴리율 및 슬리피지 리스크",
        result.event_risk_df.to_markdown(index=False),
        "",
        "## 3. R 교차 검증 결과",
        "```text",
        r_text,
        "```",
    ]

    report_md = "\n".join(sections)
    (cfg.output_dir / "blog_verification_summary.md").write_text(report_md, encoding="utf-8")
    return report_md
