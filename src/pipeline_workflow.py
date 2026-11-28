from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from config.universe_config import AnalysisConfig, DEFAULT_CONFIG
from src.chart_builder import generate_all_charts
from src.claim_verifiers import build_claim_verification_matrix
from src.data_loader import load_or_generate_panel
from src.historical_event_verifier import (
    fetch_historical_etf_and_index_panel,
    verify_empirical_market_claims,
)
from src.report_builder import build_blog_summary_markdown
from src.volatility_drag_engine import VolatilityDragResult, run_volatility_drag_verification

logger = logging.getLogger(__name__)


def _run_r_cross_validation(csv_path: Path, out_txt: Path, r_script: Path) -> Optional[Path]:
    rscript_bin = shutil.which("Rscript")
    if not rscript_bin or not r_script.exists():
        return None
    completed = subprocess.run(
        [rscript_bin, str(r_script), str(csv_path), str(out_txt)],
        capture_output=True,
        text=True,
        check=False,
    )
    return out_txt if completed.returncode == 0 else None


def run_full_workflow(
    cfg: AnalysisConfig = DEFAULT_CONFIG,
    use_cache: bool = True,
    seed: int = 2026,
    enable_live_empirical: bool = True,
    empirical_start: str = "2010-01-01",
    empirical_end: str = "2026-11-27",
) -> VolatilityDragResult:
    """
    1) 과거 실측 시장 데이터(^GSPC, SSO, UPRO, QQQ, XLP) 수집 및 실측 가설 검증
    2) 2026년 12월 1주차(11/29~12/05) 이벤트 주간 스트레스·슬리피지 패널 검증
    3) R 회귀 교차 검증 실행
    4) 전체 주장별 검증 판정표(VERIFIED) 포함 마크다운 리포트 및 차트 생성
    """
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    empirical_stats: Optional[Dict[str, object]] = None
    if enable_live_empirical:
        try:
            mkt_df = fetch_historical_etf_and_index_panel(start=empirical_start, end=empirical_end)
            mkt_df.to_csv(cfg.data_dir / "historical_etf_empirical_panel.csv", index=False)
            empirical_stats = verify_empirical_market_claims(mkt_df)
            logger.info("실측 시장 데이터(%d거래일) 검증 완료", empirical_stats["n_trading_days"])
        except Exception as exc:
            logger.warning("실측 시장 데이터 조회 건너뜀(%s) — 시뮬레이션 패널 단독 검증 진행", exc)

    panel_df = load_or_generate_panel(cfg=cfg, use_cache=use_cache, seed=seed)
    panel_csv_path = cfg.data_dir / "event_volatility_panel.csv"
    panel_df.to_csv(panel_csv_path, index=False)

    r_out_path = cfg.output_dir / "volatility_drag_r_summary.txt"
    r_script_path = cfg.r_analytics_dir / "volatility_drag_models.R"
    r_summary_file = _run_r_cross_validation(panel_csv_path, r_out_path, r_script_path)

    result = run_volatility_drag_verification(panel_df=panel_df, cfg=cfg)
    claim_matrix_df = build_claim_verification_matrix(
        sim_result=result,
        empirical_stats=empirical_stats,
    )

    md = build_blog_summary_markdown(
        result=result,
        cfg=cfg,
        r_summary_path=r_summary_file,
        panel_df=panel_df,
    )
    # 리포트 최상단에 전체 주장 검증 매트릭스 삽입
    matrix_section = (
        "\n## 0. 블로그 본문 전체 주장 검증 판정표 (실측 + 모델 통합)\n"
        + claim_matrix_df.to_markdown(index=False)
        + "\n"
    )
    full_md = md.replace("\n## 1. 가설 검증 및 변동성 드래그 입증\n", f"{matrix_section}\n## 1. 가설 검증 및 변동성 드래그 입증\n", 1)
    (cfg.output_dir / "blog_verification_summary.md").write_text(full_md, encoding="utf-8")

    generate_all_charts(result=result, cfg=cfg)
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-empirical", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    res = run_full_workflow(
        cfg=DEFAULT_CONFIG,
        use_cache=not args.no_cache,
        seed=args.seed,
        enable_live_empirical=not args.no_empirical,
    )
    print(f"검증 완료 | 3x H0 기각: {res.h0_rejected_3x} (t={res.h0_t_stat_3x:.2f}, p={res.h0_p_value_3x:.2e})")


if __name__ == "__main__":
    main()
