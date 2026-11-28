from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd

from config.universe_config import AnalysisConfig, DEFAULT_CONFIG
from src.chart_builder import generate_all_charts
from src.data_loader import load_or_generate_panel
from src.market_data_fetcher import build_override_from_prices, fetch_yfinance_override_panel
from src.report_builder import build_blog_summary_markdown
from src.volatility_drag_engine import VolatilityDragResult, run_volatility_drag_verification

logger = logging.getLogger(__name__)


def _run_r_cross_validation(csv_path: Path, out_txt: Path, r_script: Path) -> Optional[Path]:
    """Rscript가 설치되어 있고 스크립트가 존재할 때만 R 교차 검증을 수행합니다."""
    rscript_bin = shutil.which("Rscript")
    if not rscript_bin or not r_script.exists():
        logger.info("Rscript 또는 R 스크립트가 없어 Python 단독 검증 모드로 진행합니다.")
        return None

    completed = subprocess.run(
        [rscript_bin, str(r_script), str(csv_path), str(out_txt)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        logger.warning("R 교차 검증 실패(%s): %s", completed.returncode, completed.stderr.strip())
        return None
    return out_txt


def run_full_workflow(
    cfg: AnalysisConfig = DEFAULT_CONFIG,
    use_cache: bool = True,
    seed: int = 2026,
    ticker: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    csv_path: Optional[str] = None,
    price_col: str = "Close",
) -> VolatilityDragResult:
    """
    전체 검증 워크플로우:
    1) 패널 데이터 준비(실측 CSV / yfinance / 캐시 또는 폴백 생성)
    2) 변동성 드래그·귀무가설·이벤트 괴리율/슬리피지 검증
    3) R 회귀·가설 교차 검증(.txt 산출)
    4) 마크다운 리포트(.md) 및 시각화 차트(.png) 생성
    """
    override_df: Optional[pd.DataFrame] = None
    if csv_path:
        raw_df = pd.read_csv(Path(csv_path), index_col=0, parse_dates=True)
        col = price_col if price_col in raw_df.columns else raw_df.columns[0]
        target_ticker = ticker or col
        override_df = build_override_from_prices(
            close_df=pd.DataFrame({target_ticker: raw_df[col]}),
            ticker=target_ticker,
            cfg=cfg,
        )
    elif ticker:
        override_df = fetch_yfinance_override_panel(
            ticker=ticker,
            start=start or cfg.window_start,
            end=end or cfg.window_end,
            cfg=cfg,
        )

    panel_df = load_or_generate_panel(
        cfg=cfg,
        raw_override_df=override_df,
        use_cache=use_cache,
        seed=seed,
    )

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    panel_csv_path = cfg.data_dir / "event_volatility_panel.csv"
    panel_df.to_csv(panel_csv_path, index=False)

    r_out_path = cfg.output_dir / "volatility_drag_r_summary.txt"
    r_script_path = cfg.r_analytics_dir / "volatility_drag_models.R"
    r_summary_file = _run_r_cross_validation(
        csv_path=panel_csv_path,
        out_txt=r_out_path,
        r_script=r_script_path,
    )

    result = run_volatility_drag_verification(panel_df=panel_df, cfg=cfg)
    build_blog_summary_markdown(
        result=result,
        cfg=cfg,
        r_summary_path=r_summary_file,
    )
    generate_all_charts(result=result, cfg=cfg)
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="변동성 드래그 및 매크로 이벤트 전체 워크플로우 실행")
    parser.add_argument("--no-cache", action="store_true", help="캐시를 무시하고 새로 생성")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--ticker", type=str, default=None, help="실측 조회 티커 (예: ^GSPC, ^KS11)")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--csv-path", type=str, default=None)
    parser.add_argument("--price-col", type=str, default="Close")
    args = parser.parse_args()

    res = run_full_workflow(
        cfg=DEFAULT_CONFIG,
        use_cache=not args.no_cache,
        seed=args.seed,
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        csv_path=args.csv_path,
        price_col=args.price_col,
    )
    print(f"워크플로우 완료 | 3x H0 기각: {res.h0_rejected_3x} (t={res.h0_t_stat_3x:.2f}, p={res.h0_p_value_3x:.2e})")
    print(f"생성된 리포트: {DEFAULT_CONFIG.output_dir / 'blog_verification_summary.md'}")


if __name__ == "__main__":
    main()
