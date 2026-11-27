from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from config.universe_config import AnalysisConfig, DEFAULT_CONFIG
from src.data_loader import load_or_generate_panel
from src.market_data_fetcher import build_override_from_prices, fetch_yfinance_override_panel
from src.report_builder import build_blog_summary_markdown
from src.volatility_drag_engine import VolatilityDragResult, run_volatility_drag_verification


def run_verification_with_override(
    raw_override_df: pd.DataFrame,
    cfg: AnalysisConfig = DEFAULT_CONFIG,
    output_filename: str = "live_verification_summary.md",
) -> VolatilityDragResult:
    panel = load_or_generate_panel(cfg=cfg, raw_override_df=raw_override_df, use_cache=False)
    result = run_volatility_drag_verification(panel_df=panel, cfg=cfg)
    md = build_blog_summary_markdown(result=result, cfg=cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / output_filename).write_text(md, encoding="utf-8")
    return result


def main(
    ticker: str = "^GSPC",
    start: str = "2026-11-01",
    end: str = "2026-12-05",
    csv_path: Optional[str] = None,
    price_col: str = "Close",
) -> None:
    cfg = DEFAULT_CONFIG
    if csv_path:
        raw_df = pd.read_csv(Path(csv_path), index_col=0, parse_dates=True)
        col = price_col if price_col in raw_df.columns else raw_df.columns[0]
        override_df = build_override_from_prices(
            close_df=pd.DataFrame({ticker: raw_df[col]}),
            ticker=ticker,
            cfg=cfg,
        )
    else:
        override_df = fetch_yfinance_override_panel(ticker=ticker, start=start, end=end, cfg=cfg)

    res = run_verification_with_override(raw_override_df=override_df, cfg=cfg)
    print(f"[{ticker}] H0 기각: {res.h0_rejected_3x} | t={res.h0_t_stat_3x:.4f} | p={res.h0_p_value_3x:.4e}")
    print(res.summary_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="실측 시장 데이터 기반 변동성 드래그 검증")
    parser.add_argument("--ticker", type=str, default="^GSPC")
    parser.add_argument("--start", type=str, default="2026-11-01")
    parser.add_argument("--end", type=str, default="2026-12-05")
    parser.add_argument("--csv-path", type=str, default=None)
    parser.add_argument("--price-col", type=str, default="Close")
    args = parser.parse_args()
    main(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        csv_path=args.csv_path,
        price_col=args.price_col,
    )
