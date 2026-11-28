from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from config.universe_config import AnalysisConfig, DEFAULT_CONFIG
from src.volatility_drag_engine import VolatilityDragResult


def _apply_quant_blog_style() -> None:
    """seaborn-whitegrid 기반 와이드(12x5) 퀀트 리포트 스타일을 적용합니다."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (12, 5),
            "figure.dpi": 150,
            "axes.edgecolor": "#CCCCCC",
            "axes.linewidth": 0.9,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": False,
            "legend.fontsize": 10,
        }
    )


def plot_volatility_drag_decay_chart(
    result: VolatilityDragResult,
    cfg: AnalysisConfig = DEFAULT_CONFIG,
    filename: str = "chart_volatility_drag_profile.png",
) -> Path:
    """등락폭(1%, 2%, 5%) 및 보유 기간(2D, 10D, 30D)에 따른 3x 변동성 드래그(%p)를 시각화합니다."""
    _apply_quant_blog_style()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.output_dir / filename

    df = result.amplitude_duration_df.copy()
    df["scenario_label"] = df.apply(
        lambda r: f"±{r['amp_pct']:.0f}% / {int(r['holding_days'])}D", axis=1
    )

    x = np.arange(len(df))
    y = df["drag_vs_3x_base_pct"].to_numpy(dtype=float)
    mean_drag = float(np.mean(y))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axhline(0.0, color="#666666", linewidth=1.0, zorder=1)
    ax.plot(
        x,
        y,
        color="#355C7D",
        linewidth=1.8,
        marker="o",
        markersize=5,
        label="3x Leverage Volatility Drag (%p)",
        zorder=3,
    )
    ax.axhline(
        mean_drag,
        color="#C00000",
        linestyle="--",
        linewidth=1.8,
        label=f"Mean Drag: {mean_drag:.2f}%p",
        zorder=2,
    )

    ax.set_title("3x Leverage ETF Volatility Drag by Chop Amplitude & Holding Days (%p)", pad=12)
    ax.set_xlabel("Chop Scenario (Daily Amplitude / Holding Days)")
    ax.set_ylabel("Volatility Drag vs 3x Base (%p)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["scenario_label"], rotation=0)
    ax.set_ylim(min(y) * 1.15, 2.0)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_event_slippage_bar_chart(
    result: VolatilityDragResult,
    cfg: AnalysisConfig = DEFAULT_CONFIG,
    filename: str = "chart_event_slippage_divergence.png",
) -> Path:
    """이벤트 일정별 예상 슬리피지(bps) 초과분을 양/음(경고/안정) 막대로 시각화합니다."""
    _apply_quant_blog_style()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.output_dir / filename

    df = result.event_risk_df.copy()
    short_labels = {
        "month_end_rebalance": "11/30\nMonth-End",
        "kr_trade_us_ism_mfg": "12/01\nKR Trade / ISM",
        "kr_cpi_us_adp": "12/02\nKR CPI / ADP",
        "us_ism_svc_jobless": "12/03\nISM Svc / Claims",
        "us_nfp_unemployment": "12/04\nUS NFP (Key)",
    }
    df["x_label"] = df["event_name"].map(short_labels).fillna(df["date"])

    threshold_bps = 20.0
    excess_bps = df["est_slippage_bps"].to_numpy(dtype=float) - threshold_bps
    colors = ["#C44E52" if val > 0 else "#4C8C54" for val in excess_bps]

    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axhline(0.0, color="#111111", linewidth=1.0, zorder=2)

    bars = ax.bar(x, excess_bps, width=0.52, color=colors, zorder=3)
    for bar, raw_bps in zip(bars, df["est_slippage_bps"]):
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        offset = 0.6 if h >= 0 else -0.6
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + offset,
            f"{raw_bps:.1f} bps",
            ha="center",
            va=va,
            fontsize=9.5,
            fontweight="bold",
            color="#222222",
        )

    ax.set_title("Dec 2026 Week 1 Event Schedule: Slippage Risk vs 20 bps Baseline (bps)", pad=12)
    ax.set_xlabel("Macro & Flow Event Date (2026-11-30 ~ 2026-12-04)")
    ax.set_ylabel("Excess Slippage vs Baseline (bps)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["x_label"])
    ax.set_ylim(min(excess_bps) - 5.0, max(excess_bps) + 6.0)

    legend_handles = [
        Patch(facecolor="#C44E52", label="High Execution Risk (> 20 bps)"),
        Patch(facecolor="#4C8C54", label="Contained Risk (<= 20 bps)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def generate_all_charts(
    result: VolatilityDragResult,
    cfg: AnalysisConfig = DEFAULT_CONFIG,
) -> Tuple[Path, Path]:
    p1 = plot_volatility_drag_decay_chart(result=result, cfg=cfg)
    p2 = plot_event_slippage_bar_chart(result=result, cfg=cfg)
    return p1, p2
