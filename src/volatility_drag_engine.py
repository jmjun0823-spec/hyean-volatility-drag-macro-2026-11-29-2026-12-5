from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from config.universe_config import AnalysisConfig, DEFAULT_CONFIG


@dataclass(frozen=True)
class VolatilityDragResult:
    summary_df: pd.DataFrame
    event_risk_df: pd.DataFrame
    exact_example_df: pd.DataFrame
    amplitude_duration_df: pd.DataFrame
    regime_comparison_df: pd.DataFrame
    h0_t_stat_3x: float
    h0_p_value_3x: float
    h0_wilcoxon_p_3x: float
    h0_rejected_3x: bool
    high_vol_drag_worse_3x: bool
    chop_drag_worse_3x: bool
    vol_slippage_corr: float
    vol_slippage_p_val: float
    imbalance_slippage_corr: float
    imbalance_slippage_p_val: float


def compute_exact_chop_example(base_nav: float = 100.0) -> pd.DataFrame:
    """
    기초지수 +2% 상승 후 -2% 하락 시 1x/2x/3x 누적 NAV와 수익률을 정확히 계산합니다.
    - 1x: 100 -> 102 -> 99.96 (-0.04%)
    - 2x: 100 -> 104 -> 99.84 (-0.16%)
    - 3x: 100 -> 106 -> 99.64 (-0.36%)
    """
    steps = [
        ("Day 0 (Start)", 0.00),
        ("Day 1 (+2% Underlying)", 0.02),
        ("Day 2 (-2% Underlying)", -0.02),
    ]
    rows: List[Dict[str, object]] = []
    for k in (1, 2, 3):
        nav = base_nav
        for label, r_base in steps:
            if label != "Day 0 (Start)":
                nav = nav * (1.0 + k * r_base)
            rows.append(
                {
                    "leverage": f"{k}x",
                    "step": label,
                    "nav": round(nav, 4),
                    "cum_return_pct": round((nav / base_nav - 1.0) * 100.0, 4),
                }
            )
    return pd.DataFrame(rows)


def compute_amplitude_and_duration_matrix(base_nav: float = 100.0) -> pd.DataFrame:
    """
    등락폭(1%, 2%, 5%)과 교차등락 기간(2일, 10일, 30일)에 따라
    3x 레버리지의 변동성 드래그가 단조 심화되는지 검증 테이블을 생성합니다.
    """
    rows: List[Dict[str, object]] = []
    for amp in (0.01, 0.02, 0.05):
        for n_pairs in (1, 5, 15):
            n_days = 2 * n_pairs
            r_up = 3.0 * amp
            r_dn = -3.0 * amp
            nav_3x = base_nav * (((1.0 + r_up) * (1.0 + r_dn)) ** n_pairs)
            cum_ret_3x = nav_3x / base_nav - 1.0
            base_pair_ret = (1.0 + amp) * (1.0 - amp) - 1.0
            base_cum_ret = ((1.0 + base_pair_ret) ** n_pairs) - 1.0
            drag_vs_h0 = (1.0 + cum_ret_3x) - (1.0 + 3.0 * base_cum_ret)
            rows.append(
                {
                    "amp_pct": round(amp * 100.0, 2),
                    "holding_days": n_days,
                    "nav_3x": round(nav_3x, 4),
                    "cum_return_3x_pct": round(cum_ret_3x * 100.0, 4),
                    "drag_vs_3x_base_pct": round(drag_vs_h0 * 100.0, 4),
                }
            )
    return pd.DataFrame(rows)


def run_volatility_drag_verification(
    panel_df: pd.DataFrame,
    cfg: AnalysisConfig = DEFAULT_CONFIG,
    alpha: float = 0.05,
) -> VolatilityDragResult:
    """
    패널 데이터를 기반으로:
    1) 1x/2x/3x 누적수익률 및 H0(배수 비례) t-검정·윌콕슨 검정
    2) 고변동성·교차등락 레짐별 드래그 및 수급/괴리율/슬리피지 검증
    3) 변동성·수급 불균형과 슬리피지 간 상관성 검증
    을 수행합니다.
    """
    df = panel_df.copy().sort_values(["path_id", "date"]).reset_index(drop=True)

    base_cum = (
        df.groupby("path_id")["underlying_return"]
        .apply(lambda s: float(np.prod(1.0 + s.to_numpy()) - 1.0))
        .rename("base_cum_return")
    )
    path_var = df.groupby("path_id")["underlying_return"].var(ddof=1).fillna(0.0)
    path_vol_std = df.groupby("path_id")["underlying_return"].std(ddof=1).fillna(0.0)
    path_regime = df.groupby("path_id")["regime"].first()

    summary_rows: List[Dict[str, object]] = []
    drag_3x_series = None
    t_stat_3x = 0.0
    p_val_3x = 1.0
    wilcox_p_3x = 1.0

    for k in cfg.leverage_multipliers:
        lev_ret_col = f"lev_{k}x_return"
        lev_nav_col = f"lev_{k}x_nav"

        df[lev_ret_col] = np.clip(k * df["underlying_return"].to_numpy(), -0.9999, None)
        df[lev_nav_col] = cfg.base_nav * df.groupby("path_id")[lev_ret_col].cumprod()

        path_lev_cum = df.groupby("path_id")[lev_ret_col].apply(
            lambda s: float(np.prod(1.0 + s.to_numpy()) - 1.0)
        )
        diff_vs_h0 = (path_lev_cum - k * base_cum).to_numpy()
        actual_drag = (1.0 + path_lev_cum.to_numpy()) - (1.0 + k * base_cum.to_numpy())
        approx_drag = -0.5 * k * (k - 1) * path_var.to_numpy()

        if len(diff_vs_h0) >= 2 and float(np.std(diff_vs_h0, ddof=1)) > 1e-15:
            t_stat, p_val = stats.ttest_1samp(diff_vs_h0, popmean=0.0, alternative="two-sided")
            try:
                _, w_pval = stats.wilcoxon(diff_vs_h0)
                w_pval_f = float(w_pval)
            except ValueError:
                w_pval_f = 1.0
        else:
            t_stat = float(np.mean(diff_vs_h0) / 1e-6) if abs(float(np.mean(diff_vs_h0))) > 0 else 0.0
            p_val = 0.0 if abs(float(np.mean(diff_vs_h0))) > 0 else 1.0
            w_pval_f = p_val

        summary_rows.append(
            {
                "leverage": f"{k}x",
                "mean_cum_return_pct": float(np.mean(path_lev_cum) * 100.0),
                "h0_expected_return_pct": float(np.mean(k * base_cum) * 100.0),
                "mean_actual_drag_pct": float(np.mean(actual_drag) * 100.0),
                "mean_approx_drag_pct": float(np.mean(approx_drag) * 100.0),
                "t_stat_vs_h0": float(t_stat),
                "p_value_vs_h0": float(p_val),
                "wilcoxon_p_vs_h0": w_pval_f,
                "reject_h0": bool(p_val < alpha),
            }
        )

        if k == 3:
            drag_3x_series = pd.Series(actual_drag, index=base_cum.index)
            t_stat_3x = float(t_stat)
            p_val_3x = float(p_val)
            wilcox_p_3x = w_pval_f

    summary_df = pd.DataFrame(summary_rows)

    # 고변동성 경로 vs 저변동성 경로의 3x 드래그 검증
    high_vol_mask = path_vol_std >= path_vol_std.median()
    if drag_3x_series is not None and high_vol_mask.any() and (~high_vol_mask).any():
        high_vol_worse = bool(
            float(drag_3x_series[high_vol_mask].mean()) < float(drag_3x_series[~high_vol_mask].mean())
        )
    else:
        high_vol_worse = bool(
            float(summary_df.loc[summary_df["leverage"] == "3x", "mean_actual_drag_pct"].iloc[0]) < 0.0
        )

    # 일별·이벤트별 괴리율 및 슬리피지(장중/장초반/장마감 리밸런싱 포함)
    df["nav_divergence_3x_pct"] = (
        np.abs(df["lev_3x_nav"] / cfg.base_nav - (1.0 + 3.0 * df["underlying_return"])) * 100.0
    )
    vol_excess = np.maximum(df["vol_multiplier"] - 1.0, 0.0)
    df["est_slippage_bps"] = (
        df["spread_bps"]
        + cfg.slippage_vol_factor * vol_excess
        + cfg.slippage_imbalance_factor * df["order_imbalance"]
        + cfg.eod_rebalance_bps * vol_excess
    )
    df["open_close_window_slippage_bps"] = df["est_slippage_bps"] * (1.0 + 0.35 * df["order_imbalance"])

    # 레짐별 비교 테이블 (수급/변동성/드래그/슬리피지)
    path_slip = df.groupby("path_id")["est_slippage_bps"].mean()
    path_imb = df.groupby("path_id")["order_imbalance"].mean()
    regime_df = pd.DataFrame(
        {
            "regime": path_regime,
            "base_cum_pct": base_cum * 100.0,
            "drag_3x_pct": drag_3x_series * 100.0,
            "avg_imbalance": path_imb,
            "avg_slippage_bps": path_slip,
        }
    )
    regime_comparison_df = (
        regime_df.groupby("regime", as_index=False)
        .agg(
            path_count=("base_cum_pct", "count"),
            mean_base_cum_pct=("base_cum_pct", "mean"),
            mean_drag_3x_pct=("drag_3x_pct", "mean"),
            mean_imbalance=("avg_imbalance", "mean"),
            mean_slippage_bps=("avg_slippage_bps", "mean"),
        )
    )

    chop_rows = regime_comparison_df[regime_comparison_df["regime"] == "noisy_chop"]
    event_rows = regime_comparison_df[regime_comparison_df["regime"] == "event_volatility"]
    if not chop_rows.empty and not event_rows.empty:
        chop_drag_worse = bool(
            float(chop_rows["mean_drag_3x_pct"].iloc[0]) < float(event_rows["mean_drag_3x_pct"].iloc[0])
        )
    else:
        chop_drag_worse = high_vol_worse

    # 상관관계 검증
    if len(df) >= 3 and float(df["vol_multiplier"].std(ddof=1)) > 1e-12:
        r_vol, p_vol = stats.pearsonr(df["vol_multiplier"], df["est_slippage_bps"])
    else:
        r_vol, p_vol = 0.0, 1.0

    if len(df) >= 3 and float(df["order_imbalance"].std(ddof=1)) > 1e-12:
        r_imb, p_imb = stats.pearsonr(df["order_imbalance"], df["est_slippage_bps"])
    else:
        r_imb, p_imb = 0.0, 1.0

    event_risk_df = (
        df.groupby(["date", "event_name", "event_category"], as_index=False)
        .agg(
            avg_vol_multiplier=("vol_multiplier", "mean"),
            avg_spread_bps=("spread_bps", "mean"),
            avg_imbalance=("order_imbalance", "mean"),
            avg_divergence_3x_pct=("nav_divergence_3x_pct", "mean"),
            est_slippage_bps=("est_slippage_bps", "mean"),
            open_close_slippage_bps=("open_close_window_slippage_bps", "mean"),
            p95_slippage_bps=("est_slippage_bps", lambda s: float(np.percentile(s, 95))),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    return VolatilityDragResult(
        summary_df=summary_df,
        event_risk_df=event_risk_df,
        exact_example_df=compute_exact_chop_example(base_nav=cfg.base_nav),
        amplitude_duration_df=compute_amplitude_and_duration_matrix(base_nav=cfg.base_nav),
        regime_comparison_df=regime_comparison_df,
        h0_t_stat_3x=t_stat_3x,
        h0_p_value_3x=p_val_3x,
        h0_wilcoxon_p_3x=wilcox_p_3x,
        h0_rejected_3x=bool(p_val_3x < alpha),
        high_vol_drag_worse_3x=high_vol_worse,
        chop_drag_worse_3x=chop_drag_worse,
        vol_slippage_corr=float(r_vol),
        vol_slippage_p_val=float(p_vol),
        imbalance_slippage_corr=float(r_imb),
        imbalance_slippage_p_val=float(p_imb),
    )
