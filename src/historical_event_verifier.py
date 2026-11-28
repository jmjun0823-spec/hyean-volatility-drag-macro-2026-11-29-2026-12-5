def verify_empirical_market_claims(
    market_df: pd.DataFrame,
    window_days: int = 5,
    chop_block_days: int = 20,
) -> Dict[str, object]:
    df = market_df.copy().reset_index(drop=True)

    # 방어주 컬럼 호환 처리 (defensive_schd_ret 우선, 없으면 defensive_xlp_ret)
    def_col = "defensive_schd_ret" if "defensive_schd_ret" in df.columns else "defensive_xlp_ret"

    # 1) 금리 급등일(상위 10%) 기준 방어주 vs 성장주 초과수익률 검정
    if "tnx_yield_change_bps" in df.columns and df["tnx_yield_change_bps"].notna().sum() >= 30:
        rate_s = df["tnx_yield_change_bps"].dropna()
        rate_q90 = float(rate_s.quantile(0.90))
        stress = df[df["tnx_yield_change_bps"] >= rate_q90].dropna(subset=[def_col, "growth_qqq_ret"])
    else:
        vol_q90_tmp = float(df["vol_regime_mult"].quantile(0.90))
        stress = df[df["vol_regime_mult"] >= vol_q90_tmp].dropna(subset=[def_col, "growth_qqq_ret"])

    if len(stress) >= 5:
        def_spread = (stress[def_col] - stress["growth_qqq_ret"]) * 10000.0
        t_rate, p_rate = stats.ttest_1samp(def_spread, popmean=0.0, alternative="greater")
        rate_outperf_bps = float(def_spread.mean())
    else:
        t_rate, p_rate, rate_outperf_bps = float("nan"), float("nan"), float("nan")

    # 2a) 5일 횡보/교차등락 구간(|sp500_5d| < 2.0%) 조건부 H0 검정
    n_5d = len(df) // window_days
    tr_5d = df.iloc[: n_5d * window_days].copy()
    tr_5d["block_id"] = np.repeat(np.arange(n_5d), window_days)
    blk_5d = tr_5d.groupby("block_id", as_index=False).agg(
        sp500_5d=("sp500_ret", lambda s: float(np.prod(1.0 + s.dropna().to_numpy()) - 1.0)),
        upro_5d=("upro_3x_ret", lambda s: float(np.prod(1.0 + s.dropna().to_numpy()) - 1.0)),
    )
    blk_5d["drag_3x_5d_pct"] = (blk_5d["upro_5d"] - 3.0 * blk_5d["sp500_5d"]) * 100.0
    chop_5d = blk_5d[blk_5d["sp500_5d"].abs() < 0.02]["drag_3x_5d_pct"].dropna()
    if len(chop_5d) >= 5:
        t_chop5, p_chop5 = stats.ttest_1samp(chop_5d, popmean=0.0, alternative="less")
        chop5_mean = float(chop_5d.mean())
    else:
        t_chop5, p_chop5, chop5_mean = float("nan"), float("nan"), float("nan")

    # 2b) 20일 횡보 구간(|sp500_20d| < 3.0%) 조건부 H0 검정
    n_20d = len(df) // chop_block_days
    tr_20d = df.iloc[: n_20d * chop_block_days].copy()
    tr_20d["b20"] = np.repeat(np.arange(n_20d), chop_block_days)
    blk_20d = tr_20d.groupby("b20", as_index=False).agg(
        sp500_20d=("sp500_ret", lambda s: float(np.prod(1.0 + s.dropna().to_numpy()) - 1.0)),
        upro_20d=("upro_3x_ret", lambda s: float(np.prod(1.0 + s.dropna().to_numpy()) - 1.0)),
    )
    blk_20d["drag_3x_20d_pct"] = (blk_20d["upro_20d"] - 3.0 * blk_20d["sp500_20d"]) * 100.0
    sideways_20d = blk_20d[blk_20d["sp500_20d"].abs() < 0.03]["drag_3x_20d_pct"].dropna()
    if len(sideways_20d) >= 5:
        _, p_chop20 = stats.ttest_1samp(sideways_20d, popmean=0.0, alternative="less")
        sideways20_mean = float(sideways_20d.mean())
    else:
        p_chop20, sideways20_mean = float("nan"), float("nan")

    # 3) 고변동성 상위 10% vs 평시 하위 50% 3x 추적 괴리율 검정
    vol_q90 = float(df["vol_regime_mult"].quantile(0.90))
    vol_q50 = float(df["vol_regime_mult"].quantile(0.50))
    high_div = df[df["vol_regime_mult"] >= vol_q90]["tracking_diff_3x_bps"].dropna()
    calm_div = df[df["vol_regime_mult"] <= vol_q50]["tracking_diff_3x_bps"].dropna()
    if len(high_div) >= 5 and len(calm_div) >= 5:
        _, p_div = stats.ttest_ind(high_div, calm_div, equal_var=False, alternative="greater")
        div_high, div_calm = float(high_div.mean()), float(calm_div.mean())
    else:
        p_div, div_high, div_calm = float("nan"), float("nan"), float("nan")

    return {
        "n_trading_days": int(len(df)),
        "rate_shock_defensive_outperf_bps": rate_outperf_bps,
        "rate_shock_t_stat": float(t_rate),
        "rate_shock_p_value": float(p_rate),
        "chop_5d_count": int(len(chop_5d)),
        "chop_5d_mean_drag_3x_pct": chop5_mean,
        "chop_5d_t_stat": float(t_chop5),
        "chop_5d_p_value": float(p_chop5),
        "sideways_20d_count": int(len(sideways_20d)),
        "sideways_20d_mean_drag_3x_pct": sideways20_mean,
        "sideways_20d_p_value": float(p_chop20),
        "high_vol_tracking_div_bps": div_high,
        "calm_tracking_div_bps": div_calm,
        "tracking_div_surge_p_val": float(p_div),
    }
