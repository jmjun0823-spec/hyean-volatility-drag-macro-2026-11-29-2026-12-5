suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
})

args <- commandArgs(trailingOnly = TRUE)
input_csv <- ifelse(length(args) >= 1, args[1], "data/event_volatility_panel.csv")
out_txt   <- ifelse(length(args) >= 2, args[2], "outputs/volatility_drag_r_summary.txt")

dir.create(dirname(out_txt), showWarnings = FALSE, recursive = TRUE)

if (!file.exists(input_csv)) {
  stop(paste("입력 CSV가 존재하지 않습니다:", input_csv))
}

df <- read_csv(input_csv, show_col_types = FALSE) %>%
  arrange(path_id, date) %>%
  group_by(path_id) %>%
  mutate(
    lev2_ret = pmax(2 * underlying_return, -0.9999),
    lev3_ret = pmax(3 * underlying_return, -0.9999),
    base_cum = cumprod(1 + underlying_return) - 1,
    lev2_cum = cumprod(1 + lev2_ret) - 1,
    lev3_cum = cumprod(1 + lev3_ret) - 1,
    drag_3x  = (1 + lev3_cum) - (1 + 3 * base_cum),
    realized_var = var(underlying_return)
  ) %>%
  ungroup()

final_df <- df %>%
  group_by(path_id) %>%
  slice_tail(n = 1) %>%
  ungroup()

h0_test <- t.test(final_df$lev3_cum - 3 * final_df$base_cum, mu = 0)
ols_fit <- lm(drag_3x ~ realized_var + vol_multiplier, data = final_df)

sink(out_txt)
cat("=== R Cross-Validation: Volatility Drag & Event Regime ===\n")
cat(sprintf("Paths: %d | Mean 3x H0 diff: %.6f | t-stat: %.4f | p-value: %.3e\n\n",
            nrow(final_df), h0_test$estimate, h0_test$statistic, h0_test$p.value))
print(summary(ols_fit))
sink()
