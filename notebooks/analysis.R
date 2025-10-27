# Causal Inference Analysis: R Implementation
# Complete parallel implementation of analysis.ipynb in R
# ============================================================================

# Load required libraries
library(tidyverse)
library(caret)
library(MASS)
library(ordinal)
library(broom)
library(igraph)
library(ggplot2)
library(gridExtra)
library(lmtest)
library(car)
library(cowplot)

# ============================================================================
# SECTION 1: DATA LOADING AND PREPARATION
# ============================================================================

# Load the raw dataset
df_raw <- read.csv('data/raw/GlobalProtestTracker.csv')
df_processed <- read.csv('data/processed/GlobalProtestTracker_with_outcomes.csv')

cat("Raw data shape:", nrow(df_raw), "x", ncol(df_raw), "\n")
cat("Processed data shape:", nrow(df_processed), "x", ncol(df_processed), "\n")

# ============================================================================
# SECTION 2: OUTCOME VARIABLE MAPPING
# ============================================================================

# Map outcome labels to numeric ordinal values
outcome_mapping <- c(
  'No significant change' = 0,
  'partial political change' = 1,
  'Policy changed to meet demands (fully changed/reversed)' = 2,
  'regime shift' = 3
)

df_processed$outcome_numeric <- as.integer(as.character(
  sapply(df_processed$outcome_label, function(x) outcome_mapping[x])
))

cat("Outcome distribution:\n")
print(table(df_processed$outcome_numeric))

# ============================================================================
# SECTION 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

# Basic summary statistics
summary_stats <- df_processed %>%
  select(Duration_days, Peak_Size) %>%
  summary()

cat("\nDuration summary:\n")
print(summary_stats[, 'Duration_days'])

cat("\nPeak Size summary:\n")
print(summary_stats[, 'Peak_Size'])

# Categorical variable distributions
cat("\nTriggers distribution:\n")
print(table(df_processed$Triggers_category))

cat("\nMotivations distribution:\n")
print(table(df_processed$Motivations_category))

cat("\nKey Participants distribution:\n")
print(table(df_processed$Key_Participants_category))

# ============================================================================
# SECTION 4: CAUSAL FRAMEWORK - DAG VISUALIZATION
# ============================================================================

# Create a Directed Acyclic Graph (DAG) showing causal structure
# Nodes: Duration, Outcome, and potential confounders
# Edges: Show causal relationships

# Create igraph object
dag_edges <- data.frame(
  from = c('Regime_Type', 'Regime_Type', 'Social_Context', 'Social_Context',
           'Duration', 'Regime_Type', 'Social_Context'),
  to = c('Duration', 'Outcome', 'Duration', 'Outcome', 'Outcome', 'Outcome', 'Peak_Size')
)

dag <- graph_from_data_frame(dag_edges, directed = TRUE)

# Plot DAG
pdf('outputs/dag_causal_structure.pdf', width = 10, height = 8)
plot(dag, 
     layout = layout_with_sugiyama(dag),
     vertex.size = 25,
     vertex.color = 'lightblue',
     edge.width = 2,
     edge.arrow.size = 1.5,
     main = 'Causal DAG: Protest Duration → Outcome')
dev.off()

cat("Saved: outputs/dag_causal_structure.pdf\n")

# ============================================================================
# SECTION 5: PREPARE DATA FOR REGRESSION MODELS
# ============================================================================

# Select features for modeling
regression_features <- c('Duration_days', 'Peak_Size', 'Triggers_category',
                        'Motivations_category', 'Key_Participants_category',
                        'outcome_numeric')

regression_df <- df_processed[, regression_features] %>%
  drop_na()

cat("Regression dataset shape:", nrow(regression_df), "x", ncol(regression_df), "\n")

# Encode categorical variables using label encoding
le_triggers <- as.integer(as.factor(regression_df$Triggers_category)) - 1
le_motivations <- as.integer(as.factor(regression_df$Motivations_category)) - 1
le_participants <- as.integer(as.factor(regression_df$Key_Participants_category)) - 1

X_reg <- data.frame(
  Duration_days = regression_df$Duration_days,
  Peak_Size = regression_df$Peak_Size,
  Triggers_encoded = le_triggers,
  Motivations_encoded = le_motivations,
  Participants_encoded = le_participants
)

y_reg <- regression_df$outcome_numeric

# Standardize numeric features
scale_params <- list(
  duration_mean = mean(X_reg$Duration_days, na.rm = TRUE),
  duration_sd = sd(X_reg$Duration_days, na.rm = TRUE),
  peak_mean = mean(X_reg$Peak_Size, na.rm = TRUE),
  peak_sd = sd(X_reg$Peak_Size, na.rm = TRUE)
)

X_reg$Duration_days <- (X_reg$Duration_days - scale_params$duration_mean) / scale_params$duration_sd
X_reg$Peak_Size <- (X_reg$Peak_Size - scale_params$peak_mean) / scale_params$peak_sd

cat("Standardized features summary:\n")
print(summary(X_reg))

# ============================================================================
# SECTION 6: ORDINAL LOGISTIC REGRESSION
# ============================================================================

# Prepare data for ordinal model
X_ord <- data.frame(
  Duration_days = X_reg$Duration_days,
  Peak_Size = X_reg$Peak_Size,
  Triggers_encoded = X_reg$Triggers_encoded,
  Motivations_encoded = X_reg$Motivations_encoded,
  Participants_encoded = X_reg$Participants_encoded
)

y_ord <- ordered(y_reg, levels = 0:3)

# Fit ordinal logistic regression using proportional odds model
ord_model <- polr(y_ord ~ Duration_days + Peak_Size + Triggers_encoded + 
                        Motivations_encoded + Participants_encoded,
                  data = cbind(X_ord, y_ord = y_ord),
                  Hess = TRUE)

cat("\nOrdinal Logistic Regression Summary:\n")
print(summary(ord_model))

# Extract coefficients
ord_coefs <- coef(ord_model)
cat("\nCoefficients:\n")
print(ord_coefs)

# ============================================================================
# SECTION 7: MULTINOMIAL LOGISTIC REGRESSION
# ============================================================================

# For comparison: multinomial logistic regression
library(nnet)

# Create factor outcome (unordered)
y_multinom <- factor(y_reg, levels = 0:3,
                    labels = c('No_change', 'Partial', 'Policy_changed', 'Regime_shift'))

multinom_model <- multinom(y_multinom ~ Duration_days + Peak_Size + 
                                        Triggers_encoded + Motivations_encoded + 
                                        Participants_encoded,
                          data = cbind(X_ord, y_multinom = y_multinom),
                          trace = FALSE)

cat("\nMultinomial Logistic Regression Summary:\n")
print(summary(multinom_model))

# ============================================================================
# SECTION 8: LINEAR REGRESSION (OLS)
# ============================================================================

# Fit OLS regression treating outcome as continuous
ols_model <- lm(y_reg ~ Duration_days + Peak_Size + Triggers_encoded + 
                       Motivations_encoded + Participants_encoded,
              data = X_ord)

cat("\nOLS Regression Summary:\n")
print(summary(ols_model))

# Extract key statistics
duration_coef <- coef(ols_model)['Duration_days']
duration_se <- sqrt(diag(vcov(ols_model)))['Duration_days']
duration_pval <- coef(summary(ols_model))['Duration_days', 'Pr(>|t|)']
duration_ci <- confint(ols_model)['Duration_days', ]

cat("\nDuration Coefficient Statistics:\n")
cat("Coefficient:", duration_coef, "\n")
cat("Std Error:", duration_se, "\n")
cat("P-value:", duration_pval, "\n")
cat("95% CI: [", duration_ci[1], ",", duration_ci[2], "]\n")

# ============================================================================
# SECTION 9: OLS ASSUMPTION CHECKS
# ============================================================================

# 1. Linearity and Homoskedasticity: Residual plots
pdf('outputs/ols_residual_plot.pdf', width = 12, height = 5)
par(mfrow = c(1, 2))
plot(fitted(ols_model), residuals(ols_model), main = 'Residuals vs Fitted',
     xlab = 'Fitted values', ylab = 'Residuals')
abline(h = 0, col = 'red', lty = 2)
grid()

plot(y_reg, residuals(ols_model), main = 'Residuals vs Outcome',
     xlab = 'Outcome', ylab = 'Residuals')
abline(h = 0, col = 'red', lty = 2)
grid()
dev.off()

# 2. Breusch-Pagan test for heteroskedasticity
bp_test <- bptest(ols_model)
cat("\nBreusch-Pagan Heteroskedasticity Test:\n")
print(bp_test)

# 3. Q-Q plot and Shapiro-Wilk normality test
pdf('outputs/ols_qq_plot.pdf', width = 12, height = 5)
par(mfrow = c(1, 2))
qqnorm(residuals(ols_model), main = 'Q-Q Plot')
qqline(residuals(ols_model), col = 'red')
grid()

hist(residuals(ols_model), breaks = 30, main = 'Histogram of Residuals',
     xlab = 'Residuals', ylab = 'Frequency')
grid()
dev.off()

sw_test <- shapiro.test(residuals(ols_model))
cat("\nShapiro-Wilk Normality Test:\n")
print(sw_test)

# 4. VIF multicollinearity analysis
vif_results <- vif(ols_model)
cat("\nVariance Inflation Factors (VIF):\n")
print(vif_results)

# ============================================================================
# SECTION 10: PROPENSITY SCORE MATCHING (PSM)
# ============================================================================

# Create binary treatment: duration > median
duration_median <- median(df_processed$Duration_days, na.rm = TRUE)
df_processed$treatment_high <- as.integer(df_processed$Duration_days > duration_median)

cat("Duration median:", duration_median, "\n")
cat("Treatment distribution:\n")
print(table(df_processed$treatment_high))

# Prepare data for matching
matching_features <- c('Peak_Size', 'Triggers_category', 'Motivations_category',
                       'Key_Participants_category')

df_matching <- df_processed[, c(matching_features, 'treatment_high', 'outcome_numeric')] %>%
  drop_na()

cat("Matching dataset shape:", nrow(df_matching), "x", ncol(df_matching), "\n")

# Encode categorical variables for propensity score model
df_matching$Triggers_code <- as.integer(as.factor(df_matching$Triggers_category)) - 1
df_matching$Motivations_code <- as.integer(as.factor(df_matching$Motivations_category)) - 1
df_matching$Participants_code <- as.integer(as.factor(df_matching$Key_Participants_category)) - 1

# Fit propensity score model using logistic regression
ps_formula <- treatment_high ~ Peak_Size + Triggers_code + Motivations_code + Participants_code
ps_model <- glm(ps_formula, data = df_matching, family = binomial(link = 'logit'))

# Get propensity scores
df_matching$propensity_score <- predict(ps_model, type = 'response')

cat("Propensity score summary:\n")
cat("Mean:", mean(df_matching$propensity_score), "\n")
cat("SD:", sd(df_matching$propensity_score), "\n")
cat("Min:", min(df_matching$propensity_score), "\n")
cat("Max:", max(df_matching$propensity_score), "\n")

# 1:1 Caliper matching
caliper <- 0.1 * sd(df_matching$propensity_score)
cat("Caliper:", caliper, "\n")

treated_idx <- which(df_matching$treatment_high == 1)
control_idx <- which(df_matching$treatment_high == 0)

matched_pairs <- data.frame(treated_idx = integer(), control_idx = integer())

for (t_idx in treated_idx) {
  ps_treated <- df_matching$propensity_score[t_idx]
  
  # Find closest control within caliper
  distances <- abs(df_matching$propensity_score[control_idx] - ps_treated)
  
  if (min(distances) <= caliper) {
    closest_control <- control_idx[which.min(distances)]
    
    # Check if control hasn't been matched
    if (!closest_control %in% matched_pairs$control_idx) {
      matched_pairs <- rbind(matched_pairs, 
                            data.frame(treated_idx = t_idx, control_idx = closest_control))
    }
  }
}

cat("Number of matched pairs:", nrow(matched_pairs), "\n")

# Extract matched sample
matched_treated <- df_matching[matched_pairs$treated_idx, ]
matched_control <- df_matching[matched_pairs$control_idx, ]

# Estimate treatment effect (ATT) using simple difference in means
att_matched <- mean(matched_treated$outcome_numeric) - mean(matched_control$outcome_numeric)

# OLS regression on matched sample
X_matched <- data.frame(
  treatment = c(rep(1, nrow(matched_treated)), rep(0, nrow(matched_control))),
  outcome = c(matched_treated$outcome_numeric, matched_control$outcome_numeric)
)

ols_matched <- lm(outcome ~ treatment, data = X_matched)

cat("\nMatched Sample OLS Results:\n")
print(summary(ols_matched))

# ============================================================================
# SECTION 11: BALANCE DIAGNOSTICS FOR PSM
# ============================================================================

# Standardized Mean Difference (SMD) function
smd_calc <- function(treated_vals, control_vals) {
  mean_t <- mean(treated_vals, na.rm = TRUE)
  mean_c <- mean(control_vals, na.rm = TRUE)
  
  sd_t <- sd(treated_vals, na.rm = TRUE)
  sd_c <- sd(control_vals, na.rm = TRUE)
  
  pooled_sd <- sqrt((sd_t^2 + sd_c^2) / 2)
  
  (mean_t - mean_c) / pooled_sd
}

# Calculate SMD for each covariate before and after matching
balance_report <- data.frame()

for (var in c('Peak_Size', 'Triggers_code', 'Motivations_code', 'Participants_code')) {
  treated_all <- df_matching[df_matching$treatment_high == 1, var]
  control_all <- df_matching[df_matching$treatment_high == 0, var]
  
  smd_before <- smd_calc(treated_all, control_all)
  
  treated_matched <- matched_treated[, var]
  control_matched <- matched_control[, var]
  
  smd_after <- smd_calc(treated_matched, control_matched)
  
  balance_report <- rbind(balance_report, data.frame(
    feature = var,
    smd_before = smd_before,
    smd_after = smd_after,
    improvement = smd_before - smd_after
  ))
}

cat("\nBalance Assessment:\n")
print(balance_report)

cat("Covariates achieving balance (|SMD| < 0.1):",
    sum(abs(balance_report$smd_after) < 0.1), "/", nrow(balance_report), "\n")

# ============================================================================
# SECTION 12: VISUALIZATION - LOVE PLOT
# ============================================================================

pdf('outputs/love_plot_balance.pdf', width = 10, height = 6)
plot(1, type = 'n', xlim = c(-0.5, 0.5), ylim = c(1, nrow(balance_report)),
     xlab = 'Standardized Mean Difference', ylab = 'Covariate',
     main = 'Love Plot: Covariate Balance Before vs After Matching')

for (i in 1:nrow(balance_report)) {
  # Before
  points(balance_report$smd_before[i], i, col = 'red', pch = 16, cex = 1.5)
  
  # After
  points(balance_report$smd_after[i], i, col = 'green', pch = 17, cex = 1.5)
  
  # Connection line
  lines(c(balance_report$smd_before[i], balance_report$smd_after[i]), c(i, i),
        col = 'gray', lty = 1, lwd = 1)
}

# Add threshold lines
abline(v = c(-0.1, 0.1), col = 'gray', lty = 2)
abline(v = 0, col = 'black', lty = 1, lwd = 0.5)

axis(2, at = 1:nrow(balance_report), labels = balance_report$feature)

legend('topright', c('Before matching', 'After matching', 'Balance threshold'),
       pch = c(16, 17, NA), lty = c(NA, NA, 2), col = c('red', 'green', 'gray'))

grid()
dev.off()

cat("Saved: outputs/love_plot_balance.pdf\n")

# ============================================================================
# SECTION 13: BRANT TEST FOR PROPORTIONAL ODDS ASSUMPTION
# ============================================================================

# Brant test: Compare full ordinal model to sum of binary logits
# H0: Proportional odds assumption holds
# HA: Proportional odds assumption violated

# Fit binary logit models for each threshold
# Outcome 0 vs 1,2,3
y_bin_0v123 <- as.integer(y_ord > 0)
bin_model_0 <- glm(y_bin_0v123 ~ Duration_days + Peak_Size + Triggers_encoded + 
                                 Motivations_encoded + Participants_encoded,
                  data = cbind(X_ord, y_bin_0v123 = y_bin_0v123),
                  family = binomial())

# Outcome 0,1 vs 2,3
y_bin_01v23 <- as.integer(y_ord > 1)
bin_model_1 <- glm(y_bin_01v23 ~ Duration_days + Peak_Size + Triggers_encoded + 
                                Motivations_encoded + Participants_encoded,
                  data = cbind(X_ord, y_bin_01v23 = y_bin_01v23),
                  family = binomial())

# Outcome 0,1,2 vs 3
y_bin_012v3 <- as.integer(y_ord > 2)
bin_model_2 <- glm(y_bin_012v3 ~ Duration_days + Peak_Size + Triggers_encoded + 
                               Motivations_encoded + Participants_encoded,
                  data = cbind(X_ord, y_bin_012v3 = y_bin_012v3),
                  family = binomial())

# Extract loglikelihoods
ll_ord <- logLik(ord_model)[1]
ll_bin_sum <- logLik(bin_model_0)[1] + logLik(bin_model_1)[1] + logLik(bin_model_2)[1]

# LR test statistic
lr_stat <- 2 * (ll_bin_sum - ll_ord)
df_test <- length(coef(bin_model_0)) - length(coef(ord_model))
p_value_brant <- 1 - pchisq(lr_stat, df_test)

cat("\n=== BRANT TEST FOR PROPORTIONAL ODDS ASSUMPTION ===\n")
cat("Ordinal model log-likelihood:", ll_ord, "\n")
cat("Sum of binary logits LL:", ll_bin_sum, "\n")
cat("LR test statistic:", lr_stat, "\n")
cat("df:", df_test, "\n")
cat("p-value:", p_value_brant, "\n")

if (p_value_brant > 0.05) {
  cat("Result: Proportional odds assumption SATISFIED (p > 0.05)\n")
} else {
  cat("Result: Proportional odds assumption VIOLATED (p < 0.05)\n")
}

# ============================================================================
# SECTION 14: COMPARISON OF METHODS
# ============================================================================

comparison_results <- data.frame(
  Method = c('OLS Regression', 'Propensity Score Matching', 'Ordinal Logit'),
  Coefficient = c(duration_coef, att_matched, ord_coefs['Duration_days']),
  P_value = c(
    coef(summary(ols_model))['Duration_days', 'Pr(>|t|)'],
    coef(summary(ols_matched))['treatment', 'Pr(>|t|)'],
    NA
  ),
  Sample_Size = c(nrow(X_ord), nrow(matched_pairs) * 2, nrow(X_ord))
)

cat("\n=== METHOD COMPARISON ===\n")
print(comparison_results)

# ============================================================================
# SECTION 15: SAVE RESULTS
# ============================================================================

# Save all key results to CSV
write.csv(comparison_results, 'outputs/method_comparison.csv', row.names = FALSE)
write.csv(balance_report, 'outputs/balance_diagnostics.csv', row.names = FALSE)

# Summary statistics
summary_list <- list(
  ols_summary = summary(ols_model),
  ordinal_summary = summary(ord_model),
  psm_att = att_matched
)

cat("\n=== ANALYSIS COMPLETE ===\n")
cat("Results saved to outputs/ directory\n")
