# 1. Install and Load Required Packages
#install.packages("nnet")
#install.packages("dplyr")
#install.packages("cobalt")
library(nnet)
library(dplyr)
library(cobalt)

# 2. Load Data
url <- "https://raw.githubusercontent.com/jeevanbhatta/causal_inference/refs/heads/main/data/processed/GlobalProtestTracker_with_outcomes.csv"
data <- read.csv(url)

# 2.1 simple Multinomial regression
model1 <- multinom(outcome_label ~ Duration_days+Peak_Size+Key_Participants_category+Motivations_category+Triggers_category, data = data)
summary(model1)

# 3. Create Binary Treatment Variable: Long Protest (≥ 30 days)
data <- data %>%
  mutate(long_protest = ifelse(Duration_days >= 30, 1, 0))

# 4. Remove Cases with Missing Values in Any Model Covariates
covariates <- c("Peak_Size", "Key_Participants_category", "Motivations_category", "Triggers_category", "long_protest")
data_complete <- data %>%
  filter(complete.cases(select(., all_of(covariates))))

# 5. Estimate Propensity Score
ps_model <- glm(long_protest ~ Peak_Size + Key_Participants_category + Motivations_category + Triggers_category,
                data = data_complete, family = binomial)
data_complete$pscore <- predict(ps_model, type = "response")

# 6. Calculate IPW (Inverse Probability Weights)
data_complete <- data_complete %>%
  mutate(ipw = ifelse(long_protest == 1, 1/pscore, 1/(1-pscore)))

# 7. Weighted Multinomial Regression
model_ipw <- multinom(outcome_label ~ long_protest + Peak_Size + Key_Participants_category +
                      Motivations_category + Triggers_category,
                      data = data_complete, weights = ipw)
summary(model_ipw)

# 8. Covariate Balance Table
bal.tab(long_protest ~ Peak_Size + Key_Participants_category + Motivations_category + Triggers_category, 
        data = data_complete, weights = data_complete$ipw, un = TRUE)

# 9. Love Plot to Visualize Covariate Balance
love.plot(long_protest ~ Peak_Size + Key_Participants_category + Motivations_category + Triggers_category, 
          data = data_complete, weights = data_complete$ipw,
          threshold = 0.1,
          abs = TRUE,
          var.order = "unadjusted")

# 10. (Optional) Distributional Balance for a Specific Covariate
bal.plot(long_protest ~ Peak_Size, data = data_complete, weights = data_complete$ipw, which = "both")
