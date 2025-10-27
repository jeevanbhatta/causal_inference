install.packages("MatchIt")
library(MatchIt)
install.packages("rgenoud")
library(rgenoud)
install.packages("cobalt")
library(cobalt)

# loading the data
cash_voting <- read.csv("https://bit.ly/cash_voting")
head(cash_voting)

# MatchIt is already loaded
# code for propensity-score matching
# change parameters to achieve the best outcome
matched_outcome_ps = matchit(beneficiary ~ female + age + yrs_school + hdi_2000, data = cash_voting, caliper = 0.4, ratio=1)
love.plot(matched_outcome_ps)
bal.plot(matched_outcome_ps, "female", which = "both")
bal.plot(matched_outcome_ps, "age", which = "both")
bal.plot(matched_outcome_ps, "yrs_school", which = "both")
bal.plot(matched_outcome_ps, "hdi_2000", which = "both")

# code for Mahalanobis distance matching
# change parameters to achieve the best outcome
matched_outcome_mahalanobis = matchit(beneficiary ~ female + age + yrs_school + hdi_2000, data = cash_voting, method="full", distance="mahalanobis")
love.plot(matched_outcome_mahalanobis)
bal.plot(matched_outcome_mahalanobis, "female", which = "both")
bal.plot(matched_outcome_mahalanobis, "age", which = "both")
bal.plot(matched_outcome_mahalanobis, "yrs_school", which = "both")
bal.plot(matched_outcome_mahalanobis, "hdi_2000", which = "both")

# code for genetic matching
# change parameters to achieve the best outcome
matched_outcome_genetic = matchit(beneficiary ~ female + age + yrs_school + hdi_2000, data = cash_voting, method="genetic",pop.size=100, max.generations=50)
love.plot(matched_outcome_genetic)
bal.plot(matched_outcome_genetic, "female", which = "both")
bal.plot(matched_outcome_genetic, "age", which = "both")
bal.plot(matched_outcome_genetic, "yrs_school", which = "both")
bal.plot(matched_outcome_genetic, "hdi_2000", which = "both")
