# Data Transformation Summary: GlobalProtestTracker → GlobalProtestTracker_with_outcomes

## Overview
The dataset was transformed from the raw `GlobalProtestTracker.csv` (328 protests) to the processed `GlobalProtestTracker_with_outcomes.csv` (329 protests) by adding 5 new engineered features for causal inference analysis.

## Changes Made

### 1. **5 New Columns Added**

#### a) `outcome_label` (Ordinal Outcome Variable)
**Purpose**: Transforms the text-based "Outcomes" column into a standardized categorical variable for causal analysis.

**Mapping**:
- `"No significant change"` → 0 (No change)
- `"No policy/leadership change in response to protest."` → 0 (No change)
- `"No policy/leadership change in response to protests."` → 0 (No change)
- `"Policy changed to meet demands (fully changed/reversed)"` → 1 (Policy change)
- `"partial political change"` → 2 (Partial political change)
- `"regime shift"` → 3 (Regime shift)

**Why**: This creates a 4-level ordinal outcome variable representing increasing levels of protest success, preserving the ordinal nature of outcomes rather than collapsing to binary.

**Sample Values**:
```
No significant change → 0
Policy changed to meet demands (fully changed/reversed) → 1
partial political change → 2
regime shift → 3
```

---

#### b) `Duration_days` (Continuous Treatment Variable)
**Purpose**: Converts text duration descriptions into numeric days for regression analysis.

**Conversion Rules**:
- `"1 day"` → 1
- `"1 week"` → 7
- `"2 weeks"` → 14
- `"2 months"` → 60
- `"1 month"` → 30
- `"3 weeks"` → 21
- `"5 months"` → 150
- `"6 months"` → 180
- `"8 months"` → 240
- `"9 months"` → 270
- `"11 months"` → 330
- `"1 year (sporadic)"` → 365
- `"1 year, intermittently"` → 365
- `"1 year (first round); 2 months (second wave)"` → 485 (combined)
- `"Active"` → Removed during analysis (ongoing protests)
- `"6 months in 2020–2021; Active as of May 2022"` → 1803 (calculated)

**Why**: Converts qualitative duration descriptions to quantitative days for use as the treatment variable in regression models. The primary causal question: "What is the effect of protest duration on policy/political change?"

**Treatment Definition** (for matching):
- Long-duration protests: `Duration_days > median(30 days)` = 1 (treatment)
- Short-duration protests: `Duration_days ≤ median(30 days)` = 0 (control)

---

#### c) `Peak_Size` (Numeric Confounder)
**Purpose**: Converts categorical peak protest size to numeric values for confounding control.

**Conversion Rules**:
- `">1,000"` → 1,000
- `">4,000"` → 4,000
- `">10,000"` → 10,000
- `">100,000"` → 100,000
- `">1,000,000"` → 1,000,000
- `">2,000"` → 2,000
- `">5,000"` → 5,000
- `">40,000"` → 40,000
- `"<1,000"` → 500 (midpoint of assumed range)
- `">500"` → 500 (minimum estimate)
- `"30,000"` → 30,000
- `"5,000"` → 5,000
- `"15,000"` → 15,000
- `"1,500,000"` → 1,500,000
- Single values kept as-is

**Why**: Peak size is a key confounder—larger protests may be more likely to succeed AND more likely to last longer. Controlling for this prevents bias in causal estimates.

---

#### d) `Triggers_category` (Categorical Confounder)
**Purpose**: Consolidates 20+ unique trigger descriptions into 8 standardized categories for causal modeling.

**8 Categories**:
1. **Government policy/law change** - Policy shifts, legal reforms, government decisions
   - Examples: "Proposal of austerity measures", "Introduction of taxes"
   
2. **Election/political process** - Electoral fraud, electoral delays, reelection attempts
   - Examples: "Fraud in elections", "Electoral irregularities"
   
3. **Corruption/abuse allegations** - Corruption scandals, official misconduct
   - Examples: "Detention of opposition activists", "Anticorruption blogger arrested"
   
4. **Economic crisis/prices** - Rising prices, inflation, currency crises
   - Examples: "Rapid depreciation of currency", "Rising cost of living"
   
5. **Security/conflict incident** - Military conflicts, security threats, invasions
   - Examples: "Military takeover", "Invasion of Ukraine"
   
6. **Death/violence incident** - Deaths, killings, violent incidents
   - Examples: "Truck bombing", "Fatal shooting by officer"
   
7. **Infrastructure/disaster** - Natural disasters, infrastructure failures
   - Examples: "Australian bushfires", "Beirut Port explosion"
   
8. **Other** - Miscellaneous triggers not fitting above categories
   - Examples: "Announcement of peace deal", "Coronavirus restrictions"

**Why**: Triggers are potential confounders—different types of triggers may have different success rates and durations. Categorization enables statistical control.

---

#### e) `Motivations_category` (Categorical Confounder)
**Purpose**: Consolidates 15+ unique motivation descriptions into 8 standardized categories.

**8 Categories**:
1. **Economic inequality/livelihood** - Wages, employment, economic opportunity, inflation
   - Examples: "Rising inflation", "Low wages", "Unemployment"
   
2. **Political freedom/democracy** - Democratic rights, political pluralism, democratic backslide
   - Examples: "Democratic backslide", "Desire for political reforms"
   
3. **Corruption/accountability** - Corruption, impunity, lack of accountability
   - Examples: "Rampant official corruption", "Lack of accountability"
   
4. **Identity/rights (ethnic/religious/gender)** - Rights for minorities, women, ethnic/religious groups
   - Examples: "Women's rights", "Ethnic discrimination"
   
5. **Security/violence concerns** - Police brutality, violence, security threats
   - Examples: "Police violence", "Insecurity"
   
6. **Social services/infrastructure** - Healthcare, education, services
   - Examples: "Education cuts", "Poor public services"
   
7. **Environmental concerns** - Climate change, environmental protection
   - Examples: "Climate change", "Environmental degradation"
   
8. **Other** - Miscellaneous motivations
   - Examples: "Nationalist sentiment", "Vaccine skepticism"

**Why**: Motivations are confounders—different motivation types have different success rates and protest durations. Control enables cleaner causal identification.

---

#### f) `Key_Participants_category` (Categorical Confounder)
**Purpose**: Consolidates 12+ participant types into 8 standardized groups.

**8 Categories**:
1. **General public** - Unorganized public citizens
   - Examples: "General public", "Citizens"
   
2. **Labor/unions/workers** - Labor organizations, worker groups, trade unions
   - Examples: "Labor unions", "Garment workers"
   
3. **Students/youth** - Student organizations, youth groups
   - Examples: "Students", "Youth groups"
   
4. **Opposition parties/politicians** - Political opposition, opposition leaders
   - Examples: "Opposition parties", "Opposition leaders"
   
5. **Ethnic/religious groups** - Ethnic minorities, religious communities
   - Examples: "Ethnic Hazaras", "Muslim groups"
   
6. **Women/feminist groups** - Women's organizations, feminist movements
   - Examples: "Women", "Feminist groups"
   
7. **Professional organizations** - Doctors, nurses, teachers, professionals
   - Examples: "Healthcare workers", "Teachers"
   
8. **Mixed/multiple groups** - Coalition of diverse participant groups
   - Examples: "Coalition of parties", "Multiple groups"

**Why**: Participant type is a confounder—different groups have different organizing capacity, resources, and success rates. Different groups may also have different protest durations.

---

### 2. **Data Quality Notes**

| Metric | Raw Data | Processed Data |
|--------|----------|-----------------|
| Total rows | 328 | 329 |
| Rows with valid outcomes | 328 | 326 (after removing "Active" protesting) |
| Rows with duration > 0 | Variable | 326 (after removing "Active") |
| New numeric features | 0 | 2 (Duration_days, Peak_Size) |
| New categorical features | 0 | 3 (Triggers, Motivations, Participants) |

**Rows Removed During Analysis**:
- 3 rows with "Active" duration status (ongoing protests at data collection)
- These are excluded because end-state outcomes cannot be determined

---

## Implementation Details

### Where These Changes Were Made
**Script**: `src/etl.py` (or within the notebook ETL section)

### Key Functions
1. **Duration conversion** - Parsed text duration → numeric days
2. **Outcome mapping** - Text outcomes → 0-3 ordinal scale
3. **Peak size extraction** - Text descriptions → numeric values
4. **Categorization** - Triggers, motivations, participants mapped to 8-category schemas

### Validation Steps
1. **No missing values** in new columns (after removing Active protesting)
2. **Unique value counts**:
   - `outcome_label`: 4 unique values (0, 1, 2, 3)
   - `Triggers_category`: 8 unique categories
   - `Motivations_category`: 8 unique categories
   - `Key_Participants_category`: 8 unique categories
3. **Distribution checks**: Each category represents reasonable number of protests

---

## Causal Inference Applications

### Treatment Variable
- **`Duration_days`**: Continuous treatment (also binarized as long vs. short duration)

### Outcome Variable
- **`outcome_label`**: 4-level ordinal outcome (0=no change, 1=policy change, 2=partial political change, 3=regime shift)

### Confounders (Controlled in Models)
- **`Peak_Size`**: Protest magnitude (confounder for both treatment and outcome)
- **`Triggers_category`**: Type of trigger event (confounder)
- **`Motivations_category`**: Protest motivation (confounder)
- **`Key_Participants_category`**: Participant organizations (confounder)

### Causal Research Question
**What is the causal effect of protest duration on the likelihood of policy or political change?**

- **Treatment**: Long-duration protests (>30 days median)
- **Outcome**: Success (policy change, partial political change, or regime shift)
- **Methods**: OLS regression, Propensity Score Matching

---

## Summary

The transformation from raw to processed data involved:
1. ✅ **5 new feature engineering steps** creating treatment, outcome, and confounder variables
2. ✅ **Categorical consolidation** standardizing triggers, motivations, and participants into 8-category schemas
3. ✅ **Numeric conversion** transforming text-based duration and size into continuous variables
4. ✅ **Ordinal outcome creation** preserving multi-level success outcomes (not binary)
5. ✅ **Data quality filtering** removing 3 rows with unresolved ("Active") outcomes

**Result**: A clean, causal-inference-ready dataset with properly engineered treatment, outcome, and confounder variables for regression and matching analysis.
