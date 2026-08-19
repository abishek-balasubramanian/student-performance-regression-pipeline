# Student Performance Regression Pipeline

An end-to-end statistical modeling pipeline for analyzing and predicting **student exam performance** using **Multiple Linear Regression (OLS), Ridge Regression, and Lasso Regression**.

The project combines automated data preprocessing, exploratory analysis, categorical encoding, feature selection, statistical inference, regularization, regression diagnostics, and out-of-sample validation in a modular Python workflow.

## Project Overview

Student academic performance can be influenced by a combination of study behavior, classroom participation, family environment, educational resources, and demographic factors.

This project analyzes a dataset containing **6,607 students** and investigates three primary questions:

1. Which factors are statistically associated with student exam performance?
2. What is the estimated magnitude and direction of those relationships?
3. Do regularized models such as Ridge and Lasso provide better generalization than ordinary least squares?

The project was implemented as a reusable regression-analysis pipeline rather than as a single notebook.

The final comparison showed that **OLS, Ridge, and Lasso achieved almost identical predictive performance**. Because regularization provided no material improvement, **OLS was selected as the preferred model for its interpretability and statistical inference capabilities**.

---

# Key Results

| Metric | OLS | Ridge | Lasso |
|---|---:|---:|---:|
| Adjusted R² | **0.7155** | 0.7155 | 0.7155 |
| Training R² | **0.7170** | 0.7170 | 0.7170 |
| Test R² | **0.7696** | 0.7696 | 0.7697 |
| Test RMSE | **1.805** | 1.805 | 1.804 |

### Final Model

**Ordinary Least Squares Multiple Linear Regression**

was selected because:

- predictive performance was essentially identical across OLS, Ridge, and Lasso
- multicollinearity was negligible
- most predictors contained useful signal
- regularization produced no meaningful improvement
- OLS provides directly interpretable coefficients, confidence intervals, p-values, and F-tests

---

# Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Statsmodels
- SciPy
- Matplotlib
- Jupyter Notebook
- Multiple Linear Regression
- Ridge Regression
- Lasso Regression
- Statistical Hypothesis Testing

---

# Dataset

The project uses the **Student Exam Performance Factors** dataset.

The dataset contains:

```text
6,607 students
20 columns
19 predictors
1 continuous target
```

Target variable:

```text
Exam_Score
```

The raw predictors include:

- 6 numerical predictors
- 13 categorical predictors

Each row represents one student.

The dataset contains academic, behavioral, home-environment, and demographic information.

---

# Analysis Pipeline

The project was designed as a modular pipeline:

```text
Student Performance CSV
          |
          v
+-------------------------+
| Data Cleaning           |
|-------------------------|
| Type inference          |
| Duplicate detection     |
| Missing-value handling  |
| Outlier utilities       |
+-------------------------+
          |
          v
+-------------------------+
| Exploratory Analysis    |
|-------------------------|
| Distributions           |
| Pearson correlations    |
| ANOVA                   |
| Statistical summaries   |
| Visualizations          |
+-------------------------+
          |
          v
+-------------------------+
| Feature Engineering     |
|-------------------------|
| Categorical encoding    |
| Transformations         |
| Interaction utilities   |
+-------------------------+
          |
          v
     27 Predictors
          |
          v
      80/20 Split
          |
          +---------------------+
          |          |          |
          v          v          v
         OLS       Ridge      Lasso
          |          |          |
          +----------+----------+
                     |
                     v
              Model Comparison
                     |
                     v
          Regression Diagnostics
                     |
                     v
           Out-of-Sample Testing
                     |
                     v
                Final Model
                     |
                     v
                    OLS
```

---

# Data Preparation

## Column-Type Inference

The preprocessing pipeline automatically identifies predictors as either:

```text
Numeric
or
Categorical
```

based on their data types.

This makes the workflow less dependent on manually specifying every feature.

## Duplicate Handling

Exact duplicate observations are checked and removed where necessary.

## Missing Target Values

Observations with a missing `Exam_Score` are removed before modeling.

The analyzed dataset contained no missing response values.

## Categorical Encoding

Categorical variables are transformed using one-hot encoding.

For a categorical predictor containing `k` levels:

```text
k - 1
```

dummy variables are retained.

The first level is dropped to avoid perfect multicollinearity from the dummy-variable trap.

After encoding:

```text
19 raw predictors
        ↓
27 model predictors
```

---

# Train-Test Validation

The encoded dataset is split into:

```text
80% Training
20% Testing
```

using:

```text
random_state = 42
```

This produced:

```text
Training observations = 5,285
Test observations     = 1,322
```

The test set remains separate from model fitting and is used to evaluate out-of-sample predictive performance.

---

# Exploratory Data Analysis

The EDA stage evaluates both numerical and categorical predictors before regression modeling.

## Target Distribution

The average exam score is approximately:

```text
Mean Exam Score = 67.24
```

The distribution contains a right tail created by a relatively small number of high-performing students.

Reported distribution statistics include:

```text
Skewness = +1.64
Kurtosis = +10.57
Standard Deviation = 3.89
```

---

# Numerical Predictor Analysis

Pearson correlation was used to examine relationships between numerical predictors and `Exam_Score`.

Two predictors stood out particularly strongly:

```text
Attendance
r ≈ 0.58

Hours Studied
r ≈ 0.45
```

Other predictors such as:

```text
Previous Scores
Tutoring Sessions
```

showed weaker but still meaningful relationships.

In contrast:

```text
Physical Activity
Sleep Hours
```

showed little linear correlation with exam performance.

---

# Multicollinearity Exploration

Pairwise correlations among the numerical predictors were very small:

```text
|r| < 0.03
```

This provided an early indication that severe multicollinearity was unlikely.

Formal VIF diagnostics later supported the same conclusion.

---

# Categorical Predictor Analysis

Categorical predictors were evaluated using **one-way ANOVA F-tests**.

Several categorical variables showed substantial signal.

Among the strongest were:

```text
Access to Resources
Parental Involvement
Extracurricular Activities
Family Income
```

Two categorical variables did not reach statistical significance in the exploratory analysis:

```text
School Type
Gender
```

---

# Multiple Linear Regression

The primary model is:

```text
Y = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ + ε
```

where:

```text
Y   = Exam Score
X   = student-performance predictors
β   = estimated regression coefficients
ε   = residual error
```

The model is estimated using **Ordinary Least Squares (OLS)** through Statsmodels.

OLS was chosen as the primary statistical framework because it provides:

- coefficient estimates
- confidence intervals
- t-tests
- p-values
- overall F-test
- adjusted R²
- AIC
- BIC

---

# Overall Model Significance

The full OLS model produced:

```text
F(27, 5257) = 493.3
p < 10^-16
```

This indicates that the predictors collectively explain a statistically significant proportion of variation in exam performance.

---

# Most Important Predictors

The fitted model identified several strong predictors.

| Predictor | Estimated Coefficient | t-statistic | p-value |
|---|---:|---:|---:|
| Attendance | +0.199 | 79.3 | < 0.001 |
| Hours Studied | +0.293 | 60.9 | < 0.001 |
| Access to Resources — Low | -2.092 | -25.0 | < 0.001 |
| Previous Scores | +0.049 | 24.4 | < 0.001 |
| Parental Involvement — Low | -2.003 | -23.9 | < 0.001 |
| Tutoring Sessions | +0.509 | 21.8 | < 0.001 |
| Parental Involvement — Medium | -1.073 | -16.0 | < 0.001 |
| Access to Resources — Medium | -1.032 | -15.5 | < 0.001 |
| Family Income — Low | -1.106 | -13.8 | < 0.001 |
| Peer Influence — Positive | +1.045 | 13.5 | < 0.001 |

---

# Interpreting Key Coefficients

## Attendance

Estimated coefficient:

```text
β ≈ +0.199
```

Holding other predictors constant, a one-percentage-point increase in attendance is associated with approximately:

```text
+0.20 exam-score points
```

on average.

Across a larger change, such as a 20-percentage-point improvement in attendance, the estimated difference becomes much more substantial.

## Hours Studied

Estimated coefficient:

```text
β ≈ +0.293
```

Each additional weekly study hour is associated with approximately:

```text
+0.29 exam-score points
```

holding the remaining predictors constant.

## Access to Resources

Low access to educational resources is associated with approximately:

```text
-2.09 exam-score points
```

relative to the high-resource reference category.

## Parental Involvement

Low parental involvement is associated with approximately:

```text
-2.00 exam-score points
```

relative to the reference category.

## Tutoring Sessions

Each additional tutoring session is associated with approximately:

```text
+0.51 exam-score points
```

holding other predictors constant.

---

# Statistical Significance

Of the:

```text
27 encoded predictors
```

the final analysis found:

```text
24 statistically significant at α = 0.05
```

Only three predictors failed to reach significance:

```text
Sleep Hours
School Type — Public
Gender — Male
```

---

# Feature Selection

Stepwise feature selection was implemented using p-value thresholds:

```text
p_in  = 0.05
p_out = 0.10
```

Candidate specifications can be compared using:

- adjusted R²
- AIC
- BIC

This provides an additional mechanism for balancing model fit and complexity.

---

# Ridge Regression

Ridge regression introduces an L2 penalty:

```text
Loss =
Residual Sum of Squares
+
α Σ β²
```

The pipeline standardizes predictors before applying the penalty.

Candidate regularization values include:

```text
0.01
0.1
1
10
100
```

with the regularization parameter selected through **5-fold cross-validation**.

---

# Lasso Regression

Lasso introduces an L1 penalty:

```text
Loss =
Residual Sum of Squares
+
α Σ |β|
```

Lasso can shrink some coefficients toward zero and potentially perform feature selection.

The project uses `LassoCV` to determine the regularization strength automatically.

---

# OLS vs Ridge vs Lasso

All three approaches produced almost identical performance.

```text
                    OLS      Ridge     Lasso

Adjusted R²       0.7155    0.7155    0.7155
Training R²       0.7170    0.7170    0.7170
Test R²           0.7696    0.7696    0.7697
Test RMSE         1.805     1.805     1.804
```

This is an important modeling result.

Regularization generally becomes especially useful when:

- predictors are highly correlated
- coefficients are unstable
- dimensionality is large
- overfitting is substantial

Those conditions were not strongly present in this dataset.

As a result, Ridge and Lasso produced almost no improvement over OLS.

---

# Why OLS Was Selected

The final recommended model is:

# Multiple Linear Regression — OLS

OLS was selected because it provided essentially the same predictive performance as the regularized alternatives while offering much stronger interpretability.

The model provides direct access to:

```text
Coefficient Estimates
Confidence Intervals
t-statistics
p-values
F-tests
AIC
BIC
```

This is particularly valuable because the project's goal is not only prediction but also understanding which student-performance factors are statistically associated with exam outcomes.

---

# Regression Diagnostics

A major part of this project is evaluating whether the classical regression assumptions are reasonable.

The pipeline includes:

```text
Residuals vs Fitted
Normal Q-Q
Scale-Location
Residual Distribution
Breusch-Pagan Test
Shapiro-Wilk Test
Durbin-Watson Statistic
Variance Inflation Factors
```

---

# Linearity

The residuals-vs-fitted diagnostic showed no strong systematic pattern.

This supports the assumption that a linear specification provides a reasonable approximation for the relationships represented in the dataset.

---

# Homoscedasticity

The **Breusch-Pagan test** was used to evaluate constant residual variance.

The test did not reject homoscedasticity at:

```text
α = 0.05
```

The scale-location diagnostic also showed relatively stable residual spread.

---

# Residual Independence

The **Durbin-Watson statistic** was approximately:

```text
1.95
```

A value close to:

```text
2
```

indicates little evidence of residual autocorrelation.

---

# Multicollinearity

Variance Inflation Factors were calculated for the encoded predictors.

All reported VIF values were:

```text
< 1.1
```

indicating negligible multicollinearity.

This helps explain why Ridge regression provided little advantage over ordinary least squares.

---

# Residual Normality

The Shapiro-Wilk test rejected exact normality:

```text
p < 0.001
```

However, diagnostic plots showed that much of the deviation was associated with the upper tail of exam scores.

The project therefore interprets the residual distribution alongside the large training sample rather than relying exclusively on the formal normality test.

---

# Model Validation

The final OLS model achieved:

```text
Training R² = 0.7170

Test R² = 0.7696

Test RMSE = 1.805
```

The held-out test performance provides evidence that the model generalizes effectively to observations not used during fitting.

---

# Repository Architecture

```text
student-performance-regression-pipeline/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── StudentPerformanceFactors.csv
│
├── data_cleaning/
│   ├── preprocess.py
│   └── helpers/
│       ├── duplicates.py
│       ├── missing.py
│       ├── outliers.py
│       └── types.py
│
├── exploratory_analysis/
│   ├── exploratory-analysis.ipynb
│   └── helpers/
│       ├── stats.py
│       └── visualize.py
│
├── feature_engineering/
│   ├── add_new_features.py
│   └── helpers/
│       ├── encodings.py
│       ├── interactions.py
│       └── transforms.py
│
├── modeling/
│   ├── fit_model.py
│   └── helpers/
│       ├── comparison.py
│       ├── diagnostics.py
│       ├── lasso.py
│       ├── metrics.py
│       ├── mlr.py
│       ├── ridge.py
│       ├── save_outputs.py
│       ├── selection.py
│       └── validation.py
│
├── reports/
│   ├── model_outputs/
│   └── summary/
│
└── docs/
    └── project-report.pdf
```

---

# Module Overview

## `data_cleaning/`

Contains reusable preprocessing functionality for:

- duplicate handling
- missing-data handling
- outlier utilities
- data-type identification

## `exploratory_analysis/`

Contains the EDA notebook and supporting statistical/visualization utilities.

## `feature_engineering/`

Contains reusable utilities for:

- categorical encoding
- transformations
- interaction features

## `modeling/`

Contains the main modeling pipeline and model-specific modules.

### `mlr.py`

Multiple Linear Regression implementation.

### `ridge.py`

Ridge regression and regularization workflow.

### `lasso.py`

Lasso regression and cross-validated regularization.

### `selection.py`

Predictor-selection utilities.

### `diagnostics.py`

Classical regression diagnostic tests and plots.

### `metrics.py`

Model evaluation metrics.

### `comparison.py`

OLS/Ridge/Lasso model comparison.

### `validation.py`

Out-of-sample validation utilities.

### `save_outputs.py`

Saves model summaries, coefficients, metrics, and other generated results.

---

# Generated Outputs

The pipeline generates reproducible modeling artifacts under:

```text
reports/
```

including model outputs such as:

```text
MLR coefficients
Ridge coefficients
Lasso coefficients

MLR metrics
Ridge metrics
Lasso metrics

Model summaries
Diagnostic outputs
Comparison results
```

This allows the statistical results to be inspected without rerunning every stage manually.

---

# Running the Project

## 1. Clone the Repository

```bash
git clone <repository-url>
cd student-performance-regression-pipeline
```

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the Regression Pipeline

From the repository root:

```bash
python -m modeling.fit_model \
  --data data/StudentPerformanceFactors.csv \
  --target Exam_Score
```

On Windows PowerShell, the same command can be entered on one line:

```bash
python -m modeling.fit_model --data data/StudentPerformanceFactors.csv --target Exam_Score
```

The pipeline performs:

```text
Load Dataset
    ↓
Preprocess
    ↓
Encode Predictors
    ↓
Train/Test Split
    ↓
Fit OLS
    ↓
Fit Ridge
    ↓
Fit Lasso
    ↓
Calculate Metrics
    ↓
Run Diagnostics
    ↓
Compare Models
    ↓
Save Results
```

---

# Key Concepts Demonstrated

- Multiple Linear Regression
- Ordinary Least Squares
- Ridge Regression
- Lasso Regression
- Regularization
- Cross-Validation
- Statistical Inference
- Hypothesis Testing
- Feature Selection
- One-Hot Encoding
- Exploratory Data Analysis
- Pearson Correlation
- ANOVA
- Adjusted R²
- RMSE
- MAE
- AIC
- BIC
- Variance Inflation Factor
- Breusch-Pagan Test
- Durbin-Watson Statistic
- Shapiro-Wilk Test
- Residual Diagnostics
- Out-of-Sample Validation
- Modular Python Pipelines
- Reproducible Data Science

---

# Limitations

Several limitations should be considered when interpreting the results.

### Observational Data

The dataset is cross-sectional and observational.

Therefore, regression coefficients represent **associations**, not causal effects.

For example, the positive coefficient on study hours does not by itself prove that increasing study time by one hour will causally increase a student's score by exactly the estimated amount.

### Self-Reported Predictors

Variables such as:

```text
Hours Studied
Sleep Hours
Motivation Level
```

may contain measurement error because they are self-reported.

### Exam-Score Ceiling

A small number of students score near the upper limit of the exam scale, contributing to the right-tail behavior observed in the residual distribution.

### Interaction Effects

The primary specification focuses on main effects.

Potential interactions such as:

```text
Hours Studied × Access to Resources
```

could be investigated in future extensions.

---

# Future Work

Potential extensions include:

- interaction-effect modeling
- nonlinear regression
- censored regression for the exam-score ceiling
- additional feature engineering
- external validation on another student-performance dataset
- deployment as an interactive prediction application
- comparison with tree-based regression models

---

# Academic Context

Developed as part of graduate coursework at **Arizona State University**.

The project applies a complete multiple-linear-regression methodology to student-performance data, including exploratory analysis, model fitting, statistical inference, regularized comparison models, regression diagnostics, and held-out validation.

The work was completed collaboratively as a team project.
