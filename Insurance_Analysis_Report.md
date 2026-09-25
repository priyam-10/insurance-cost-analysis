# Insurance Cost Analysis --- Final Report

**Project:** Insurance Cost Exploratory Data Analysis and Feature
Selection\
**Dataset:** `insurance.csv`\
**Notebook:** `Insurance.ipynb`\
**Target Variable:** `charges`\
**Analysis Type:** Exploratory Data Analysis (EDA), Data Cleaning,
Feature Engineering, Correlation Analysis, and Statistical Feature
Selection

------------------------------------------------------------------------

## 1. Executive Summary

This project analyzes a health insurance dataset containing **1,338
observations and 7 original variables**. The objective is to understand
the structure of insurance charges and identify variables that show
meaningful statistical relationships with the target variable,
`charges`.

The analysis includes:

1.  Exploratory data analysis
2.  Data-quality checks
3.  Duplicate removal
4.  Categorical-variable encoding
5.  BMI category feature engineering
6.  Standardization of numerical variables
7.  Pearson correlation analysis
8.  Chi-square feature selection for categorical variables
9.  Construction of a final analysis dataset

The strongest linear association with insurance charges is observed for
**smoking status (`is_smoker`)**, with a Pearson correlation of
approximately **0.787**. Age has the next-largest positive correlation
at approximately **0.298**. BMI has a smaller positive correlation of
approximately **0.196**.

The chi-square analysis, using a significance level of **0.05**,
identifies `is_smoker`, `region_southeast`, and `is_female` as
statistically associated with charge quartiles. The engineered
`bmi_category_Obese` variable does **not** meet the 0.05 threshold in
the executed analysis (`p ≈ 0.0537`), although it remains in the
notebook's final selected dataframe.

> **Interpretation note:** These results describe statistical
> associations in this dataset. They do not establish causal
> relationships.

------------------------------------------------------------------------

## 2. Dataset Overview

The original dataset contains the following variables:

  Variable     Description                     Role
  ------------ ------------------------------- -----------------------
  `age`        Age of the individual           Predictor
  `sex`        Sex of the individual           Categorical predictor
  `bmi`        Body Mass Index                 Predictor
  `children`   Number of children/dependents   Predictor
  `smoker`     Smoking status                  Categorical predictor
  `region`     Residential region              Categorical predictor
  `charges`    Medical insurance charges       Target

### Dataset dimensions

-   **Rows:** 1,338
-   **Columns:** 7
-   **Duplicate records:** 1
-   **Rows after duplicate removal:** 1,337

No missing values were identified during the notebook's data-quality
checks.

------------------------------------------------------------------------

## 3. Descriptive Statistics

The following statistics are calculated from the dataset after duplicate
removal.

  Variable          Mean     Median    Minimum     Maximum
  ---------- ----------- ---------- ---------- -----------
  Age              39.22      39.00      18.00       64.00
  BMI              30.66      30.40      15.96       53.13
  Children          1.10       1.00       0.00        5.00
  Charges      13,279.12   9,386.16   1,121.87   63,770.43

The target variable, `charges`, has a substantially higher maximum than
its median, indicating that insurance costs are distributed unevenly and
include high-cost observations.

------------------------------------------------------------------------

## 4. Categorical Distribution

### Sex

After duplicate removal:

  Category     Count
  ---------- -------
  Male           675
  Female         662

The dataset is relatively balanced by sex.

### Smoking Status

  Category       Count
  ------------ -------
  Non-smoker     1,063
  Smoker           274

Smokers represent a smaller portion of the dataset.

### Region

  Region        Count
  ----------- -------
  Southeast       364
  Southwest       325
  Northwest       324
  Northeast       324

The regional distribution is broadly balanced, with the Southeast
containing the largest number of observations.

------------------------------------------------------------------------

## 5. Exploratory Data Analysis

The notebook examines the numerical variables using:

-   Histograms with KDE
-   Boxplots
-   Count plots for categorical variables
-   A correlation heatmap

The numerical variables analyzed are:

-   `age`
-   `bmi`
-   `children`
-   `charges`

The categorical variables examined include:

-   `sex`
-   `smoker`
-   `region`

The visual analysis is intended to identify distributions, potential
outliers, class balance, and relationships between variables before
preprocessing and statistical testing.

------------------------------------------------------------------------

## 6. Data Cleaning

A copy of the original dataframe was created for preprocessing.

### Duplicate removal

The notebook applies:

``` python
df_cleaned.drop_duplicates(inplace=True)
```

This removes **1 duplicate record**, reducing the dataset from **1,338
to 1,337 observations**.

### Missing values

The notebook checks missing values before and after cleaning. No missing
values are identified.

------------------------------------------------------------------------

## 7. Categorical Encoding

### Sex encoding

The original `sex` variable is transformed as:

``` text
male   → 0
female → 1
```

The resulting column is renamed:

``` text
is_female
```

### Smoking encoding

The original `smoker` variable is transformed as:

``` text
no  → 0
yes → 1
```

The resulting column is renamed:

``` text
is_smoker
```

### Region encoding

One-hot encoding is applied to `region` using:

``` python
pd.get_dummies(df_cleaned, columns=['region'], drop_first=True)
```

The resulting regional indicator variables are:

-   `region_northwest`
-   `region_southeast`
-   `region_southwest`

The Northeast category functions as the omitted reference category.

------------------------------------------------------------------------

## 8. Feature Engineering --- BMI Categories

A derived categorical variable called `bmi_category` is created using
the following bins:

  BMI           Category
  ------------- -------------
  `< 18.5`      Underweight
  `18.5–24.9`   Normal
  `25.0–29.9`   Overweight
  `≥ 30.0`      Obese

The BMI categories are then one-hot encoded with the first category
dropped.

The resulting variables used in the analysis are:

-   `bmi_category_Normal`
-   `bmi_category_Overweight`
-   `bmi_category_Obese`

------------------------------------------------------------------------

## 9. Numerical Feature Scaling

The notebook standardizes:

-   `age`
-   `bmi`
-   `children`

using `StandardScaler`.

Standardization transforms these variables to a common scale based on
their mean and standard deviation.

The target variable, `charges`, is not standardized.

------------------------------------------------------------------------

## 10. Pearson Correlation Analysis

Pearson correlation is used to measure the linear association between
each selected feature and `charges`.

### Results

  Feature                       Pearson Correlation
  --------------------------- ---------------------
  `is_smoker`                            **0.7872**
  `age`                                  **0.2983**
  `bmi_category_Obese`                   **0.1977**
  `bmi`                                  **0.1962**
  `region_southeast`                         0.0736
  `children`                                 0.0674
  `region_northwest`                        -0.0387
  `region_southwest`                        -0.0436
  `is_female`                               -0.0580
  `bmi_category_Normal`                     -0.1057
  `bmi_category_Overweight`                 -0.1183

### Interpretation

`is_smoker` has the largest Pearson correlation with `charges` at
approximately **0.7872**, indicating a strong positive linear
association within the analyzed data.

`age` has the second-largest positive correlation at approximately
**0.2983**.

BMI and the engineered obese BMI category have positive but smaller
correlations, approximately **0.1962** and **0.1977**, respectively.

The remaining variables show relatively weak linear correlations with
charges.

------------------------------------------------------------------------

## 11. Chi-Square Feature Selection

The notebook evaluates categorical features using a chi-square test of
independence.

First, `charges` is divided into four approximately equal-sized groups
using quartiles:

``` python
pd.qcut(df_cleaned['charges'], q=4, labels=False)
```

The resulting variable is:

``` text
charges_bin
```

The significance level is:

``` text
α = 0.05
```

### Results

  -------------------------------------------------------------------------------------
  Feature                     Chi-Square Statistic              p-value Decision at α =
                                                                        0.05
  --------------------------- -------------------- -------------------- ---------------
  `is_smoker`                             848.2192          \< 0.000001 **Keep /
                                                                        Significant**

  `region_southeast`                       15.9982             0.001135 **Keep /
                                                                        Significant**

  `is_female`                              10.2588             0.016490 **Keep /
                                                                        Significant**

  `region_southwest`                        5.0919             0.165191 Drop / Not
                                                                        significant

  `bmi_category_Overweight`                 4.2016             0.240504 Drop / Not
                                                                        significant

  `bmi_category_Normal`                     4.2637             0.234364 Drop / Not
                                                                        significant

  `bmi_category_Obese`                      7.6545             0.053720 Drop / Not
                                                                        significant

  `region_northwest`                        1.1342             0.768815 Drop / Not
                                                                        significant
  -------------------------------------------------------------------------------------

### Statistical interpretation

At the 5% significance level:

-   `is_smoker` shows a statistically significant association with
    charge quartiles.
-   `region_southeast` shows a statistically significant association
    with charge quartiles.
-   `is_female` shows a statistically significant association with
    charge quartiles.
-   The remaining tested categorical variables do not reach statistical
    significance at `α = 0.05`.

The `bmi_category_Obese` result is particularly close to the threshold
(`p ≈ 0.0537`) but does not meet the predefined 0.05 criterion.

------------------------------------------------------------------------

## 12. Final Feature Set in the Notebook

The notebook constructs the final dataframe using:

``` python
[
    'age',
    'is_female',
    'bmi',
    'children',
    'is_smoker',
    'charges',
    'region_southeast',
    'bmi_category_Obese'
]
```

Therefore, the final dataset contains:

  Feature                Role
  ---------------------- -----------
  `age`                  Predictor
  `is_female`            Predictor
  `bmi`                  Predictor
  `children`             Predictor
  `is_smoker`            Predictor
  `charges`              Target
  `region_southeast`     Predictor
  `bmi_category_Obese`   Predictor

### Important methodological note

The notebook's final dataframe retains `bmi_category_Obese` even though
its chi-square p-value is approximately **0.0537**, which is slightly
above the selected significance threshold of **0.05**.

This means the final dataframe should be described as the **notebook's
final selected dataset**, rather than as a dataset containing only
statistically significant chi-square features.

------------------------------------------------------------------------

## 13. Key Findings

### Finding 1 --- Smoking status has the strongest association

The Pearson correlation between `is_smoker` and `charges` is
approximately:

**r = 0.7872**

This is substantially larger than the correlation of any other feature
tested.

The chi-square analysis also produces an extremely small p-value for
`is_smoker`, providing strong statistical evidence of an association
between smoking status and charge quartiles in this dataset.

### Finding 2 --- Age shows a positive association

Age has:

**r = 0.2983**

This indicates a positive linear association between age and insurance
charges in the analyzed data.

### Finding 3 --- BMI has a weaker positive relationship

BMI has:

**r = 0.1962**

The engineered `bmi_category_Obese` variable has:

**r = 0.1977**

These relationships are considerably weaker than the observed
relationship for smoking status.

### Finding 4 --- Regional effects differ by region

The `region_southeast` variable is statistically significant in the
chi-square analysis (`p ≈ 0.001135`), while `region_northwest` and
`region_southwest` are not statistically significant at the 5% level.

### Finding 5 --- Gender shows a statistically significant categorical association

`is_female` has a chi-square p-value of approximately **0.01649**,
meeting the 0.05 significance threshold.

However, its Pearson correlation with charges is small:

**r = -0.0580**

This illustrates that statistical association with charge groups and
linear correlation are different analytical concepts.

------------------------------------------------------------------------

## 14. Business / Analytical Interpretation

The analysis suggests that **smoking status is the most prominent
variable associated with insurance charges in this dataset**.

Age also demonstrates a meaningful positive relationship, while BMI has
a smaller positive relationship.

Regional and demographic variables show comparatively smaller linear
relationships, although some categorical variables demonstrate
statistical association when charges are divided into quartiles.

These findings can be useful for:

-   exploratory insurance-cost analysis
-   feature selection
-   predictive-model development
-   understanding customer risk segments
-   preparing a regression modeling pipeline

They should not, however, be interpreted as proof that any individual
characteristic directly causes a particular insurance charge.

------------------------------------------------------------------------

## 15. Limitations

This analysis has several limitations.

### 15.1 No predictive model is evaluated

The notebook does not currently train or evaluate regression models such
as:

-   Linear Regression
-   Ridge/Lasso Regression
-   Random Forest
-   Gradient Boosting

Therefore, model accuracy and predictive performance have not been
established.

### 15.2 Correlation does not imply causation

Pearson correlation measures linear association. It cannot establish
that one variable causes changes in insurance charges.

### 15.3 Chi-square analysis uses charge quartiles

The continuous `charges` variable is converted into four groups for the
chi-square analysis. This simplifies the target and may discard
information contained in the original continuous values.

### 15.4 BMI categorization loses information

Converting continuous BMI into categories can make interpretation easier
but can also remove some of the information contained in the original
BMI measurement.

### 15.5 Dataset scope

The conclusions apply to the observations contained in this dataset.
They should not automatically be generalized to other insurance
populations without additional validation.

------------------------------------------------------------------------

## 16. Recommended Next Steps

For a complete machine-learning project, the next stage should be:

1.  Define `charges` as the regression target.
2.  Split the data into training and test sets.
3.  Build baseline regression models.
4.  Compare multiple algorithms.
5.  Evaluate using:
    -   MAE
    -   MSE
    -   RMSE
    -   R²
6.  Perform cross-validation.
7.  Tune important model hyperparameters.
8.  Analyze feature importance.
9.  Compare predicted versus actual charges.
10. Document model limitations and conclusions.

This would extend the current project from **EDA and statistical feature
selection** into a complete **insurance cost prediction project**.

------------------------------------------------------------------------

## 17. Final Conclusion

The project successfully performs data exploration, cleaning,
categorical encoding, BMI feature engineering, numerical
standardization, correlation analysis, and categorical feature
selection.

The most notable result is the strong positive association between
**smoking status and insurance charges (`r ≈ 0.7872`)**. Age also shows
a positive association (`r ≈ 0.2983`), while BMI-related variables show
weaker positive relationships.

The chi-square analysis identifies statistically significant
associations for **smoking status, Southeast region, and sex** at the 5%
significance level. The engineered obese BMI category is close to, but
does not cross, the 0.05 significance threshold.

Overall, the notebook provides a solid foundation for the next stage:
developing and evaluating a regression model capable of predicting
insurance charges.

------------------------------------------------------------------------

## Project Files

Recommended GitHub repository structure:

``` text
insurance-cost-analysis/
│
├── Insurance.ipynb
├── insurance.csv
├── Insurance_Analysis_Report.md
└── README.md
```

**Analysis status:** Completed through exploratory analysis,
preprocessing, feature engineering, correlation analysis, and
statistical feature selection.

**Next phase:** Predictive modeling and model evaluation.
