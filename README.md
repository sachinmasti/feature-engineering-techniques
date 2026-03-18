# 🛠️ Feature Engineering Techniques

A collection of feature engineering techniques documented with practical examples using Python and Pandas.

---

## 📌 About

This repository contains hands-on notebooks covering various feature engineering techniques used in data preprocessing and machine learning pipelines. The goal is to document useful techniques for future reference and learning.

---

## 📂 Notebooks

| Notebook | Description |
|---|---|
| `feature_engineering_for_mixed_data_string_and_numbers.ipynb` | Handling columns with mixed data (strings + numbers) using `str.extract`, `str.split`, and `pd.to_numeric` |
| `Date_time_data_handing.ipynb` | Extracting and engineering features from DateTime columns using `pd.to_datetime`, `.dt` accessor, cyclical encoding, and time-based derived features |
| `Missing_Data_Mechanisms_MCAR_MAR_MNAR_v2.ipynb` | Understanding and simulating missing data mechanisms — MCAR, MAR, and MNAR — with practical examples using NumPy and Pandas |
| `Feature_Engineering_Stats_Guide_Commented.ipynb` | Statistical testing for feature selection — ANOVA (F-test) and Eta Squared (effect size) to evaluate the impact of categorical variables on numeric features |
| `Handling_Missing_Numerical_Data_GOLD_VERSION.ipynb` | Numerical missing value imputation strategies — Mean, Median, Arbitrary Value, End of Distribution, and Random Sample Imputation — with model performance comparison using Linear Regression and R² Score |
| `Categorical_Missing_Value_Handling.ipynb` | Categorical missing value imputation strategies — Most Frequent, Constant ('Missing' category), Random Imputation, and Missing Indicator — with concept explanation, use cases, and risks for each method |
| `KNN_Iterative_Imputer_Guide.ipynb` | Advanced missing value imputation using KNN Imputer (distance-based) and Iterative Imputer (model-based / MICE) — covers when to use each, industry best practices, data leakage prevention, and Pipeline integration |
| `Outlier_Handling_Master_Notebook.ipynb` | Complete outlier detection and treatment guide — IQR Method, Z-Score Method, Capping/Winsorization, Log Transformation, Robust Scaling, and model-based comparison (Linear Regression vs Random Forest) |
| `Feature_Construction_Reference_Final.ipynb` | Practical feature construction techniques — Date features, Count-based features, Multi-value flags, Binary keyword flags, Aggregation features, and Interaction features — with Hinglish explanations and industry-style examples |
| `Feature_Engineering_Hacks.ipynb` | 10 powerful feature engineering hacks — Target Encoding, Frequency Encoding, Binning, Lag Features, Rolling Window, Polynomial Features, Log Transformation, Ratio Features, Missing Indicators, and Text-Based Features — with Hinglish comments and a common sample dataset |

---

## 🧪 Techniques Covered

### ✅ Currently Added

- **Mixed Data Handling** — Extracting numbers/letters from columns like `B1-1000`, `z9-1299` using regex and string methods
- **str.extract** — Extract specific patterns from string columns
- **str.split** — Split a column into multiple columns by a delimiter
- **pd.to_numeric** — Convert string columns to numeric with error handling
- **Date/Time Feature Extraction** — Parsing datetime strings, extracting year, month, day, hour, weekday, quarter, and more using `.dt` accessor
- **Derived DateTime Features** — Creating `is_weekend`, `is_month_end`, `day_name`, `month_name` from datetime columns
- **Cyclical Encoding** — Encoding cyclic features like month and day using sine/cosine transformation to preserve circular nature
- **Time Difference Features** — Calculating days/hours elapsed since a reference date
- **Missing Data Mechanisms (MCAR, MAR, MNAR)** — Simulating and identifying different types of missingness; understanding when MAR assumption holds for imputation in ML workflows
- **ANOVA (One-Way F-Test)** — Statistical test to check whether a categorical variable has a significant effect on a numeric variable across multiple groups
- **Eta Squared (Effect Size)** — Measures how strongly a categorical variable explains variance in a numeric variable; used alongside ANOVA for complete feature evaluation
- **Numerical Missing Value Imputation** — Five imputation strategies for numerical data with train/test split to avoid data leakage, and R² Score-based model performance comparison across all methods
- **Categorical Missing Value Imputation** — Four strategies for categorical data: Most Frequent, Constant/Missing category, Random Imputation, and Missing Indicator; includes when-to-use, risks, and production-level best practices
- **KNN Imputer** — Distance-based imputation using K-nearest neighbors; fills missing values based on the most similar rows using Euclidean distance; includes scaling requirements and n_neighbors tuning
- **Iterative Imputer (MICE)** — Model-based imputation using chained equations; treats each column with missing values as a regression target and predicts it using remaining features; supports custom estimators and multiple iterations for convergence
- **Outlier Detection & Treatment** — End-to-end outlier handling pipeline covering IQR Method, Z-Score Method, Capping/Winsorization, Log Transformation, RobustScaler, and model-level robustness comparison between Linear Regression and Random Forest
- **Date Feature Construction** — Breaking date columns into year, month, dayofweek, and is_weekend flags for time-based pattern modeling
- **Count-Based Features** — Counting elements in comma-separated columns like cast, genres, tags using `str.split` + `apply(len)`
- **Multi-Value Flags** — Creating binary flags from multi-value columns (e.g., `multi_country`) to reduce complexity
- **Binary Keyword Flags** — Generating `is_drama`, `is_comedy` style features from high-cardinality categorical columns using `str.contains`
- **Aggregation Features** — Adding group-level behavior (sum, mean, count) using `groupby` + `transform` for user/product/category-level signals
- **Interaction Features** — Combining two features to create stronger signals (e.g., `age_duration`, `cast_per_duration`)
- **Target Encoding** — Replacing each category with its mean target value; powerful for high-cardinality columns with tree-based models
- **Frequency Encoding** — Replacing each category with its proportion in the dataset; leakage-safe alternative to target encoding
- **Binning / Bucketing** — Converting continuous columns into groups using `pd.cut` (equal width) and `pd.qcut` (equal frequency) to reduce noise
- **Lag Features** — Creating "N days ago" features from time-series data using `.shift()` to capture past behavior
- **Rolling Window Features** — Computing sliding window statistics (mean, sum, std) using `.rolling()` to capture trends and volatility
- **Polynomial Features** — Generating squared, cubed, and cross-product features using `PolynomialFeatures` to help linear models learn non-linear patterns
- **Log Transformation** — Applying `np.log1p` to skewed columns like price and income to normalize distribution for linear models
- **Ratio Features** — Dividing two related columns (e.g., `price / area`, `income / debt`) to create scale-independent signals
- **Has / Missing Indicator** — Converting null or zero values into 0/1 binary flags to capture missingness as a meaningful signal
- **Text-Based Features** — Extracting character length, word count, and keyword presence from raw text columns without full NLP

---

### 🔜 Coming Soon

- Label Encoding & One Hot Encoding
- ~~Handling Missing Values (NaN)~~ *(Numerical, Categorical, KNN & Iterative imputation strategies fully covered)*
- ~~Outlier Detection & Treatment~~ *(Fully covered in Outlier_Handling_Master_Notebook.ipynb)*
- ~~Binning & Bucketing~~ *(Covered in Feature_Engineering_Hacks.ipynb)*
- Feature Scaling (MinMaxScaler, StandardScaler)
- ~~Creating New Features from Existing Ones~~ *(Covered in Feature_Construction_Reference_Final.ipynb and Feature_Engineering_Hacks.ipynb)*
- Mutual Information & Model-Based Feature Selection

---

## 📊 Statistical Testing for Feature Selection

When working with **categorical → numeric** relationships, two tests are commonly used together:

| Test | Purpose | Output |
|---|---|---|
| **ANOVA (f_oneway)** | Is the difference between groups statistically significant? | F-statistic, p-value |
| **Eta Squared (η²)** | How strong is the effect? | Value between 0 and 1 |

**Eta Squared Interpretation:**

| Value | Meaning |
|---|---|
| < 0.01 | Negligible effect |
| 0.01 – 0.06 | Small effect |
| 0.06 – 0.14 | Medium effect |
| > 0.14 | Large effect |

> ⚠️ In real ML workflows, use ANOVA and Eta Squared for **feature understanding**, not as a replacement for feature importance, mutual information, or model-based selection.

---

## 🔨 Feature Construction – Quick Reference

| Technique | Use When | Example |
|---|---|---|
| **Date Feature Extraction** | Time-based patterns exist | `is_weekend`, `month`, `year` from date column |
| **Count-Based Features** | Multi-value columns (comma-separated) | `num_cast`, `num_genres` from cast/genre columns |
| **Multi-Value Flags** | Reduce complexity of list-type columns | `multi_country` = 1 if more than 1 country |
| **Binary Keyword Flags** | High cardinality categorical columns | `is_drama`, `is_comedy` from listed_in column |
| **Aggregation Features** | Group-level behavior needed | `user_total`, `user_avg` using `groupby + transform` |
| **Interaction Features** | Two features combined give better signal | `age_duration`, `cast_per_duration` |

> ✅ Always ask: **"What signal does this feature give the model?"** — If no clear answer, skip it.

---

## 🔥 Feature Engineering Hacks – Quick Reference

| Hack | Use When | Key Function |
|---|---|---|
| **Target Encoding** | High cardinality, tree-based models | `groupby().transform('mean')` |
| **Frequency Encoding** | Category popularity is a signal | `value_counts(normalize=True)` |
| **Binning (cut)** | Custom range grouping needed | `pd.cut()` |
| **Binning (qcut)** | Equal-frequency groups needed | `pd.qcut()` |
| **Lag Features** | Time-series, past behavior matters | `.shift(n)` |
| **Rolling Window** | Trend or volatility capture | `.rolling(n).mean()` |
| **Polynomial Features** | Linear model on non-linear data | `PolynomialFeatures(degree=2)` |
| **Log Transformation** | Skewed data (price, income) | `np.log1p()` |
| **Ratio Features** | Scale-independent comparison | `col_a / (col_b + 1)` |
| **Has / Missing Indicator** | Missingness is meaningful | `.notna().astype(int)` |
| **Text-Based Features** | Raw text columns present | `.str.len()`, `.str.contains()` |

> ✅ Golden Rule — Always fit encodings on **train data only** and transform test data separately to avoid data leakage.

---

## 🔢 Numerical Imputation – When To Use Which Method?

| Method | Use When | Avoid When |
|---|---|---|
| **Mean Imputation** | Data is normally distributed, few outliers, using linear models | Data is skewed or has strong outliers |
| **Median Imputation** | Data is skewed or has outliers | — |
| **Arbitrary Value (-999 etc.)** | Using tree-based models (Random Forest, XGBoost), missing as signal | Linear regression (causes distortion) |
| **End of Distribution (Mean + 3×Std)** | Missing values are informative, treat missing as extreme | — |
| **Random Sample Imputation** | Need to preserve original distribution and statistical properties | Dataset is very small |
| **KNN Imputer** | Features are correlated, pattern-based fill needed, missing % is low | Very large datasets (computationally slow), unscaled data |
| **Iterative Imputer (MICE)** | Features are strongly correlated, MAR-type missingness, complex relationships | Small datasets, computation-constrained environments |

> ✅ Always fit imputation on **training data only** and apply to test data — avoids data leakage.

---

## 🔤 Categorical Imputation – When To Use Which Method?

| Method | Use When | Risk |
|---|---|---|
| **Most Frequent** | Missing % is small, no business meaning behind missing | Can introduce bias if missing has meaning |
| **Constant ('Missing' category)** | Missing has real meaning, missing % is high | — |
| **Random Imputation** | Experimentation only | Results change every run, hard to reproduce |
| **Missing Indicator** | Missing itself is a useful signal | Use together with another imputation method |

> ✅ If 40%+ values are missing: check business importance, check correlation with target, and evaluate with cross-validation before deciding to drop or impute.

---

## ⚔️ KNN Imputer vs Iterative Imputer

| Feature | KNN Imputer | Iterative Imputer |
|---|---|---|
| **Logic** | Distance-based (Euclidean) | Model-based (BayesianRidge by default) |
| **Accuracy** | Medium | High |
| **Speed** | Slow on large data | Medium |
| **Relationship Capture** | Weak (local neighbors only) | Strong (global patterns via regression) |
| **Scaling Required** | ✅ Yes | Recommended |
| **Custom Estimator** | ❌ No | ✅ Yes (any sklearn regressor) |
| **Complexity** | Medium | High |
| **Best For** | Correlated numeric features, low missing % | MAR missingness, complex feature interactions |

---

## 🚨 Outlier Detection – Quick Reference

| Method | Best For | Threshold |
|---|---|---|
| **IQR Method** | Skewed / non-normal data | < Q1 - 1.5×IQR or > Q3 + 1.5×IQR |
| **Z-Score** | Normally distributed data | \|Z\| > 3 |
| **Isolation Forest** | High-dimensional, complex patterns | contamination parameter |
| **Local Outlier Factor** | Density-based detection | n_neighbors tuning |

## 🛠️ Outlier Treatment Strategies

| Strategy | Use When | Algorithm Compatibility |
|---|---|---|
| **Remove** | Clear data entry errors | Any |
| **Capping / Winsorization** | Valid extreme values, data loss not acceptable | Any |
| **Log Transformation** | Right-skewed data | Linear models benefit most |
| **RobustScaler** | Scaling step in pipeline | Distance/weight-based models |
| **Use Robust Model** | Outliers are genuine and informative | Tree-based: RF, XGBoost |

> ⚠️ Never remove outliers during EDA — always observe and document first. Treat outliers only during the preprocessing phase.

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### Clone the Repo
```bash
git clone https://github.com/sachinmasti/feature-engineering-techniques.git
cd feature-engineering-techniques
```

### Run Notebooks
Open any `.ipynb` file in **Jupyter Notebook** or **VS Code**.

---

## 📊 Sample Datasets Used

**Mixed Data Notebook:**
```python
data = {
    'names': ['john', 'devid', 'jack', 'ruby', 'rahul'],
    'gender': ['male', 'male', 'male', 'female', 'male'],
    'cabin': ['B1-1000', 'z9-1299', 'y6-2304', 'a6-1223', 'g7-2356'],
    'city': ['tokyo', 'helsinki', 'london', 'newyork', 'melbourne'],
    'age': [29, 28, 29, 30, 31]
}
```

**Date-Time Notebook:**
```python
data = {
    'order_id': [1, 2, 3, 4, 5],
    'order_date': ['2023-01-15', '2023-03-22', '2023-07-04', '2023-11-11', '2023-12-25']
}
```

**Feature Construction Notebook:**
```python
# Count-based features
df['num_cast'] = df['cast'].fillna('').str.split(',').apply(len)
df['num_genres'] = df['listed_in'].fillna('').str.split(',').apply(len)

# Binary keyword flags
df['is_drama'] = df['listed_in'].str.contains('Drama', na=False).astype(int)

# Interaction features
df['age_duration'] = df['content_age'] * df['duration']
df['cast_per_duration'] = df['num_cast'] / (df['duration'] + 1)
```

**Feature Engineering Hacks Notebook:**
```python
import numpy as np
import pandas as pd

np.random.seed(42)
df = pd.DataFrame({
    'age':    np.random.randint(18, 70, 200),
    'income': np.random.randint(20000, 200000, 200),
    'price':  np.random.randint(100, 10000, 200),
    'area':   np.random.randint(500, 5000, 200),
    'sales':  np.random.randint(50, 500, 200),
    'city':   np.random.choice(['Mumbai', 'Delhi', 'Pune', 'Hyderabad', 'Bangalore'], 200),
    'target': np.random.randint(0, 2, 200),
    'date':   pd.date_range('2023-01-01', periods=200, freq='D')
})

# Target Encoding
df['city_target_encoded'] = df.groupby('city')['target'].transform('mean')

# Lag Features
df['sales_lag_7day'] = df['sales'].shift(7)

# Log Transformation
df['log_price'] = np.log1p(df['price'])

# Ratio Features
df['price_per_sqft'] = df['price'] / (df['area'] + 1)
```

**Outlier Handling Notebook:**
```python
import numpy as np
import pandas as pd

np.random.seed(42)
data = np.random.normal(50, 10, 100)
data = np.append(data, [150, 170, 200])  # Intentional outliers
df = pd.DataFrame({'feature': data})

# IQR Detection
Q1, Q3 = df['feature'].quantile(0.25), df['feature'].quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

# Capping
df['capped_feature'] = np.clip(df['feature'], lower, upper)

# Log Transform
df['log_feature'] = np.log1p(df['feature'])

# Robust Scaling
from sklearn.preprocessing import RobustScaler
df['robust_scaled'] = RobustScaler().fit_transform(df[['feature']])
```

---

## 🤝 Contributing
Feel free to open a PR if you want to add more techniques or improve existing notebooks!

---

## 📬 Connect
- GitHub: [@sachinmasti](https://github.com/sachinmasti)

---

⭐ **If this repo helped you, please give it a star!**
