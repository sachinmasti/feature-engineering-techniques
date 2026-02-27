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

---

### 🔜 Coming Soon

- Label Encoding & One Hot Encoding
- ~~Handling Missing Values (NaN)~~ *(Numerical, Categorical, KNN & Iterative imputation strategies fully covered)*
- Outlier Detection & Treatment
- Binning & Bucketing
- Feature Scaling (MinMaxScaler, StandardScaler)
- Creating New Features from Existing Ones
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

**Missing Data Mechanisms Notebook:**
```python
df = pd.DataFrame({
    "age": np.random.randint(20, 60, 100),
    "gender": np.random.choice(["Male", "Female"], 100),
    "salary": np.random.randint(20000, 100000, 100)
})
```

**Statistical Testing Notebook:**
```python
# Example: Check if Gender affects Age
male_age   = df[df['Gender'] == 'Male']['Age']
female_age = df[df['Gender'] == 'Female']['Age']
other_age  = df[df['Gender'] == 'Other']['Age']

f_oneway(male_age, female_age, other_age)  # ANOVA
```

**Numerical Imputation Notebook:**
```python
data = {
    "age": [25, 30, 35, np.nan, 45, 50, np.nan, 60, 65, 70],
    "salary": [30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000]
}
df = pd.DataFrame(data)
```

**Categorical Imputation Notebook:**
```python
data = {
    'City': ['Pune', 'Mumbai', np.nan, 'Pune', 'Delhi', np.nan, 'Mumbai'],
    'Loan_Status': ['Approved', 'Rejected', np.nan, 'Approved', 'Rejected', np.nan, 'Approved']
}
df = pd.DataFrame(data)
```

**KNN & Iterative Imputer Notebook:**
```python
data = {
    'age': [25, 30, np.nan, 28],
    'salary': [20000, 25000, 24000, 23000]
}
df = pd.DataFrame(data)

# KNN Imputer
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=2)
df_imputed = imputer.fit_transform(df)

# Iterative Imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imputer_iter = IterativeImputer(max_iter=10, random_state=42)
df_iter_imputed = imputer_iter.fit_transform(df)
```

---

## 🤝 Contributing
Feel free to open a PR if you want to add more techniques or improve existing notebooks!

---

## 📬 Connect
- GitHub: [@sachinmasti](https://github.com/sachinmasti)

---

⭐ **If this repo helped you, please give it a star!**
