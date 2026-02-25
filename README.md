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

---
### 🔜 Coming Soon
- Label Encoding & One Hot Encoding
- ~~Handling Missing Values (NaN)~~ *(Missing data mechanisms covered — imputation strategies coming soon)*
- Outlier Detection & Treatment
- Binning & Bucketing
- Feature Scaling (MinMaxScaler, StandardScaler)
- Creating New Features from Existing Ones
---
## 🚀 Getting Started
### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
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
---
## 🤝 Contributing
Feel free to open a PR if you want to add more techniques or improve existing notebooks!
---
## 📬 Connect
- GitHub: [@sachinmasti](https://github.com/sachinmasti)
---
⭐ **If this repo helped you, please give it a star!**
