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

---

## 🧪 Techniques Covered

### ✅ Currently Added
- **Mixed Data Handling** — Extracting numbers/letters from columns like `B1-1000`, `z9-1299` using regex and string methods
- **str.extract** — Extract specific patterns from string columns
- **str.split** — Split a column into multiple columns by a delimiter
- **pd.to_numeric** — Convert string columns to numeric with error handling

---

### 🔜 Coming Soon
- Label Encoding & One Hot Encoding
- Handling Missing Values (NaN)
- Outlier Detection & Treatment
- Date/Time Feature Extraction
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

## 📊 Sample Dataset Used

```python
data = {
    'names': ['john', 'devid', 'jack', 'ruby', 'rahul'],
    'gender': ['male', 'male', 'male', 'female', 'male'],
    'cabin': ['B1-1000', 'z9-1299', 'y6-2304', 'a6-1223', 'g7-2356'],
    'city': ['tokyo', 'helsinki', 'london', 'newyork', 'melbourne'],
    'age': [29, 28, 29, 30, 31]
}
```

---

## 🤝 Contributing

Feel free to open a PR if you want to add more techniques or improve existing notebooks!

---

## 📬 Connect

- GitHub: [@sachinmasti](https://github.com/sachinmasti)

---

⭐ **If this repo helped you, please give it a star!**
