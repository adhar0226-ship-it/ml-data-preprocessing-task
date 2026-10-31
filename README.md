# Python for Machine Learning & Data Preprocessing

This repository demonstrates essential data preprocessing workflows using Python, Pandas, NumPy, and Scikit-Learn. Preprocessing is a crucial step in machine learning to prepare raw datasets for training predictive models.

## Objectives
1. Understand Machine Learning data processing fundamentals in Python.
2. Load, inspect, and handle missing numerical and categorical values.
3. Encode categorical features and scale numerical attributes.
4. Export the clean dataset ready for machine learning model training.

---

## Preprocessing Pipeline Steps

### 1. Data Inspection & Cleaning
- Identified missing values across numerical (`Age`, `Salary`) and categorical (`Department`) columns.
- Imputed numerical missing values using the **Median** to mitigate outlier impact.
- Imputed categorical missing values using the **Mode** (most frequent value).

### 2. Categorical Variable Encoding
- Binary mapping applied to the target variable (`Purchased`: `No` -> 0, `Yes` -> 1).
- **One-Hot Encoding** (`pd.get_dummies`) applied to `Department` and `Gender` features to avoid multi-collinearity and implicit ordinal bias.

### 3. Feature Scaling & Normalization
- Scaled `Age` and `Salary` columns using **MinMaxScaler**, bringing values into a standardized range of `[0, 1]`.

### 4. Feature Selection
- Dropped irrelevant non-predictive identifiers (`ID`).

---

## Project Structure
```text
.
├── preprocessing_script.py  # Python script containing complete ETL & preprocessing steps
├── cleaned_dataset.csv      # Exported preprocessed output file
└── README.md                # Project documentation
```

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/machine-learning-preprocessing.git
   ```
2. Run the processing script:
   ```bash
   python preprocessing_script.py
   ```
