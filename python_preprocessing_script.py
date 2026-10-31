import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 1. Create a sample dataset for demonstration
raw_data = {
    'ID': [101, 102, 103, 104, 105, 106, 107, 108],
    'Age': [22.0, 38.0, np.nan, 35.0, 54.0, 2.0, 27.0, np.nan],
    'Salary': [72000.0, 48000.0, 54000.0, 61000.0, np.nan, 98000.0, 52000.0, 64000.0],
    'Department': ['Tech', 'HR', 'Tech', np.nan, 'Finance', 'HR', 'Tech', 'Finance'],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female'],
    'Purchased': ['No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No']
}

df = pd.DataFrame(raw_data)
print("=== Initial Raw Dataset ===")
print(df)
print("\n=== Missing Values Summary ===")
print(df.isnull().sum())

# 2. Handling Missing Values
# Impute numerical features with median
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Salary'] = df['Salary'].fillna(df['Salary'].median())

# Impute categorical features with mode
df['Department'] = df['Department'].fillna(df['Department'].mode()[0])

# 3. Categorical Encoding
# Target Encoding (Binary mapping for target column)
df['Purchased'] = df['Purchased'].map({'No': 0, 'Yes': 1})

# One-Hot Encoding for nominal features
df = pd.get_dummies(df, columns=['Department', 'Gender'], drop_first=True, dtype=int)

# 4. Feature Scaling (MinMax Normalization)
scaler = MinMaxScaler()
df[['Age', 'Salary']] = scaler.fit_transform(df[['Age', 'Salary']])

# 5. Feature Selection
# Drop identifiers that carry no predictive power
df = df.drop(columns=['ID'])

print("\n=== Preprocessed Dataset ===")
print(df.head())

# 6. Save Cleaned Dataset
df.to_csv('cleaned_dataset.csv', index=False)
print("\nSuccessfully exported 'cleaned_dataset.csv'.")