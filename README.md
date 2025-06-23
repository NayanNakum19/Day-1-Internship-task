# Day-1-Internship-task
Clean and preprocess automobile data for ML: handle missing values, convert data types, encode categories, normalize features, remove outliers, and split into train/test sets. Includes CSV dataset and Jupyter Notebook with all preprocessing steps.


# 📦 Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ✅ 1. Load Dataset & Replace '?' with NaN
df = pd.read_csv("Automobile_data.csv")
df.replace('?', np.nan, inplace=True)

# ✅ 2. Convert Data Types
numeric_columns = ['normalized-losses', 'bore', 'stroke', 'horsepower', 'peak-rpm', 'price']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ✅ Handle Missing Values
df['normalized-losses'].fillna(df['normalized-losses'].mean(), inplace=True)
df['bore'].fillna(df['bore'].mean(), inplace=True)
df['stroke'].fillna(df['stroke'].mean(), inplace=True)
df['horsepower'].fillna(df['horsepower'].mean(), inplace=True)
df['peak-rpm'].fillna(df['peak-rpm'].mean(), inplace=True)
df['price'].fillna(df['price'].median(), inplace=True)
df['num-of-doors'].fillna(df['num-of-doors'].mode()[0], inplace=True)

# ✅ 3. Encode Categorical Columns
# Label Encoding binary fields
le = LabelEncoder()
df['fuel-type'] = le.fit_transform(df['fuel-type'])
df['aspiration'] = le.fit_transform(df['aspiration'])
df['num-of-doors'] = le.fit_transform(df['num-of-doors'])

# One-hot Encoding multiclass fields
df = pd.get_dummies(df, columns=[
    'make', 'body-style', 'drive-wheels',
    'engine-location', 'engine-type', 'fuel-system'
], drop_first=True)

# Convert 'num-of-cylinders' from text to number
cylinder_map = {
    'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'eight': 8,
    'twelve': 12
}
df['num-of-cylinders'] = df['num-of-cylinders'].map(cylinder_map)

# ✅ 4. Standardize Numeric Features
scaler = StandardScaler()
numeric_to_scale = ['normalized-losses', 'bore', 'stroke', 'horsepower', 'peak-rpm', 'engine-size', 'price']
df[numeric_to_scale] = scaler.fit_transform(df[numeric_to_scale])

# ✅ 5. Visualize and Remove Outliers using IQR
for col in numeric_to_scale:
    # Visualize
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

    # Remove outliers
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

# ✅ Dataset is now clean and ready for ML
print("✅ Final cleaned dataset shape:", df.shape)
