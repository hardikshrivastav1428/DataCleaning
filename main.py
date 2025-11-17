import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


df = pd.read_csv("data/titanic.csv")
print("FIRST FIVE ROWS:")
print(df.head())

print("\nINFO:")
print(df.info())

print("\nNULL VALUES:")
print(df.isnull().sum())

df['Age'] = df['Age'].fillna(df['Age'].median())

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df = df.drop(columns=['Cabin'])

print("\nAFTER HANDLING MISSING VALUES:")
print(df.isnull().sum())

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

print("\nAFTER ENCODING:")
print(df.head())

scaler = StandardScaler()

numeric_cols = ['Age', 'Fare']

df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\nAFTER SCALING:")
print(df[numeric_cols].head())

plt.figure(figsize=(8,6))
sns.boxplot(data=df[['Age', 'Fare']])
plt.title("Boxplot Before Removing Outliers")
plt.show()

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

df_clean = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

print("\nSHAPE BEFORE:", df.shape)
print("SHAPE AFTER REMOVING OUTLIERS:", df_clean.shape)

plt.figure(figsize=(8,6))
sns.boxplot(data=df_clean[['Age', 'Fare']])
plt.title("Boxplot After Removing Outliers")
plt.show()

print("\nCLEANED DATA PREVIEW:")
print(df_clean.head())

df_clean.to_csv("titanic_cleaned.csv", index=False)
print("\nSaved cleaned dataset as titanic_cleaned.csv")