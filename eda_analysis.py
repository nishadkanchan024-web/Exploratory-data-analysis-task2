import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("student_data.csv")

# Show first 5 rows
print("First 5 rows of dataset")
print(data.head())

# Dataset information
print("\nDataset Info")
print(data.info())

# Statistical summary
print("\nStatistical Summary")
print(data.describe())

# Shape and columns
print("\nDataset Shape:", data.shape)
print("\nColumns:", data.columns)

# Missing values
print("\nMissing Values")
print(data.isnull().sum())

# Fill missing values
data = data.fillna(0)

# Correlation matrix
correlation = data.corr(numeric_only=True)
print("\nCorrelation Matrix")
print(correlation)

# Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Histogram
data.hist(figsize=(10,8))
plt.show()

# Scatter Plot
plt.scatter(data['study_hours'], data['exam_score'])
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Study Hours vs Exam Score")
plt.show()

# Box Plot
sns.boxplot(data=data)
plt.title("Box Plot")
plt.show()
