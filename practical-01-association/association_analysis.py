"""
Practical 1: Identify the association between dependent and independent variables
Dataset : Diabetes dataset (built into scikit-learn)
Dependent variable   : 'progression' (disease progression after one year)
Independent variables: age, sex, bmi, bp, s1-s6 (blood serum measurements)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # remove this line if you run in Jupyter / Colab
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.datasets import load_diabetes

pd.set_option("display.max_columns", None)   # show all columns in output
pd.set_option("display.width", 200)

# ---------------------------------------------------------------
# Step 1: Load the dataset
# ---------------------------------------------------------------
data = load_diabetes(as_frame=True)
df = data.frame.rename(columns={"target": "progression"})
# To use your own file instead: df = pd.read_csv("your_file.csv")

print("Shape of dataset:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# ---------------------------------------------------------------
# Step 2: Basic data checks
# ---------------------------------------------------------------
print("\nMissing values per column:")
print(df.isnull().sum())

print("\nSummary statistics:")
print(df.describe().round(3))

# ---------------------------------------------------------------
# Step 3: Define dependent and independent variables
# ---------------------------------------------------------------
dependent = "progression"
independent = [c for c in df.columns if c != dependent]
print("\nDependent variable  :", dependent)
print("Independent variables:", independent)

# ---------------------------------------------------------------
# Step 4: Correlation matrix (Pearson)
# ---------------------------------------------------------------
corr = df.corr(method="pearson")
print("\nCorrelation of each variable with", dependent, ":")
print(corr[dependent].drop(dependent).sort_values(ascending=False).round(3))

# ---------------------------------------------------------------
# Step 5: Heatmap of the correlation matrix
# ---------------------------------------------------------------
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("heatmap.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# Step 6: Scatter plots (each independent variable vs dependent)
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for ax, col in zip(axes.ravel(), independent):
    sns.regplot(x=df[col], y=df[dependent], ax=ax,
                scatter_kws={"s": 12, "alpha": 0.5}, line_kws={"color": "red"})
    ax.set_title(f"{col} vs {dependent}")
plt.tight_layout()
plt.savefig("scatter_plots.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# Step 7: Strength and significance of association
# ---------------------------------------------------------------
def strength(r):
    r = abs(r)
    if r >= 0.7: return "Strong"
    if r >= 0.4: return "Moderate"
    if r >= 0.2: return "Weak"
    return "Very weak / none"

rows = []
for col in independent:
    r, p = pearsonr(df[col], df[dependent])
    rows.append([col, round(r, 3), round(r**2, 3), f"{p:.2e}",
                 "Positive" if r > 0 else "Negative", strength(r),
                 "Yes" if p < 0.05 else "No"])

result = pd.DataFrame(rows, columns=["Variable", "r", "r_squared", "p_value",
                                     "Direction", "Strength", "Significant(p<0.05)"])
result = result.sort_values("r", key=abs, ascending=False).reset_index(drop=True)
print("\nAssociation summary with", dependent, ":")
print(result.to_string(index=False))

# ---------------------------------------------------------------
# Step 8: Conclusion
# ---------------------------------------------------------------
top = result.iloc[0]
print(f"\nConclusion: '{top['Variable']}' has the strongest association with "
      f"{dependent} (r = {top['r']}, {top['Direction'].lower()}, {top['Strength'].lower()}).")
