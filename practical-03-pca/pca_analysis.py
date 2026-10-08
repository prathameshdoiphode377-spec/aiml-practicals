"""
Practical 3: Apply dimensionality reduction using PCA
Dataset : Breast Cancer Wisconsin dataset (built into scikit-learn)
          569 samples, 30 numeric features, 2 classes (malignant / benign)
Goal    : Reduce 30 features to a few principal components while keeping
          most of the information, and check that a classifier still works.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # remove this line if you run in Jupyter / Colab
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# ---------------------------------------------------------------
# Step 1: Load the dataset
# ---------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target
print("Shape of dataset:", X.shape)
print("Classes: malignant =", int((y == 0).sum()), ", benign =", int((y == 1).sum()))
print("Missing values:", int(X.isnull().sum().sum()))
# To use your own file: df = pd.read_csv("your_file.csv"); X = df.drop("target", axis=1); y = df["target"]

# ---------------------------------------------------------------
# Step 2: Standardize the features (mean = 0, std = 1)
# ---------------------------------------------------------------
X_scaled = StandardScaler().fit_transform(X)

# ---------------------------------------------------------------
# Step 3: Fit PCA with all components and study the variance
# ---------------------------------------------------------------
pca_full = PCA().fit(X_scaled)
var_ratio = pca_full.explained_variance_ratio_
cum_var = np.cumsum(var_ratio)

print("\nExplained variance of first 10 components:")
for i in range(10):
    print(f"PC{i+1:<3} variance = {var_ratio[i]*100:6.2f}%   cumulative = {cum_var[i]*100:6.2f}%")

n_95 = int(np.argmax(cum_var >= 0.95) + 1)
n_90 = int(np.argmax(cum_var >= 0.90) + 1)
print(f"\nComponents needed for 90% variance: {n_90}")
print(f"Components needed for 95% variance: {n_95}")

# ---------------------------------------------------------------
# Step 4: Scree plot and cumulative variance plot
# ---------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].bar(range(1, len(var_ratio) + 1), var_ratio * 100)
ax[0].set_title("Scree Plot")
ax[0].set_xlabel("Principal component")
ax[0].set_ylabel("Explained variance (%)")
ax[1].plot(range(1, len(cum_var) + 1), cum_var * 100, "o-")
ax[1].axhline(95, color="red", linestyle="--", label="95% threshold")
ax[1].axvline(n_95, color="green", linestyle="--", label=f"{n_95} components")
ax[1].set_title("Cumulative Explained Variance")
ax[1].set_xlabel("Number of components")
ax[1].set_ylabel("Cumulative variance (%)")
ax[1].legend()
plt.tight_layout()
plt.savefig("variance_plots.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# Step 5: Reduce to 2 components and visualise
# ---------------------------------------------------------------
pca2 = PCA(n_components=2)
X_2d = pca2.fit_transform(X_scaled)
print("\nVariance kept by 2 components:", round(pca2.explained_variance_ratio_.sum() * 100, 2), "%")

plt.figure(figsize=(7, 5.5))
for label, name, color in [(0, "malignant", "red"), (1, "benign", "blue")]:
    plt.scatter(X_2d[y == label, 0], X_2d[y == label, 1], s=18, alpha=0.6, c=color, label=name)
plt.title("Breast Cancer data reduced to 2 principal components")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend()
plt.tight_layout()
plt.savefig("pca_2d_scatter.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# Step 6: Which original features matter most for PC1 and PC2?
# ---------------------------------------------------------------
loadings = pd.DataFrame(pca2.components_.T, index=X.columns, columns=["PC1", "PC2"])
print("\nTop 5 features contributing to PC1:")
print(loadings["PC1"].abs().sort_values(ascending=False).head(5).round(3))

# ---------------------------------------------------------------
# Step 7: Compare classifier before and after PCA
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
acc_all = accuracy_score(y_test, clf.predict(X_test))

pca_k = PCA(n_components=n_95).fit(X_train)
clf_pca = LogisticRegression(max_iter=1000).fit(pca_k.transform(X_train), y_train)
acc_pca = accuracy_score(y_test, clf_pca.predict(pca_k.transform(X_test)))

print("\nLogistic Regression accuracy:")
print(f"Using all {X.shape[1]} features       : {acc_all:.4f}")
print(f"Using {n_95} PCA components      : {acc_pca:.4f}")

# ---------------------------------------------------------------
# Step 8: Conclusion
# ---------------------------------------------------------------
print(f"\nConclusion: {n_95} principal components keep 95% of the information "
      f"of all {X.shape[1]} features, reducing the data by "
      f"{(1 - n_95 / X.shape[1]) * 100:.0f}% with only a small drop in accuracy "
      f"({acc_all*100:.2f}% to {acc_pca*100:.2f}%).")
