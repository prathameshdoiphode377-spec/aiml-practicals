"""
Practical 2: Apply clustering techniques using unsupervised learning
Dataset   : Iris dataset (built into scikit-learn); the species labels are
            NOT used for training, only for comparison at the end.
Algorithms: K-Means (with elbow method + silhouette score) and
            Agglomerative / Hierarchical clustering (with dendrogram)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # remove this line if you run in Jupyter / Colab
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# ---------------------------------------------------------------
# Step 1: Load the dataset (labels are dropped: unsupervised)
# ---------------------------------------------------------------
iris = load_iris(as_frame=True)
X = iris.data
true_labels = iris.target                     # used only to compare at the end
print("Shape of dataset:", X.shape)
print("\nFirst 5 rows:")
print(X.head())
print("\nMissing values:", int(X.isnull().sum().sum()))
# To use your own file: X = pd.read_csv("your_file.csv")[["col1", "col2", ...]]

# ---------------------------------------------------------------
# Step 2: Scale the features
# ---------------------------------------------------------------
X_scaled = StandardScaler().fit_transform(X)

# ---------------------------------------------------------------
# Step 3: Elbow method + silhouette score to choose K
# ---------------------------------------------------------------
k_range = range(2, 11)
inertia, sil_scores = [], []
for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

print("\nK  Inertia(WCSS)  Silhouette")
for k, i, s in zip(k_range, inertia, sil_scores):
    print(f"{k:<3}{i:<15.2f}{s:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(list(k_range), inertia, "o-")
ax[0].set_title("Elbow Method")
ax[0].set_xlabel("Number of clusters (K)")
ax[0].set_ylabel("Inertia (WCSS)")
ax[1].plot(list(k_range), sil_scores, "o-", color="green")
ax[1].set_title("Silhouette Score vs K")
ax[1].set_xlabel("Number of clusters (K)")
ax[1].set_ylabel("Silhouette score")
plt.tight_layout()
plt.savefig("elbow_silhouette.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# Step 4: Final K-Means model (K = 3, from the elbow bend)
# ---------------------------------------------------------------
K = 3
kmeans = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X_scaled)
km_labels = kmeans.labels_
print(f"\nK-Means with K={K}")
print("Cluster sizes:", np.bincount(km_labels))
print("Silhouette score:", round(silhouette_score(X_scaled, km_labels), 4))

# ---------------------------------------------------------------
# Step 5: Hierarchical (Agglomerative) clustering
# ---------------------------------------------------------------
Z = linkage(X_scaled, method="ward")
plt.figure(figsize=(12, 5))
dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90, leaf_font_size=9)
plt.title("Dendrogram (Ward linkage)")
plt.xlabel("Samples (truncated)")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig("dendrogram.png", dpi=150)
plt.close()

agg = AgglomerativeClustering(n_clusters=K, linkage="ward").fit(X_scaled)
agg_labels = agg.labels_
print(f"\nHierarchical clustering with K={K}")
print("Cluster sizes:", np.bincount(agg_labels))
print("Silhouette score:", round(silhouette_score(X_scaled, agg_labels), 4))

# ---------------------------------------------------------------
# Step 6: Visualise clusters in 2D using PCA
# ---------------------------------------------------------------
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)
centers_2d = pca.transform(kmeans.cluster_centers_)

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].scatter(X_2d[:, 0], X_2d[:, 1], c=km_labels, cmap="viridis", s=30)
ax[0].scatter(centers_2d[:, 0], centers_2d[:, 1], c="red", marker="X", s=200, label="Centroids")
ax[0].set_title("K-Means Clusters (PCA view)")
ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2"); ax[0].legend()
ax[1].scatter(X_2d[:, 0], X_2d[:, 1], c=agg_labels, cmap="viridis", s=30)
ax[1].set_title("Hierarchical Clusters (PCA view)")
ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2")
plt.tight_layout()
plt.savefig("clusters.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# Step 7: Compare with the real species (only for evaluation)
# ---------------------------------------------------------------
print("\nAdjusted Rand Index vs true species (1 = perfect match):")
print("K-Means      :", round(adjusted_rand_score(true_labels, km_labels), 4))
print("Hierarchical :", round(adjusted_rand_score(true_labels, agg_labels), 4))

print("\nK-Means clusters vs true species:")
print(pd.crosstab(km_labels, true_labels,
                  rownames=["Cluster"], colnames=["Species"]))

# ---------------------------------------------------------------
# Step 8: Conclusion
# ---------------------------------------------------------------
print("\nConclusion: The data forms", K, "natural groups. K-Means and "
      "hierarchical clustering give similar clusters, found without using any labels.")
