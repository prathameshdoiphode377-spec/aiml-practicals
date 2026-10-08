"""
Practical 5: Apply the KNN algorithm for classification
Dataset : Iris dataset (built into scikit-learn)
          150 samples, 4 features, 3 classes (setosa, versicolor, virginica)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # remove this line if you run in Jupyter / Colab
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# ---------------------------------------------------------------
# Step 1: Load the dataset
# ---------------------------------------------------------------
iris = load_iris(as_frame=True)
X, y = iris.data, iris.target
class_names = [str(c) for c in iris.target_names]
print("Shape of dataset:", X.shape)
print("Classes:", class_names, "-> counts", np.bincount(y).tolist())
print("Missing values:", int(X.isnull().sum().sum()))
# To use your own file: df = pd.read_csv("your_file.csv"); X = df.drop("target", axis=1); y = df["target"]

# ---------------------------------------------------------------
# Step 2: Split into training (80%) and testing (20%) sets
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("Training samples:", len(X_train), "| Testing samples:", len(X_test))

# ---------------------------------------------------------------
# Step 3: Scale the features (KNN is distance based)
# ---------------------------------------------------------------
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# ---------------------------------------------------------------
# Step 4: Find the best K using 5-fold cross-validation on the TRAINING data
# (the test set is not used to choose K, so the final test result stays honest)
# Odd values of K are used to avoid ties in voting.
# ---------------------------------------------------------------
k_values = list(range(1, 22, 2))
cv_acc, train_acc, test_acc = [], [], []
for k in k_values:
    m = KNeighborsClassifier(n_neighbors=k)
    cv_acc.append(cross_val_score(m, X_train_s, y_train, cv=5).mean())
    m.fit(X_train_s, y_train)
    train_acc.append(m.score(X_train_s, y_train))
    test_acc.append(m.score(X_test_s, y_test))

print("\nK   CV accuracy   Train accuracy   Test accuracy")
for k, cv_, tr, te in zip(k_values, cv_acc, train_acc, test_acc):
    print(f"{k:<4}{cv_:<14.4f}{tr:<17.4f}{te:.4f}")

best_k = k_values[int(np.argmax(cv_acc))]   # K with the highest cross-validation accuracy
print(f"\nBest K (highest cross-validation accuracy): {best_k}")

plt.figure(figsize=(7, 4.5))
plt.plot(k_values, cv_acc, "o-", label="Cross-validation accuracy")
plt.plot(k_values, train_acc, "^--", label="Train accuracy", alpha=0.7)
plt.plot(k_values, test_acc, "s--", label="Test accuracy", alpha=0.7)
plt.axvline(best_k, color="red", linestyle="--", label=f"Best K = {best_k}")
plt.xticks(k_values)
plt.title("Accuracy vs K")
plt.xlabel("Number of neighbours (K)")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_vs_k.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# Step 5: Train the final KNN model and evaluate
# ---------------------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_s, y_train)
y_pred = knn.predict(X_test_s)

acc = accuracy_score(y_test, y_pred)
print(f"\nFinal model: KNN with K = {best_k}")
print(f"Accuracy on test set: {acc:.4f}")
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion matrix:")
print(cm)
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=class_names))

plt.figure(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title(f"Confusion Matrix - KNN (K={best_k})")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# Step 6: Predict a new, unseen sample
# ---------------------------------------------------------------
new_flower = pd.DataFrame([[5.9, 3.0, 5.1, 1.8]], columns=X.columns)
new_scaled = scaler.transform(new_flower)
pred = knn.predict(new_scaled)[0]
proba = knn.predict_proba(new_scaled)[0]
print("\nNew sample (sepal length, sepal width, petal length, petal width) = [5.9, 3.0, 5.1, 1.8]")
print("Predicted species:", class_names[pred])
for name, p in zip(class_names, proba):
    print(f"  P({name}) = {p:.3f}")

# ---------------------------------------------------------------
# Step 7: Decision boundary using two features (petal length & width)
# ---------------------------------------------------------------
X2 = X[["petal length (cm)", "petal width (cm)"]].values
sc2 = StandardScaler().fit(X2)
knn2 = KNeighborsClassifier(n_neighbors=best_k).fit(sc2.transform(X2), y)

xx, yy = np.meshgrid(np.linspace(X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5, 300),
                     np.linspace(X2[:, 1].min() - 0.3, X2[:, 1].max() + 0.3, 300))
Z = knn2.predict(sc2.transform(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)

plt.figure(figsize=(7, 5))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"]))
for c, name, color in zip(range(3), class_names, ["red", "green", "blue"]):
    plt.scatter(X2[y == c, 0], X2[y == c, 1], c=color, s=25, edgecolor="k", label=name)
plt.title(f"KNN Decision Boundary (K={best_k}, petal features)")
plt.xlabel("Petal length (cm)"); plt.ylabel("Petal width (cm)")
plt.legend()
plt.tight_layout()
plt.savefig("decision_boundary.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# Step 8: Conclusion
# ---------------------------------------------------------------
print(f"\nConclusion: KNN with K = {best_k} classified the Iris test data with "
      f"{acc*100:.2f}% accuracy.")
