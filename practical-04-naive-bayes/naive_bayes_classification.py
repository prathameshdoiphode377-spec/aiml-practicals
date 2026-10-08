"""
Practical 4: Apply Naive Bayes classification algorithms
Part A : Gaussian Naive Bayes on the Wine dataset (continuous features)
Part B : Multinomial Naive Bayes on a small spam / not-spam text sample
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # remove this line if you run in Jupyter / Colab
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# ===============================================================
# PART A: Gaussian Naive Bayes
# ===============================================================
print("=" * 60)
print("PART A: Gaussian Naive Bayes (Wine dataset)")
print("=" * 60)

# Step 1: Load the dataset
wine = load_wine(as_frame=True)
X, y = wine.data, wine.target
print("Shape of dataset:", X.shape)
print("Classes:", [str(c) for c in wine.target_names], "-> counts", np.bincount(y).tolist())
print("Missing values:", int(X.isnull().sum().sum()))
# To use your own file: df = pd.read_csv("your_file.csv"); X = df.drop("target", axis=1); y = df["target"]

# Step 2: Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("Training samples:", len(X_train), "| Testing samples:", len(X_test))

# Step 3: Train the model
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print("\nClass priors P(class):", gnb.class_prior_.round(3).tolist())

# Step 4: Predict and evaluate
y_pred = gnb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy on test set: {acc:.4f}")
print("\nConfusion matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

cv = cross_val_score(GaussianNB(), X, y, cv=5)
print("5-fold cross-validation accuracy:", cv.round(3).tolist())
print("Mean cross-validation accuracy  :", round(cv.mean(), 4))

# Step 5: Plot the confusion matrix
plt.figure(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=wine.target_names, yticklabels=wine.target_names)
plt.title("Confusion Matrix - Gaussian Naive Bayes")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# Step 6: Probabilities for one test sample
sample = X_test.iloc[[0]]
proba = gnb.predict_proba(sample)[0]
print("\nPrediction for one test sample:")
for name, p in zip(wine.target_names, proba):
    print(f"  P({name}) = {p:.4f}")
print("  Predicted class:", wine.target_names[gnb.predict(sample)[0]],
      "| Actual class:", wine.target_names[y_test.iloc[0]])

# ===============================================================
# PART B: Multinomial Naive Bayes (text classification demo)
# ===============================================================
print("\n" + "=" * 60)
print("PART B: Multinomial Naive Bayes (small spam filter demo)")
print("=" * 60)

messages = [
    "win a free prize now", "free cash offer click now", "claim your free reward today",
    "congratulations you won lottery", "urgent offer limited time free", "get cheap loans now",
    "win money now click here", "free entry in prize draw",
    "meeting at 10 am tomorrow", "please send the project report", "lunch with the team today",
    "can we reschedule the class", "assignment submission is due tomorrow", "see you at the lab",
    "thanks for the notes", "call me when you are free",
]
labels = [1] * 8 + [0] * 8                      # 1 = spam, 0 = not spam

vec = CountVectorizer()
X_text = vec.fit_transform(messages)
mnb = MultinomialNB().fit(X_text, labels)

new_msgs = ["free prize click now", "project report due tomorrow", "win cash offer today"]
preds = mnb.predict(vec.transform(new_msgs))
probs = mnb.predict_proba(vec.transform(new_msgs))
for m, p, pr in zip(new_msgs, preds, probs):
    print(f"'{m}' -> {'SPAM' if p == 1 else 'NOT SPAM'} (P(spam) = {pr[1]:.3f})")

# ===============================================================
# Conclusion
# ===============================================================
print(f"\nConclusion: Gaussian Naive Bayes classified the Wine data with "
      f"{acc*100:.2f}% test accuracy (mean 5-fold CV {cv.mean()*100:.2f}%). "
      "Multinomial Naive Bayes correctly separated spam from normal messages in the text demo.")
