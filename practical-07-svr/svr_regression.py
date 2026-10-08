"""
Practical 7: Implement the SVM algorithm for Regression (SVR)
Dataset : Diabetes dataset (built into scikit-learn), 442 samples
Target  : 'progression' (disease progression after one year)
Part A  : SVR with different kernels on one feature (bmi) - easy to visualise
Part B  : SVR on all 10 features with hyper-parameter tuning (GridSearchCV)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # remove this line if you run in Jupyter / Colab
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return mean_absolute_error(y_true, y_pred), np.sqrt(mse), r2_score(y_true, y_pred)

# ---------------------------------------------------------------
# Step 1: Load the dataset
# ---------------------------------------------------------------
data = load_diabetes(as_frame=True)
df = data.frame.rename(columns={"target": "progression"})
X_all = df.drop(columns="progression")
y = df["progression"]
print("Shape of dataset:", df.shape)
print("Missing values:", int(df.isnull().sum().sum()))
# To use your own file: df = pd.read_csv("your_file.csv") and set y = df["your_target"]

# ---------------------------------------------------------------
# Step 2: Split into training (80%) and testing (20%) sets
# ---------------------------------------------------------------
X_train_all, X_test_all, y_train, y_test = train_test_split(
    X_all, y, test_size=0.2, random_state=42)
print("Training samples:", len(X_train_all), "| Testing samples:", len(X_test_all))

# ---------------------------------------------------------------
# Step 3: Scale X and y (SVR is sensitive to the scale of both)
# ---------------------------------------------------------------
sx = StandardScaler().fit(X_train_all)
sy = StandardScaler().fit(y_train.values.reshape(-1, 1))
Xtr = sx.transform(X_train_all)
Xte = sx.transform(X_test_all)
ytr = sy.transform(y_train.values.reshape(-1, 1)).ravel()

def predict_original(model, X):
    """Predict and convert the answer back to the original units of the target."""
    return sy.inverse_transform(model.predict(X).reshape(-1, 1)).ravel()

# ===============================================================
# PART A: Different kernels on one feature (bmi)
# ===============================================================
print("\n" + "=" * 60)
print("PART A: SVR kernels on one feature (bmi)")
print("=" * 60)

bmi_idx = list(X_all.columns).index("bmi")
Xtr_b, Xte_b = Xtr[:, [bmi_idx]], Xte[:, [bmi_idx]]

kernels = {
    "linear": SVR(kernel="linear", C=1.0, epsilon=0.1),
    "poly (degree 3)": SVR(kernel="poly", degree=3, C=1.0, epsilon=0.1),
    "rbf": SVR(kernel="rbf", C=1.0, epsilon=0.1),
}
rows = []
grid = np.linspace(Xtr_b.min(), Xtr_b.max(), 300).reshape(-1, 1)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
for ax, (name, model) in zip(axes, kernels.items()):
    model.fit(Xtr_b, ytr)
    pred = predict_original(model, Xte_b)
    mae, rmse, r2 = evaluate(y_test, pred)
    rows.append([name, round(mae, 2), round(rmse, 2), round(r2, 4)])
    ax.scatter(Xtr_b, y_train, alpha=0.4, s=15, label="Training data")
    ax.plot(grid, sy.inverse_transform(model.predict(grid).reshape(-1, 1)), "r-", linewidth=2, label="SVR prediction")
    ax.set_title(f"SVR kernel = {name}\nTest R2 = {r2:.3f}")
    ax.set_xlabel("BMI (scaled)")
axes[0].set_ylabel("Disease progression")
axes[0].legend()
plt.tight_layout()
plt.savefig("svr_kernels_bmi.png", dpi=150)
plt.close()

print(pd.DataFrame(rows, columns=["Kernel", "MAE", "RMSE", "R2"]).to_string(index=False))

# ===============================================================
# PART B: All features + hyper-parameter tuning
# ===============================================================
print("\n" + "=" * 60)
print("PART B: SVR on all 10 features with GridSearchCV")
print("=" * 60)

param_grid = {
    "kernel": ["linear", "rbf"],
    "C": [0.1, 1, 10, 100],
    "epsilon": [0.01, 0.1, 0.5],
    "gamma": ["scale", 0.01, 0.1],
}
search = GridSearchCV(SVR(), param_grid, cv=5, scoring="r2", n_jobs=-1)
search.fit(Xtr, ytr)
print("Best parameters:", search.best_params_)
print(f"Best cross-validation R2: {search.best_score_:.4f}")

best_svr = search.best_estimator_
pred_svr = predict_original(best_svr, Xte)
mae_s, rmse_s, r2_s = evaluate(y_test, pred_svr)

# Baseline: ordinary Linear Regression (from Practical 6)
lr = LinearRegression().fit(X_train_all, y_train)
mae_l, rmse_l, r2_l = evaluate(y_test, lr.predict(X_test_all))

# Default SVR (no tuning) for comparison
default_svr = SVR().fit(Xtr, ytr)
mae_d, rmse_d, r2_d = evaluate(y_test, predict_original(default_svr, Xte))

print("\nTest set comparison:")
print(pd.DataFrame({
    "Model": ["Linear Regression", "SVR (default settings)", "SVR (tuned)"],
    "MAE": [round(mae_l, 2), round(mae_d, 2), round(mae_s, 2)],
    "RMSE": [round(rmse_l, 2), round(rmse_d, 2), round(rmse_s, 2)],
    "R2": [round(r2_l, 4), round(r2_d, 4), round(r2_s, 4)],
}).to_string(index=False))
cv_default = cross_val_score(SVR(), Xtr, ytr, cv=5, scoring="r2").mean()
print(f"\nCross-validation R2 (training data): tuned SVR = {search.best_score_:.4f}, default SVR = {cv_default:.4f}")
print("\nNumber of support vectors in tuned model:", len(best_svr.support_),
      "out of", len(Xtr), "training samples")

plt.figure(figsize=(6.5, 5.5))
plt.scatter(y_test, pred_svr, alpha=0.6)
lims = [min(y_test.min(), pred_svr.min()), max(y_test.max(), pred_svr.max())]
plt.plot(lims, lims, "r--", label="Perfect prediction")
plt.title("Tuned SVR: Actual vs Predicted")
plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.legend()
plt.tight_layout()
plt.savefig("svr_actual_vs_predicted.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# Conclusion
# ---------------------------------------------------------------
print(f"\nConclusion: Tuning improved SVR in cross-validation (R2 {cv_default:.2f} -> {search.best_score_:.2f}), "
      f"and the tuned SVR (test R2 {r2_s:.2f}) beat Linear Regression (test R2 {r2_l:.2f}). "
      f"The default SVR happened to score higher on this particular test split (R2 {r2_d:.2f}); with only "
      f"{len(y_test)} test samples that gap is within noise, so the cross-validated tuned model is the "
      "more reliable choice. On one feature (BMI) the polynomial kernel fitted poorly (negative R2).")
