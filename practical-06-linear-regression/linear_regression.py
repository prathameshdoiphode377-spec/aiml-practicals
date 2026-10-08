"""
Practical 6: Implement the Linear Regression algorithm
Dataset : Diabetes dataset (built into scikit-learn), 442 samples
Target  : 'progression' (disease progression after one year)
Part A  : Simple Linear Regression  (one feature: bmi)
Part B  : Multiple Linear Regression (all 10 features)
Part C  : Linear Regression implemented from scratch (normal equation) as a check
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # remove this line if you run in Jupyter / Colab
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return mae, mse, np.sqrt(mse), r2_score(y_true, y_pred)

# ---------------------------------------------------------------
# Step 1: Load the dataset
# ---------------------------------------------------------------
data = load_diabetes(as_frame=True)
df = data.frame.rename(columns={"target": "progression"})
print("Shape of dataset:", df.shape)
print("Missing values:", int(df.isnull().sum().sum()))
print("\nFirst 5 rows:")
print(df.head())
# To use your own file: df = pd.read_csv("your_file.csv") and set y = df["your_target"]

X_all = df.drop(columns="progression")
y = df["progression"]

# ---------------------------------------------------------------
# Step 2: Split into training (80%) and testing (20%) sets
# ---------------------------------------------------------------
X_train_all, X_test_all, y_train, y_test = train_test_split(
    X_all, y, test_size=0.2, random_state=42)
print("\nTraining samples:", len(X_train_all), "| Testing samples:", len(X_test_all))

# ===============================================================
# PART A: Simple Linear Regression (bmi -> progression)
# ===============================================================
print("\n" + "=" * 60)
print("PART A: Simple Linear Regression (bmi -> progression)")
print("=" * 60)

X_train_s, X_test_s = X_train_all[["bmi"]], X_test_all[["bmi"]]
simple = LinearRegression().fit(X_train_s, y_train)
print(f"Slope (coefficient): {simple.coef_[0]:.2f}")
print(f"Intercept          : {simple.intercept_:.2f}")
print(f"Equation: progression = {simple.intercept_:.2f} + {simple.coef_[0]:.2f} * bmi")

pred_s = simple.predict(X_test_s)
mae, mse, rmse, r2 = evaluate(y_test, pred_s)
print(f"\nTest MAE : {mae:.2f}\nTest MSE : {mse:.2f}\nTest RMSE: {rmse:.2f}\nTest R2  : {r2:.4f}")

plt.figure(figsize=(7, 5))
plt.scatter(X_test_s, y_test, alpha=0.6, label="Actual (test data)")
xs = np.linspace(X_all["bmi"].min(), X_all["bmi"].max(), 100).reshape(-1, 1)
plt.plot(xs, simple.predict(pd.DataFrame(xs, columns=["bmi"])), "r-", linewidth=2, label="Regression line")
plt.title("Simple Linear Regression: BMI vs Disease Progression")
plt.xlabel("BMI (standardized)"); plt.ylabel("Disease progression")
plt.legend()
plt.tight_layout()
plt.savefig("simple_regression_line.png", dpi=150)
plt.close()

# ===============================================================
# PART B: Multiple Linear Regression (all features)
# ===============================================================
print("\n" + "=" * 60)
print("PART B: Multiple Linear Regression (all 10 features)")
print("=" * 60)

multi = LinearRegression().fit(X_train_all, y_train)
coef = pd.Series(multi.coef_, index=X_all.columns).sort_values(key=abs, ascending=False)
print("Coefficients (largest effect first):")
print(coef.round(2))
print(f"Intercept: {multi.intercept_:.2f}")

pred_m = multi.predict(X_test_all)
mae_m, mse_m, rmse_m, r2_m = evaluate(y_test, pred_m)
print(f"\nTest MAE : {mae_m:.2f}\nTest MSE : {mse_m:.2f}\nTest RMSE: {rmse_m:.2f}\nTest R2  : {r2_m:.4f}")
print(f"Train R2 : {multi.score(X_train_all, y_train):.4f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(y_test, pred_m, alpha=0.6)
lims = [min(y_test.min(), pred_m.min()), max(y_test.max(), pred_m.max())]
ax[0].plot(lims, lims, "r--", label="Perfect prediction")
ax[0].set_title("Actual vs Predicted (Multiple Regression)")
ax[0].set_xlabel("Actual"); ax[0].set_ylabel("Predicted"); ax[0].legend()
residuals = y_test - pred_m
ax[1].scatter(pred_m, residuals, alpha=0.6)
ax[1].axhline(0, color="red", linestyle="--")
ax[1].set_title("Residual Plot")
ax[1].set_xlabel("Predicted"); ax[1].set_ylabel("Residual (actual - predicted)")
plt.tight_layout()
plt.savefig("multiple_regression_plots.png", dpi=150)
plt.close()

# ===============================================================
# PART C: From scratch using the normal equation  w = (XᵀX)⁻¹ Xᵀy
# ===============================================================
print("\n" + "=" * 60)
print("PART C: Linear Regression from scratch (normal equation)")
print("=" * 60)

Xb = np.c_[np.ones(len(X_train_s)), X_train_s.values]       # add a column of 1s for the intercept
w = np.linalg.inv(Xb.T @ Xb) @ Xb.T @ y_train.values
print(f"Intercept (scratch): {w[0]:.2f} | Slope (scratch): {w[1]:.2f}")
print(f"Intercept (sklearn): {simple.intercept_:.2f} | Slope (sklearn): {simple.coef_[0]:.2f}")
print("Both methods give the same line:", bool(np.allclose(w, [simple.intercept_, simple.coef_[0]])))

# ===============================================================
# Comparison and conclusion
# ===============================================================
print("\nModel comparison on the test set:")
print(pd.DataFrame({"Model": ["Simple (bmi only)", "Multiple (10 features)"],
                    "RMSE": [round(rmse, 2), round(rmse_m, 2)],
                    "R2": [round(r2, 4), round(r2_m, 4)]}).to_string(index=False))

print(f"\nConclusion: The multiple regression model (R2 = {r2_m:.2f}) explains more of the "
      f"variation than the simple model using BMI alone (R2 = {r2:.2f}), but about half of "
      "the variation is still unexplained, so a linear model has limits on this data.")
