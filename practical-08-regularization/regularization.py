"""
Practical 8: Apply regularization to avoid overfitting
Dataset : Diabetes dataset (built into scikit-learn), 442 samples, 10 features
Trick   : Polynomial features of degree 3 expand 10 features into 285 features.
          A plain Linear Regression then OVERFITS (memorises the training data).
Fix     : Ridge (L2), Lasso (L1) and ElasticNet (L1 + L2) regularization.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # remove this line if you run in Jupyter / Colab
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# ---------------------------------------------------------------
# Step 1: Load the dataset
# ---------------------------------------------------------------
data = load_diabetes(as_frame=True)
X, y = data.data, data.target
print("Original shape:", X.shape)
print("Missing values:", int(X.isnull().sum().sum()))
# To use your own file: df = pd.read_csv("your_file.csv"); X = df.drop("target", axis=1); y = df["target"]

# ---------------------------------------------------------------
# Step 2: Split first, then create polynomial features (degree 3)
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=3, include_bias=False)
Xtr_poly = poly.fit_transform(X_train)
Xte_poly = poly.transform(X_test)
print(f"After polynomial expansion: {Xtr_poly.shape[1]} features, {Xtr_poly.shape[0]} training samples")

scaler = StandardScaler().fit(Xtr_poly)
Xtr = scaler.transform(Xtr_poly)
Xte = scaler.transform(Xte_poly)

def scores(model):
    p_tr, p_te = model.predict(Xtr), model.predict(Xte)
    return (r2_score(y_train, p_tr), r2_score(y_test, p_te),
            np.sqrt(mean_squared_error(y_test, p_te)))

# ---------------------------------------------------------------
# Step 3: Show the overfitting problem (plain Linear Regression)
# ---------------------------------------------------------------
lr = LinearRegression().fit(Xtr, y_train)
tr_lr, te_lr, rmse_lr = scores(lr)
print("\nPlain Linear Regression (no regularization):")
print(f"  Train R2 = {tr_lr:.4f} | Test R2 = {te_lr:.4f} | Test RMSE = {rmse_lr:.2f}")
print("  -> Big gap between train and test = overfitting" if tr_lr - te_lr > 0.15
      else "  -> Gap between train and test is small")

# ---------------------------------------------------------------
# Step 4: Choose alpha using cross-validation (training data only)
# ---------------------------------------------------------------
alphas = np.logspace(-2, 4, 40)
ridge_cv = RidgeCV(alphas=alphas, cv=5).fit(Xtr, y_train)
lasso_cv = LassoCV(alphas=np.logspace(0, 1.5, 30), cv=5, max_iter=100000, random_state=42).fit(Xtr, y_train)
print(f"\nBest alpha by cross-validation: Ridge = {ridge_cv.alpha_:.3f} | Lasso = {lasso_cv.alpha_:.3f}")

ridge = Ridge(alpha=ridge_cv.alpha_).fit(Xtr, y_train)
lasso = Lasso(alpha=lasso_cv.alpha_, max_iter=100000).fit(Xtr, y_train)
enet = ElasticNet(alpha=lasso_cv.alpha_, l1_ratio=0.5, max_iter=100000).fit(Xtr, y_train)

# ---------------------------------------------------------------
# Step 5: Compare all models
# ---------------------------------------------------------------
results = []
for name, model in [("Linear Regression", lr), ("Ridge (L2)", ridge),
                    ("Lasso (L1)", lasso), ("ElasticNet", enet)]:
    tr, te, rmse = scores(model)
    nonzero = int(np.sum(np.abs(model.coef_) > 1e-8))
    results.append([name, round(tr, 4), round(te, 4), round(tr - te, 4), round(rmse, 2), nonzero])
res = pd.DataFrame(results, columns=["Model", "Train R2", "Test R2", "Gap", "Test RMSE", "Non-zero coefs"])
print("\nModel comparison:")
print(res.to_string(index=False))

# ---------------------------------------------------------------
# Step 6: Effect of alpha on train/test R2
# ---------------------------------------------------------------
tr_curve, te_curve = [], []
for a in alphas:
    m = Ridge(alpha=a).fit(Xtr, y_train)
    tr_curve.append(m.score(Xtr, y_train)); te_curve.append(m.score(Xte, y_test))

plt.figure(figsize=(7.5, 5))
plt.semilogx(alphas, tr_curve, label="Train R2")
plt.semilogx(alphas, te_curve, label="Test R2")
plt.axvline(ridge_cv.alpha_, color="red", linestyle="--", label=f"Best alpha = {ridge_cv.alpha_:.1f}")
plt.title("Ridge: effect of regularization strength (alpha)")
plt.xlabel("alpha (log scale)  -> more regularization"); plt.ylabel("R2 score")
plt.legend()
plt.tight_layout()
plt.savefig("alpha_effect.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# Step 7: Compare coefficient sizes
# ---------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
for a_, (name, model) in zip(ax, [("Linear Regression", lr), ("Ridge", ridge), ("Lasso", lasso)]):
    a_.bar(range(len(model.coef_)), model.coef_, width=1.0)
    a_.set_title(f"{name} coefficients")
    a_.set_xlabel("Feature index"); a_.set_ylabel("Coefficient")
plt.tight_layout()
plt.savefig("coefficients.png", dpi=150)
plt.close()

print(f"\nLargest |coefficient|: Linear = {np.abs(lr.coef_).max():.1f} | "
      f"Ridge = {np.abs(ridge.coef_).max():.1f} | Lasso = {np.abs(lasso.coef_).max():.1f}")
print(f"Lasso kept {int(np.sum(np.abs(lasso.coef_) > 1e-8))} of {len(lasso.coef_)} features "
      "(the rest were set exactly to zero).")

# ---------------------------------------------------------------
# Conclusion
# ---------------------------------------------------------------
best = res.sort_values("Test R2", ascending=False).iloc[0]
print(f"\nConclusion: Plain Linear Regression overfit (train R2 {tr_lr:.2f}, test R2 {te_lr:.2f}). "
      f"Regularization reduced the train-test gap; the best test R2 was {best['Test R2']:.2f} "
      f"with {best['Model']}.")
