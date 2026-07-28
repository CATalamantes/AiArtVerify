import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------------------------------------------------------------------
# 1. LOAD PREPROCESSED DATA
# ---------------------------------------------------------------------------
with open("preprocessed.pkl", "rb") as f:
    data = pickle.load(f)

X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
feature_names = data["feature_names"]

print(f"Train: {X_train.shape[0]:,} rows x {X_train.shape[1]} features")
print(f"Test:  {X_test.shape[0]:,} rows x {X_test.shape[1]} features")


def evaluate(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n{name}")
    print(f"  R^2   : {r2:.4f}")
    print(f"  MAE   : {mae:.4f}")
    print(f"  RMSE  : {rmse:.4f}")
    return {"r2": r2, "mae": mae, "rmse": rmse}


# ---------------------------------------------------------------------------
# 2. LINEAR REGRESSION
# ---------------------------------------------------------------------------
print("\n==== LINEAR REGRESSION ====")
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_pred = lin_reg.predict(X_test)
lin_scores = evaluate("Linear Regression (test set)", y_test, lin_pred)

# Coefficients — safe to compare directly against each other because features
# were standardized (mean 0, std 1) before this, so each coefficient reflects
# "effect per one standard deviation of that feature," on a common scale.
coef_df = list(zip(feature_names, lin_reg.coef_))
coef_df.sort(key=lambda x: abs(x[1]), reverse=True)
print("\nTop 15 Linear Regression coefficients (by absolute size):")
for name, coef in coef_df[:15]:
    print(f"  {name:40s} {coef:+.4f}")

# ---------------------------------------------------------------------------
# 3. RANDOM FOREST
# ---------------------------------------------------------------------------
print("\n==== RANDOM FOREST ====")
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,       # let trees grow fully; we'll check for overfitting below
    min_samples_leaf=3,   # light regularization to reduce overfitting on ~thousands of rows
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
rf_pred_test = rf.predict(X_test)
rf_pred_train = rf.predict(X_train)

rf_scores_test = evaluate("Random Forest (test set)", y_test, rf_pred_test)
rf_scores_train = evaluate("Random Forest (train set — overfitting check)", y_train, rf_pred_train)

if rf_scores_train["r2"] - rf_scores_test["r2"] > 0.15:
    print("\n  WARNING: large train/test R^2 gap suggests overfitting — "
          "consider reducing max_depth or increasing min_samples_leaf.")

importances = list(zip(feature_names, rf.feature_importances_))
importances.sort(key=lambda x: x[1], reverse=True)
print("\nTop 15 Random Forest feature importances:")
for name, imp in importances[:15]:
    print(f"  {name:40s} {imp:.4f}")

# ---------------------------------------------------------------------------
# 4. COMPARISON
# ---------------------------------------------------------------------------
print("\n==== COMPARISON (test set) ====")
print(f"{'Metric':10s} {'Linear Reg':>12s} {'Random Forest':>15s}")
print(f"{'R^2':10s} {lin_scores['r2']:>12.4f} {rf_scores_test['r2']:>15.4f}")
print(f"{'MAE':10s} {lin_scores['mae']:>12.4f} {rf_scores_test['mae']:>15.4f}")
print(f"{'RMSE':10s} {lin_scores['rmse']:>12.4f} {rf_scores_test['rmse']:>15.4f}")

# ---------------------------------------------------------------------------
# 5. SAVE TRAINED MODELS FOR THE EXPLAINABILITY PHASE
# ---------------------------------------------------------------------------
with open("trained_models.pkl", "wb") as f:
    pickle.dump(
        {
            "linear_regression": lin_reg,
            "random_forest": rf,
            "feature_names": feature_names,
            "X_test": X_test,
            "y_test": y_test,
        },
        f,
    )
print("\nSaved trained_models.pkl")

