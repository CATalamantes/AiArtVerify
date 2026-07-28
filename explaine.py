import pickle
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. LOAD TRAINED MODELS + TEST DATA
# ---------------------------------------------------------------------------
with open("trained_models.pkl", "rb") as f:
    m = pickle.load(f)

lin_reg = m["linear_regression"]
rf = m["random_forest"]
feature_names = m["feature_names"]
X_test = m["X_test"]
y_test = m["y_test"]

# Try to use SHAP for Random Forest — the principled way to attribute a tree
# ensemble's prediction to individual features. Falls back to a heuristic
# (feature's importance x how far this row's value is from average) if SHAP
# isn't installed, which is an approximation, not a rigorous attribution.
try:
    import shap
    explainer = shap.TreeExplainer(rf)
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("NOTE: shap not installed — using a heuristic fallback for Random "
          "Forest explanations instead of true SHAP attributions. "
          "`pip install shap` for the more rigorous version.\n")


def percentile_context(prediction, distribution):
    pct = (distribution < prediction).mean() * 100
    return pct


def explain_linear(row_values):
    """Exact per-feature contribution: coefficient x standardized feature value."""
    contributions = list(zip(feature_names, lin_reg.coef_ * row_values))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return contributions


def explain_forest_heuristic(row_values):
    """Approximation: importance-weighted distance from average (z-score,
    since features are already standardized to mean 0 / std 1)."""
    importances = rf.feature_importances_
    signal = importances * row_values  # direction + magnitude, weighted by importance
    contributions = list(zip(feature_names, signal))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return contributions


def explain_forest_shap(row_values):
    shap_values = explainer.shap_values(row_values.reshape(1, -1))[0]
    contributions = list(zip(feature_names, shap_values))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return contributions


def describe_prediction(row_index):
    row_values = X_test.iloc[row_index].values
    row_series = X_test.iloc[row_index]

    lin_pred = lin_reg.predict(row_series.values.reshape(1, -1))[0]
    rf_pred = rf.predict(row_series.values.reshape(1, -1))[0]
    actual = y_test.iloc[row_index]

    print(f"\n{'='*60}")
    print(f"Prediction for test row {row_index}")
    print(f"{'='*60}")
    print(f"Actual engagement rate:            {actual:.4f}")
    print(f"Linear Regression prediction:      {lin_pred:.4f} "
          f"(better than {percentile_context(lin_pred, y_test.values):.0f}% of videos)")
    print(f"Random Forest prediction:          {rf_pred:.4f} "
          f"(better than {percentile_context(rf_pred, y_test.values):.0f}% of videos)")

    print("\n--- Why Linear Regression predicted this ---")
    for name, contrib in explain_linear(row_values)[:5]:
        direction = "pushed UP" if contrib > 0 else "pushed DOWN"
        print(f"  {name:35s} {direction:10s} (contribution: {contrib:+.4f})")

    print("\n--- Why Random Forest predicted this ---")
    if HAS_SHAP:
        forest_contribs = explain_forest_shap(row_values)
    else:
        forest_contribs = explain_forest_heuristic(row_values)
    for name, contrib in forest_contribs[:5]:
        direction = "pushed UP" if contrib > 0 else "pushed DOWN"
        print(f"  {name:35s} {direction:10s} (signal: {contrib:+.4f})")


# ---------------------------------------------------------------------------
# 2. DEMO: explain the first few test rows
# ---------------------------------------------------------------------------
for i in range(3):
    describe_prediction(i)