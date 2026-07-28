"""Step 2 of the v2 pipeline: train every model, run both gates, save metrics.

Run this BEFORE touching the UI. Gate A decides whether the app asks for channel
stats at all; Gate B decides whether the advice panel ships a model or a plain
data summary. Building the dashboard first would mean building it twice.

Outputs
    artifacts/models.pkl    fitted pipelines the app loads
    artifacts/metrics.json  honest, cross-validated numbers + gate verdicts
"""

import gzip
import json
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

import config as C
import pipeline as P

warnings.filterwarnings("ignore", category=UserWarning)

# Chosen by a size-vs-quality sweep (see README). v1's 300/None/3 scored
# AUC 0.8591 at 22 MB per forest; this scores 0.8469 at 5.0 MB. Trading 0.012
# AUC for a 4.4x smaller artifact is the right call for a free-tier deploy that
# re-clones the repo on every push.
RF_CLF = dict(
    n_estimators=150,
    max_depth=12,  # v1 used max_depth=None, which is what produced a 41 MB pickle
    min_samples_leaf=10,
    random_state=C.RANDOM_STATE,
    n_jobs=-1,
)
RF_REG = dict(RF_CLF)

print("Loading features_v2.csv ...")
df = pd.read_csv(C.FEATURES_V2)
print(f"  {len(df):,} rows")

y_hit = df[C.TARGET_HIT]
skf = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
kf = KFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)

metrics = {"n_rows": int(len(df)), "base_rate": float(y_hit.mean()), "cv_folds": C.CV_FOLDS}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def cv_classifier(name, feature_list):
    """Cross-validated out-of-fold probabilities -> AUC, Brier, accuracy.

    No CalibratedClassifierCV: a sweep showed isotonic moved Brier only
    0.1340 -> 0.1320 while *lowering* AUC and costing 2.2x the artifact size.
    The raw forest probabilities are already about as calibrated as they get.
    """
    pipe = P.make_pipeline(feature_list, RandomForestClassifier(**RF_CLF))
    proba = cross_val_predict(pipe, df, y_hit, cv=skf, method="predict_proba", n_jobs=1)[:, 1]

    auc = roc_auc_score(y_hit, proba)
    brier = brier_score_loss(y_hit, proba)
    acc = ((proba >= 0.5).astype(int) == y_hit).mean()
    tn, fp, fn, tp = confusion_matrix(y_hit, (proba >= 0.5).astype(int)).ravel()

    print(f"  {name:24s} AUC={auc:.4f}  Brier={brier:.4f}  acc={acc:.3f}  n_feat={len(feature_list)}")
    return {
        "auc": float(auc),
        "brier": float(brier),
        "accuracy": float(acc),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_features": len(feature_list),
    }, proba


def cv_regressor(name, target_col, feature_list, model, real_scale_col=None):
    """CV a regressor; for log targets also report error on the real scale."""
    y = df[target_col]
    pipe = P.make_pipeline(feature_list, model)
    pred = cross_val_predict(pipe, df, y, cv=kf, n_jobs=1)

    out = {
        "r2": float(r2_score(y, pred)),
        "mae": float(mean_absolute_error(y, pred)),
        "n_features": len(feature_list),
    }
    line = f"  {name:24s} R2={out['r2']:+.4f}  MAE={out['mae']:.4f}"

    if real_scale_col is not None:
        # Log-scale MAE means nothing to a user. "Typically within 2.3x of
        # actual" does.
        actual = df[real_scale_col].clip(lower=1)
        predicted = np.expm1(pred).clip(min=1)
        ratio = np.maximum(predicted / actual, actual / predicted)
        out["median_ratio_error"] = float(np.median(ratio))
        line += f"  typical error={out['median_ratio_error']:.2f}x"

    print(line)
    return out


def baseline_regressor(target_col):
    y = df[target_col]
    pred = cross_val_predict(DummyRegressor(strategy="mean"), df[P.SERVE_FEATURES], y, cv=kf)
    return {"r2": float(r2_score(y, pred)), "mae": float(mean_absolute_error(y, pred))}


# ---------------------------------------------------------------------------
# GATE A -- do channel stats earn their place?
# ---------------------------------------------------------------------------
print("\n==== GATE A: channel-stat ablation ====")
full_clf, full_proba = cv_classifier("hit | full", P.SERVE_FEATURES)
nochan_clf, _ = cv_classifier("hit | no channel", P.NO_CHANNEL)

auc_gain = full_clf["auc"] - nochan_clf["auc"]
keep_channel = auc_gain >= C.GATE_A_MIN_AUC_GAIN
print(f"\n  AUC gain from channel features: {auc_gain:+.4f} (threshold {C.GATE_A_MIN_AUC_GAIN})")
print(f"  VERDICT: {'KEEP channel inputs' if keep_channel else 'DROP channel inputs'}")

metrics["gate_a"] = {
    "auc_full": full_clf["auc"],
    "auc_without_channel": nochan_clf["auc"],
    "auc_gain": float(auc_gain),
    "threshold": C.GATE_A_MIN_AUC_GAIN,
    "keep_channel_features": bool(keep_channel),
}

# Whichever arm won becomes the headline model's feature set.
HEADLINE_FEATURES = P.SERVE_FEATURES if keep_channel else P.NO_CHANNEL

# ---------------------------------------------------------------------------
# GATE B -- is there real signal in content alone?
# ---------------------------------------------------------------------------
print("\n==== GATE B: content-only signal ====")
content_clf, _ = cv_classifier("hit | content only", P.CONTENT_ONLY)
advice_is_real = content_clf["auc"] >= C.GATE_B_MIN_AUC
print(f"\n  content-only AUC {content_clf['auc']:.4f} vs threshold {C.GATE_B_MIN_AUC}")
print(
    f"  VERDICT: {'model-based advice is meaningful' if advice_is_real else 'FALL BACK to empirical benchmarks only'}"
)

metrics["gate_b"] = {
    "auc_content_only": content_clf["auc"],
    "threshold": C.GATE_B_MIN_AUC,
    "model_advice_is_meaningful": bool(advice_is_real),
}

dummy_auc = roc_auc_score(
    y_hit,
    cross_val_predict(
        DummyClassifier(strategy="prior"), df[P.SERVE_FEATURES], y_hit, cv=skf, method="predict_proba"
    )[:, 1],
)
metrics["classifier"] = {
    "headline": full_clf if keep_channel else nochan_clf,
    "content_only": content_clf,
    "baseline_auc": float(dummy_auc),
}

# ---------------------------------------------------------------------------
# REGRESSORS -- the concrete numbers
# ---------------------------------------------------------------------------
print("\n==== REGRESSORS (cross-validated) ====")
metrics["regressors"] = {}
for key, target_col in C.REGRESSION_TARGETS.items():
    metrics["regressors"][key] = cv_regressor(
        f"log {key}",
        target_col,
        HEADLINE_FEATURES,
        RandomForestRegressor(**RF_REG),
        real_scale_col=f"_target_{key}",
    )
    metrics["regressors"][key]["baseline"] = baseline_regressor(target_col)

# Engagement rate is no longer shown as a headline number in the UI -- it read
# as untrustworthy to a first-time user ("you can't calculate that"). Linear
# Regression is kept anyway: the project brief calls for forest + linear
# regression specifically, and its standardized coefficients now drive the
# plain-English "what's driving this" text in narrative.py instead of a chart.
# The engagement random forest that used to back the tile is no longer trained.
print("\n==== LINEAR REGRESSION (engagement rate -- powers narrative.py only) ====")
metrics["linear_engagement"] = cv_regressor(
    "engagement | linear", C.TARGET_ENGAGEMENT, HEADLINE_FEATURES, LinearRegression()
)
metrics["linear_engagement"]["baseline"] = baseline_regressor(C.TARGET_ENGAGEMENT)

# ---------------------------------------------------------------------------
# FIT FINAL MODELS ON ALL DATA AND SAVE
# ---------------------------------------------------------------------------
print("\n==== fitting final models on all rows ====")
models = {}

clf = P.make_pipeline(HEADLINE_FEATURES, RandomForestClassifier(**RF_CLF))
clf.fit(df, y_hit)
models["hit_classifier"] = clf
print("  hit_classifier          fitted")

content_clf_final = P.make_pipeline(P.CONTENT_ONLY, RandomForestClassifier(**RF_CLF))
content_clf_final.fit(df, y_hit)
models["content_classifier"] = content_clf_final
print("  content_classifier      fitted")

for key, target_col in C.REGRESSION_TARGETS.items():
    m = P.make_pipeline(HEADLINE_FEATURES, RandomForestRegressor(**RF_REG))
    m.fit(df, df[target_col])
    models[f"reg_{key}"] = m
    print(f"  reg_{key:19s} fitted")

lin = P.make_pipeline(HEADLINE_FEATURES, LinearRegression())
lin.fit(df, df[C.TARGET_ENGAGEMENT])
models["linear_engagement"] = lin
print("  linear_engagement       fitted")

models["headline_features"] = HEADLINE_FEATURES
models["content_features"] = P.CONTENT_ONLY
models["keep_channel_features"] = bool(keep_channel)
models["advice_is_real"] = bool(advice_is_real)

# Logged for the README / dev reference only -- not shown in the UI (the
# "Model honesty" panel was cut). Sourced from the headline classifier, since
# that's the model actually driving predictions.
names = P.transformed_feature_names(clf)
metrics["top_importances"] = sorted(
    zip(names, clf.named_steps["model"].feature_importances_.tolist()),
    key=lambda t: -t[1],
)[:20]

metrics["rf_params"] = {k: v for k, v in RF_CLF.items() if k != "n_jobs"}

C.ARTIFACTS.mkdir(exist_ok=True)
# gzip roughly 3.3x's the forests -- worth ~1s of load time on a free tier that
# clones this repo on every deploy.
with gzip.open(C.MODELS_PKL, "wb", compresslevel=6) as fh:
    pickle.dump(models, fh, protocol=pickle.HIGHEST_PROTOCOL)
with open(C.METRICS_JSON, "w", encoding="utf-8") as fh:
    json.dump(metrics, fh, indent=2)

size_mb = C.MODELS_PKL.stat().st_size / 1e6
print(f"\nSaved {C.MODELS_PKL.name} ({size_mb:.1f} MB gzipped) and {C.METRICS_JSON.name}")
if size_mb > 25:
    print("  WARNING: artifact is large for the backend Docker image -- reduce n_estimators/max_depth")

print("\n==== TOP FEATURE IMPORTANCES (hit classifier) ====")
for n, v in metrics["top_importances"][:15]:
    print(f"  {n:46s} {v:.4f}")
