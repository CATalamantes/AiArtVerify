"""Model pipeline construction.

The v1 pipeline used pd.get_dummies + a manual column list, which meant serving
a prediction required hand-rebuilding a 78-column vector in exactly the right
order, with silently-dropped baseline categories. That is the single most likely
source of a wrong-but-plausible prediction.

Here the encoder is fitted as part of the model, so inference is one
`.predict(raw_dataframe)` call and column order cannot drift. OneHotEncoder uses
handle_unknown="ignore", so a category or country the model never saw during
training degrades to all-zeros instead of raising.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as C


def make_pipeline(feature_list, model):
    """Wrap `model` in encode -> scale -> fit over exactly `feature_list`."""
    categorical = [c for c in feature_list if c in C.CATEGORICAL_FEATURES]
    numeric = [c for c in feature_list if c not in C.CATEGORICAL_FEATURES]

    encode = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("num", "passthrough", numeric),
        ]
    )

    # Scaling happens after encoding so the one-hot columns are standardized
    # too. That is what makes LinearRegression coefficients directly comparable
    # against each other in the "what's driving this" chart.
    return Pipeline(
        [
            ("encode", encode),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


def transformed_feature_names(fitted_pipeline):
    """Column names after encoding, aligned with model.coef_ / feature_importances_."""
    return list(fitted_pipeline.named_steps["encode"].get_feature_names_out())


# Feature sets ---------------------------------------------------------------
FULL = C.ALL_FEATURES

# What actually gets trained and served: everything except trending_country,
# which the v1 UI exposed as a confusing "Target Market" dropdown that didn't
# even list the US/UK. Measured cost of dropping it: none (AUC 0.847 -> 0.849,
# within CV noise). channel_country stays -- see config.py's note.
SERVE_FEATURES = [f for f in C.ALL_FEATURES if f != C.DROP_FROM_SERVING]

# What the advice engine runs on: only things a creator can actually change,
# plus the horizon as a control variable so sweeping a title knob doesn't get
# confounded by video age.
CONTENT_ONLY = C.CONTENT_FEATURES + C.UPLOAD_FEATURES + [C.HORIZON_FEATURE]

# Gate A comparison arm: everything (minus trending_country) except the
# channel's own stats.
NO_CHANNEL = [f for f in SERVE_FEATURES if f not in C.CHANNEL_FEATURES]
