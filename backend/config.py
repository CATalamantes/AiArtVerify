"""Single source of truth for paths, feature groups and scoring constants.

Every other module imports from here. A feature can only ever belong to one
group, so it is impossible for training and inference to disagree about which
bucket something lives in.
"""

from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
ROOT = BACKEND_DIR.parent  # repo root -- shared data files (cleaned_data.csv etc) live here

# artifacts/ ships inside the backend Docker image, so it lives under backend/,
# not at the repo root where the large training CSVs sit.
ARTIFACTS = BACKEND_DIR / "artifacts"

CLEANED_DATA = ROOT / "cleaned_data.csv"
FEATURES_V2 = ROOT / "features_v2.csv"

MODELS_PKL = ARTIFACTS / "models.pkl.gz"
METRICS_JSON = ARTIFACTS / "metrics.json"
REFERENCE_JSON = ARTIFACTS / "reference.json"
TAG_VOCAB_JSON = ARTIFACTS / "tag_vocab.json"

# ---------------------------------------------------------------------------
# FEATURE GROUPS
# ---------------------------------------------------------------------------
# Everything a creator can change tonight. The advice engine only ever touches
# these, which is what stops every recommendation collapsing into
# "get more subscribers".
CONTENT_FEATURES = [
    "publish_hour",
    "publish_day_of_week",
    "publish_month",
    "publish_is_weekend",
    "title_length_chars",
    "title_word_count",
    "title_has_capitalized_word",
    "title_uppercase_ratio",
    "title_exclamation_count",
    "title_question_count",
    "title_has_number",
    "title_has_bracket",
    "title_emoji_count",
    "description_length_chars",
    "description_word_count",
    "description_has_url",
    "description_url_count",
    "description_line_count",
    "has_description",
    "tag_count",
    "has_tags",
    "video_category",
]

# Known at upload time but not part of the title/description/tags trio.
UPLOAD_FEATURES = [
    "duration_seconds",
    "is_short",
    "video_definition_hd",
    "video_licensed_content",
]

# Genuinely known before publishing, but they describe the channel rather than
# the video. Subject to the Gate A ablation in train_v2.py.
CHANNEL_FEATURES = [
    "channel_age_days_at_publish",
    "channel_have_hidden_subscribers",
    "channel_sub_to_view_ratio",
    "channel_description_length",
    "channel_has_custom_url",
    "log_channel_subscriber_count",
    "log_channel_view_count",
    "log_channel_video_count",
    "log_channel_avg_views_per_video",
]

# trending_country ("which country's Trending page") is not a question a
# creator can answer about an unpublished video -- excluded from the trained
# model in pipeline.py (see SERVE_FEATURES). channel_country ("where are you
# based?") stays: it's a real proxy for audience location and something a
# creator actually knows about themselves.
MARKET_FEATURES = ["trending_country_grouped", "channel_country_grouped"]
DROP_FROM_SERVING = "trending_country_grouped"

# Video age at the moment of measurement. In training this is the real gap
# between publish and the trending snapshot; at inference the user picks it as
# a prediction horizon ("views at 48h"). See features.py.
HORIZON_FEATURE = "hours_since_publish"

ALL_FEATURES = (
    CONTENT_FEATURES + UPLOAD_FEATURES + CHANNEL_FEATURES + MARKET_FEATURES + [HORIZON_FEATURE]
)

CATEGORICAL_FEATURES = ["video_category", "trending_country_grouped", "channel_country_grouped"]

# ---------------------------------------------------------------------------
# TARGETS
# ---------------------------------------------------------------------------
TARGET_HIT = "is_hit"
TARGET_ENGAGEMENT = "_target_engagement_rate"
REGRESSION_TARGETS = {
    "views": "_target_log_views",
    "likes": "_target_log_likes",
    "comments": "_target_log_comments",
}

# ---------------------------------------------------------------------------
# SCORING CONSTANTS
# ---------------------------------------------------------------------------
# A video is a "hit" if its blended reach+engagement percentile is in the top
# quartile of the trending pack. Blending stops the label degenerating into
# pure reach (= big channel) or pure engagement (= tiny loyal audience).
HIT_THRESHOLD = 0.75
REACH_WEIGHT = 0.5
ENGAGEMENT_WEIGHT = 0.5

TOP_N_COUNTRIES = 15
RARE_CATEGORY_LABEL = "Other"

# Gauge bands, expressed as P(hit). The base rate is 0.25 by construction, so
# anything above that is genuinely above-average *for a trending video*.
HIT_BANDS = [
    (0.10, "FLOP"),
    (0.25, "MID"),
    (0.50, "HIT"),
    (1.01, "VIRAL"),
]
BASE_RATE = 1.0 - HIT_THRESHOLD

# ---------------------------------------------------------------------------
# GATES (see train_v2.py) — thresholds that decide what the app ships
# ---------------------------------------------------------------------------
GATE_A_MIN_AUC_GAIN = 0.03  # channel features must earn this much AUC to stay
GATE_B_MIN_AUC = 0.58  # content-only model must beat this for advice to be real

RANDOM_STATE = 42
CV_FOLDS = 5
