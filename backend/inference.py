"""Turn a filled-in form into predictions.

The trick that keeps this honest: instead of hand-assembling a feature vector,
we assemble a one-row frame in the *raw cleaned_data.csv schema* and push it
through the same features.extract_features() the training set went through.
Serving and training therefore cannot drift -- test_parity.py proves it.

`video_trending__date` is synthesised as publish time + the user's chosen
horizon, which is what turns the old `hours_to_trending` leak into the
"predicted views at 48h" control.
"""

import gzip
import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import config as C
from features import apply_country_grouping, extract_features


@dataclass
class VideoInput:
    """Everything the form collects. Defaults are dataset medians."""

    title: str = ""
    description: str = ""
    tags: str = ""
    category: str = "Gaming"
    duration_seconds: float = 600.0
    publish_dt: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    )
    horizon_hours: float = 48.0
    # Never asked in the form -- trending_country isn't a question a creator
    # can answer about an unpublished video, and it's excluded from the
    # trained model (pipeline.SERVE_FEATURES), so this value is inert.
    trending_country: str = "Other"
    channel_country: str = "United States"  # asked in the form as "Where are you based?"
    subscribers: float = 100_000.0
    channel_views: float = 50_000_000.0
    channel_videos: float = 500.0
    channel_age_days: float = 2000.0
    channel_description_length: int = 300
    channel_has_custom_url: bool = True
    channel_hidden_subscribers: bool = False
    definition_hd: bool = True
    licensed_content: bool = False


def load_artifacts():
    with gzip.open(C.MODELS_PKL, "rb") as fh:
        models = pickle.load(fh)
    with open(C.REFERENCE_JSON, encoding="utf-8") as fh:
        reference = json.load(fh)
    with open(C.METRICS_JSON, encoding="utf-8") as fh:
        metrics = json.load(fh)
    return models, reference, metrics


def build_raw_row(inp: VideoInput) -> pd.DataFrame:
    """One row in the cleaned_data.csv schema, ready for extract_features()."""
    publish = pd.Timestamp(inp.publish_dt)
    if publish.tzinfo is None:
        publish = publish.tz_localize("UTC")

    return pd.DataFrame(
        [
            {
                "video_title": inp.title,
                "video_description": inp.description,
                "video_tags": inp.tags,
                "video_category_id": inp.category,
                "video_duration": float(inp.duration_seconds),
                "video_definition": "hd" if inp.definition_hd else "sd",
                "video_licensed_content": bool(inp.licensed_content),
                "video_published_at": publish,
                # "measured" this many hours after publish -> the horizon control
                "video_trending__date": publish + pd.Timedelta(hours=float(inp.horizon_hours)),
                "video_trending_country": inp.trending_country,
                "channel_country": inp.channel_country,
                "channel_published_at": publish - pd.Timedelta(days=float(inp.channel_age_days)),
                "channel_subscriber_count": float(inp.subscribers),
                "channel_view_count": float(inp.channel_views),
                "channel_video_count": float(inp.channel_videos),
                "channel_have_hidden_subscribers": bool(inp.channel_hidden_subscribers),
                # only the length is used as a feature, so a filler string of the
                # right length round-trips identically
                "channel_description": "x" * int(inp.channel_description_length),
                "channel_custom_url": "x" if inp.channel_has_custom_url else None,
            }
        ]
    )


def to_features(inp: VideoInput, reference: dict) -> pd.DataFrame:
    feat = extract_features(build_raw_row(inp))
    return apply_country_grouping(feat, reference["country_groups"])


def percentile_of(value: float, distribution) -> float:
    """What share of real trending videos this beats, 0-100."""
    arr = np.asarray(distribution, dtype=float)
    return float((arr < value).mean() * 100.0)


def band_for(prob: float) -> str:
    for cutoff, label in C.HIT_BANDS:
        if prob < cutoff:
            return label
    return C.HIT_BANDS[-1][1]


def predict(models: dict, reference: dict, feat: pd.DataFrame) -> dict:
    """Full prediction bundle for one feature row."""
    hit_prob = float(models["hit_classifier"].predict_proba(feat)[:, 1][0])
    content_prob = float(models["content_classifier"].predict_proba(feat)[:, 1][0])

    out = {
        "hit_probability": hit_prob,
        "hit_score": round(hit_prob * 100),
        "band": band_for(hit_prob),
        "content_probability": content_prob,
        "content_score": round(content_prob * 100),
        "base_rate": reference["base_rate"],
    }

    for key in C.REGRESSION_TARGETS:
        log_pred = float(models[f"reg_{key}"].predict(feat)[0])
        out[key] = float(np.expm1(log_pred))
        out[f"{key}_percentile"] = percentile_of(log_pred, reference[f"dist_log_{key}"])

    # Engagement rate is intentionally not surfaced as a headline number --
    # it read as untrustworthy to a first-time user. The linear model that
    # would have backed it is kept (see linear_contributions below) purely to
    # power the plain-English narrative text.
    return out


def linear_contributions(models: dict, feat: pd.DataFrame, top_n: int | None = 12):
    """Exact per-feature contributions from the linear model, as dicts.

    Valid to compare directly because pipeline.py standardizes every column
    (including one-hots) before the model sees them, so each coefficient is
    "effect per one standard deviation". `top_n=None` returns the full sorted
    list (narrative.py needs it to compute magnitude tertiles).

    The encoder has no drop_first, so every category/country gets its own
    column -- after scaling, an "off" dummy (raw 0) still produces a non-zero
    contribution. Left in, that would let a narrative claim "not being Sports
    is hurting you," which is meaningless. Only the ONE category and ONE
    channel-country dummy that are actually true for this row (raw == 1) are
    kept; every "off" sibling dummy is dropped from the candidate pool here.
    """
    pipe = models["linear_engagement"]
    encoded = pipe.named_steps["encode"].transform(feat)
    names = pipe.named_steps["encode"].get_feature_names_out()
    scaled = pipe.named_steps["scale"].transform(encoded)
    contrib = pipe.named_steps["model"].coef_ * scaled[0]

    means = pipe.named_steps["scale"].mean_
    rows = []
    for n, v, raw_v, mean_v in zip(names, contrib, encoded[0], means):
        stem = n.split("__", 1)[-1]
        is_dummy = stem.startswith(("video_category_", "channel_country_grouped_", "trending_country_grouped_"))
        if is_dummy and raw_v < 0.5:
            continue
        rows.append({
            "stem": stem, "pretty": _pretty(n), "value": float(v),
            "raw": float(raw_v), "above_mean": bool(raw_v > mean_v),
        })

    rows.sort(key=lambda d: -abs(d["value"]))
    return rows if top_n is None else rows[:top_n]


def growth_curve(models: dict, feat: pd.DataFrame, hours=None):
    """Predicted views/likes/comments across the horizon -- one batched call.

    hours_since_publish is independent of every other feature, so the row is
    built once and only that column is varied. Rebuilding features per point
    would be ~50x the pandas work for an identical answer.
    """
    hours = hours if hours is not None else list(range(1, 169, 3))
    batch = pd.concat([feat] * len(hours), ignore_index=True)
    batch[C.HORIZON_FEATURE] = hours

    return pd.DataFrame(
        {
            "hours": hours,
            "views": np.expm1(models["reg_views"].predict(batch)),
            "likes": np.expm1(models["reg_likes"].predict(batch)),
            "comments": np.expm1(models["reg_comments"].predict(batch)),
            "hit_probability": models["hit_classifier"].predict_proba(batch)[:, 1],
        }
    )


PRETTY = {
    "hours_since_publish": "Video age at measurement",
    "title_length_chars": "Title length",
    "title_word_count": "Title word count",
    "title_uppercase_ratio": "Title ALL-CAPS ratio",
    "title_has_capitalized_word": "Has an ALL-CAPS word",
    "title_exclamation_count": "Exclamation marks",
    "title_question_count": "Question marks",
    "title_has_number": "Title contains a number",
    "title_has_bracket": "Title uses (brackets)",
    "title_emoji_count": "Title emoji count",
    "description_length_chars": "Description length",
    "description_word_count": "Description word count",
    "description_url_count": "Links in description",
    "description_line_count": "Description line count",
    "description_has_url": "Description has a link",
    "has_description": "Has a description",
    "tag_count": "Number of tags",
    "has_tags": "Has tags",
    "duration_seconds": "Video duration",
    "is_short": "Is a Short (<60s)",
    "publish_hour": "Publish hour",
    "publish_day_of_week": "Publish weekday",
    "publish_month": "Publish month",
    "publish_is_weekend": "Published on a weekend",
    "log_channel_subscriber_count": "Channel subscribers",
    "log_channel_view_count": "Channel total views",
    "log_channel_video_count": "Channel video count",
    "log_channel_avg_views_per_video": "Channel avg views/video",
    "channel_sub_to_view_ratio": "Subscriber loyalty ratio",
    "channel_age_days_at_publish": "Channel age",
    "channel_description_length": "Channel description length",
    "channel_has_custom_url": "Channel has custom URL",
    "channel_have_hidden_subscribers": "Hidden subscriber count",
    "video_definition_hd": "HD video",
    "video_licensed_content": "Licensed content",
}


def _pretty(name: str) -> str:
    """'cat__video_category_Gaming' -> 'Category: Gaming'."""
    stem = name.split("__", 1)[-1]
    if stem.startswith("video_category_"):
        return f"Category: {stem[len('video_category_'):]}"
    if stem.startswith("trending_country_grouped_"):
        return f"Market: {stem[len('trending_country_grouped_'):]}"
    if stem.startswith("channel_country_grouped_"):
        return f"Channel country: {stem[len('channel_country_grouped_'):]}"
    return PRETTY.get(stem, stem.replace("_", " ").capitalize())
