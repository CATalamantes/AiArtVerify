"""
Streamlit app — predicts a YouTube video's view count from pre-publish
metadata only, using the deduplicated / group-split Random Forest
(models/random_forest_dedup_grouped.pkl, R^2 0.083 on held-out videos,
see notebooks/04_dedup_group_split.ipynb).

This is the pre-publish-only model (no days_to_trend) — the version meant
for deployment, since days_to_trend isn't knowable before a video trends.
"""

import json
import os

import joblib
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "random_forest_dedup_grouped.pkl")
FEATURES_PATH = os.path.join(ROOT, "models", "baseline_feature_columns.json")
STATS_PATH = os.path.join(ROOT, "models", "baseline_feature_stats.json")

# From notebooks/04_dedup_group_split.ipynb — validation RMSE/R^2 of this
# exact model artifact, on videos never seen in training.
MODEL_RMSE = 7_011_337
MODEL_R2 = 0.083

GREEN = "#0ca30c"  # dataviz skill status palette — "good" (pushes prediction up)
RED = "#d03b3b"    # dataviz skill status palette — "critical" (pushes prediction down)

CATEGORIES = [
    "Autos & Vehicles",
    "Comedy",
    "Education",
    "Entertainment",
    "Film & Animation",
    "Gaming",
    "Howto & Style",
    "Music",
    "News & Politics",
    "Nonprofits & Activism",
    "People & Blogs",
    "Pets & Animals",
    "Science & Technology",
    "Sports",
    "Travel & Events",
]

DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_feature_columns():
    with open(FEATURES_PATH) as f:
        return json.load(f)


@st.cache_resource
def load_baseline_stats():
    with open(STATS_PATH) as f:
        return json.load(f)


def encode_row(feature_columns, category, tag_count, title_length, title_has_caps_word,
                publish_hour, publish_dayofweek):
    row = {col: 0 for col in feature_columns}
    row["tag_count"] = tag_count
    row["title_length"] = title_length
    row["title_has_caps_word"] = title_has_caps_word
    row["publish_hour"] = publish_hour
    row["publish_dayofweek"] = publish_dayofweek

    category_col = f"video_category_id_{category}"
    if category_col in row:
        row[category_col] = 1
    # else: category is the dropped baseline ("Autos & Vehicles") — all dummies stay 0

    return pd.DataFrame([row], columns=feature_columns)


def describe_feature(key, value):
    if key == "category":
        return f"Video category: {value}"
    if key == "tag_count":
        return f"Tag count: {round(value)} tags"
    if key == "title_length":
        return f"Title length: {round(value)} characters"
    if key == "title_has_caps_word":
        return "Title has an ALL-CAPS word" if value else "Title has no ALL-CAPS word"
    if key == "publish_hour":
        return f"Posting at {round(value):02d}:00 UTC"
    if key == "publish_dayofweek":
        return f"Posting on {DAYS_OF_WEEK[round(value) % 7]}"
    return key


def compute_prediction_and_contributions(model, feature_columns, stats, category, tag_count,
                                          title_length, title_has_caps_word, publish_hour,
                                          publish_dayofweek):
    user_values = {
        "category": category,
        "tag_count": tag_count,
        "title_length": title_length,
        "title_has_caps_word": title_has_caps_word,
        "publish_hour": publish_hour,
        "publish_dayofweek": publish_dayofweek,
    }
    typical_values = {
        "category": stats["video_category_id_mode"],
        "tag_count": stats["tag_count_mean"],
        "title_length": stats["title_length_mean"],
        "title_has_caps_word": stats["title_has_caps_word_mode"],
        "publish_hour": stats["publish_hour_mean"],
        "publish_dayofweek": stats["publish_dayofweek_mean"],
    }

    X_user = encode_row(feature_columns, **user_values)
    prediction = model.predict(X_user)[0]

    contributions = []
    for key in user_values:
        ablated_values = dict(user_values)
        ablated_values[key] = typical_values[key]
        X_ablated = encode_row(feature_columns, **ablated_values)
        ablated_prediction = model.predict(X_ablated)[0]
        contributions.append({
            "label": describe_feature(key, user_values[key]),
            "contribution": prediction - ablated_prediction,
        })

    contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)
    return prediction, contributions


def format_compact(n):
    n = max(n, 0)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return f"{n:.0f}"


st.set_page_config(page_title="YouTube View Count Predictor", page_icon="📈", layout="centered")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 760px; }
    div[data-testid="stMetricValue"] { font-size: 2.6rem; }
    h1 { font-size: 2rem; margin-bottom: 0.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 YouTube View Count Predictor")
st.caption(
    "Estimates view count from metadata available **before publishing** — "
    "no channel history, no thumbnail, no trending-day signal."
)

model = load_model()
feature_columns = load_feature_columns()
stats = load_baseline_stats()

with st.container(border=True):
    st.subheader("Tell us about your video")

    title = st.text_input(
        "Video title",
        value="My Awesome New Video",
        help="Used to compute title length and whether it contains an ALL-CAPS word.",
    )

    category = st.selectbox("Category", CATEGORIES, index=CATEGORIES.index("Entertainment"))
    st.caption("Pick the category closest to your video's content.")

    tag_count = st.slider("Number of tags", min_value=0, max_value=30, value=15)
    st.caption("Most trending videos in our dataset use somewhere between 10 and 20 tags.")

    col1, col2 = st.columns(2)
    with col1:
        publish_hour = st.slider("Planned publish hour", min_value=0, max_value=23, value=14)
    with col2:
        publish_day_label = st.selectbox("Planned publish day", DAYS_OF_WEEK, index=4)
        publish_dayofweek = DAYS_OF_WEEK.index(publish_day_label)
    st.caption("Hour is in UTC — convert from your local time zone if needed.")

    submitted = st.button("Predict View Count", type="primary", use_container_width=True)

with st.expander("How this works"):
    st.markdown(
        "This model was trained on roughly **9.9 million rows** from real YouTube "
        "trending videos, using only information that's knowable **before a video "
        "is published** — category, tag count, title characteristics, and planned "
        "posting time. It's a **Random Forest**, a model that combines many decision "
        "trees to capture non-linear patterns a simple formula would miss.\n\n"
        f"It explains about **{MODEL_R2:.0%} of the variation** in view counts on "
        "videos it never saw during training. The rest comes down to things this "
        "model has no way to see — title *quality* (not just length), thumbnail, "
        "channel size and history, timing luck, and how a video gets promoted or "
        "shared after it goes live. Treat every prediction here as a rough "
        "directional signal, not a forecast."
    )

if submitted:
    title_length = len(title)
    title_has_caps_word = int(any(w.isupper() and len(w) > 1 for w in title.split()))

    prediction, contributions = compute_prediction_and_contributions(
        model, feature_columns, stats, category, tag_count, title_length,
        title_has_caps_word, publish_hour, publish_dayofweek,
    )
    low = max(0, prediction - MODEL_RMSE)
    high = prediction + MODEL_RMSE

    st.divider()
    with st.container(border=True):
        st.subheader("Results")

        st.metric("Estimated view count", format_compact(prediction))
        st.caption(
            f"Exact model output: {prediction:,.0f} views · "
            f"Typical range: {format_compact(low)} – {format_compact(high)} views"
        )

        st.info(
            f"**Take this with a grain of salt.** This model explains about "
            f"{MODEL_R2:.0%} of what drives view counts. Category, tags, and timing "
            "matter, but title quality, thumbnail, channel size, and luck matter "
            "far more — treat this as a rough directional signal, not a forecast. "
            "The range above (±7M views) reflects real model uncertainty, not "
            "precision."
        )

        st.markdown("#### What moved this prediction")
        chart_df = pd.DataFrame(contributions)
        chart_df["color"] = chart_df["contribution"].apply(lambda v: GREEN if v >= 0 else RED)
        chart_df = chart_df.iloc[::-1]  # smallest |contribution| first -> largest ends up on top

        st.bar_chart(
            chart_df,
            x="label",
            y="contribution",
            color="color",
            horizontal=True,
            sort=False,
            x_label="",
            y_label="Change in predicted views",
        )
        st.caption(
            "🟢 Green = pushes the prediction up · 🔴 Red = pushes it down. "
            "This shows how much each choice moved the prediction compared to a "
            "typical video in our dataset — not a guarantee of what will happen."
        )
