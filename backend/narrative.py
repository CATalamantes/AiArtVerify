"""Plain-English drivers, replacing the "what's driving this" bar chart.

That chart looked bad and wasn't legible to a first-time user. Same underlying
math -- inference.linear_contributions(), the standardized Linear Regression
coefficients -- just turned into sentences instead of bars. This keeps Linear
Regression doing real work in the project instead of existing only to satisfy
a checklist.

IMPORTANT correctness note (found by testing, not by inspection): a feature's
coefficient sign does not reliably match intuition. has_description's fitted
coefficient is *negative* -- the model learned that, all else equal, having no
description very slightly correlates with a higher engagement RATE (likes+
comments / views) in this dataset, plausibly because heavily-described videos
skew toward passive, informational content. A canned template like "Having a
description is helping" would be flatly wrong whenever description is absent
but the contribution still comes out positive (raw=0 sits far below the
feature's ~0.98 mean, so a negative coefficient times a very negative
standardized value is positive).

So sentences here are built from the row's actual state (present/absent,
above/below the training mean) plus whether the contribution is helping or
hurting -- never from a hardcoded assumption about which direction is "good".
That is slower to write but cannot end up contradicting the input.
"""

import config as C

# Binary features -- sentence states present/absent, never claims a direction.
BINARY_LABELS: dict[str, str] = {
    "has_description": "a description",
    "has_tags": "tags",
    "title_has_number": "a number in the title",
    "title_has_bracket": "a bracket in the title, like (Full Guide)",
    "title_has_capitalized_word": "an ALL-CAPS word in the title",
    "description_has_url": "a link in the description",
    "is_short": "Shorts-length",
    "video_licensed_content": "licensed content status",
    "channel_has_custom_url": "a custom channel URL",
    "channel_have_hidden_subscribers": "a hidden subscriber count",
}

# Continuous/count features -- sentence states above/below the training-set
# mean for this category's model, never a specific "add more" claim.
CONTINUOUS_LABELS: dict[str, str] = {
    "title_length_chars": "Your title length",
    "title_word_count": "Your title's word count",
    "title_uppercase_ratio": "How much of your title is capitalized",
    "title_exclamation_count": "The exclamation marks in your title",
    "title_question_count": "The question marks in your title",
    "title_emoji_count": "The emoji in your title",
    "description_length_chars": "Your description length",
    "description_word_count": "Your description's word count",
    "description_url_count": "The number of links in your description",
    "description_line_count": "How your description is broken into lines",
    "tag_count": "Your number of tags",
    "duration_seconds": "Your video length",
}
ACTIONABLE_STEMS = set(BINARY_LABELS) | set(CONTINUOUS_LABELS)

# Real, but not a "change it tonight" lever -- shown as closing context.
CONTEXT_LABELS: dict[str, str] = {
    "log_channel_subscriber_count": "Your subscriber count",
    "log_channel_view_count": "Your channel's total views",
    "log_channel_video_count": "How many videos you've published",
    "log_channel_avg_views_per_video": "Your channel's average views per video",
    "channel_sub_to_view_ratio": "How loyal vs. viral-driven your audience is",
    "channel_age_days_at_publish": "Your channel's age",
    "channel_description_length": "Your channel description",
    "publish_hour": "Your publish time",
    "publish_day_of_week": "Your publish day",
    "publish_month": "The time of year you're publishing",
    "publish_is_weekend": "Publishing on a weekend",
    "video_definition_hd": "Publishing in HD",
}


def _magnitude_word(value: float, all_values: list[float]) -> str:
    """slightly / (nothing) / significantly, by tertile of |value| among all
    contributions for this prediction -- relative to this row, not a global
    calibration, which keeps it simple and self-consistent."""
    mags = sorted(abs(v) for v in all_values)
    if not mags:
        return ""
    lo = mags[len(mags) // 3]
    hi = mags[(2 * len(mags)) // 3]
    a = abs(value)
    if a <= lo:
        return "slightly "
    if a >= hi:
        return "significantly "
    return ""


def _sentence(row: dict, magnitude: str) -> str | None:
    stem, value = row["stem"], row["value"]
    helping = value > 0
    verb = "helping" if helping else "working against"

    if stem in BINARY_LABELS:
        present = row["raw"] >= 0.5
        state = f"Having {BINARY_LABELS[stem]}" if present else f"Not having {BINARY_LABELS[stem]}"
        return f"{state} is {magnitude}{verb} this prediction."

    if stem in CONTINUOUS_LABELS:
        position = "above" if row["above_mean"] else "below"
        return (f"{CONTINUOUS_LABELS[stem]} is {position} what's typical for this "
                f"category — that's {magnitude}{verb} the prediction.")

    if stem in CONTEXT_LABELS:
        return f"{CONTEXT_LABELS[stem]} is {magnitude}{verb} this prediction — not something to change tonight."

    if stem.startswith("video_category_"):
        cat = stem[len("video_category_"):]
        return f"Filing this under {cat} is {magnitude}{verb} you a little."

    if stem.startswith("channel_country_grouped_"):
        country = stem[len("channel_country_grouped_"):]
        return f"Being based in {country} is {magnitude}{verb} this a little."

    if stem == C.HORIZON_FEATURE:
        return None  # shown numerically in the growth curve already, skip here

    return f"{row['pretty']} is {magnitude}{verb} the prediction."


def build_narrative(contributions: list[dict], top_n: int = 4, context_slots: int = 1) -> list[str]:
    """contributions: the full (top_n=None) list from inference.linear_contributions().

    Channel-authority coefficients run 5-30x larger than any content
    coefficient (e.g. log_channel_view_count ~0.33 vs title_length_chars
    ~0.01) -- a single global top-N by |value| would ALWAYS surface four
    channel sentences and never once mention the title, description, or
    tags, no matter what's actually typed. That defeats the entire point of
    replacing the driver chart with content feedback.

    So this partitions into an actionable pool (title/description/tags/
    duration -- things a creator can change tonight) and a context pool
    (channel/timing/category), takes the strongest `top_n - context_slots`
    from the actionable pool first, then appends up to `context_slots`
    context sentences at the end -- content feedback leads, channel-size
    context closes.
    """
    all_values = [c["value"] for c in contributions]
    actionable = [c for c in contributions if c["stem"] in ACTIONABLE_STEMS]
    context = [c for c in contributions if c["stem"] not in ACTIONABLE_STEMS]

    out = []
    for row in actionable[: max(top_n - context_slots, 1)]:
        sentence = _sentence(row, _magnitude_word(row["value"], all_values))
        if sentence:
            out.append(sentence)

    remaining = top_n - len(out)
    for row in context[:remaining]:
        sentence = _sentence(row, _magnitude_word(row["value"], all_values))
        if sentence:
            out.append(sentence)

    return out[:top_n]
