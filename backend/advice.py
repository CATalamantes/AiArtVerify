"""What should the creator actually change?

Two independent routes to every recommendation:

  1. Model counterfactual -- hold everything else fixed, sweep one knob through
     the classifier, keep the best value.
  2. Empirical hit-rate curve -- what really happened to real trending videos in
     that bucket, straight from the data, no model involved.

A tip is only surfaced when both agree on the direction. That filter is the
difference between an advice engine and a random number generator.

Knobs are swept in *coupled groups* -- changing title_length_chars without
title_word_count would ask the model about a title that cannot exist.
"""

import numpy as np
import pandas as pd

import config as C

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MIN_DELTA = 0.01  # below a full point the tip isn't worth the screen space
MIN_BIN_SUPPORT = 40  # empirical bins thinner than this don't get a vote


# ---------------------------------------------------------------------------
# coupled setters -- keep derived features mutually consistent
# ---------------------------------------------------------------------------
def _set_title_length(row, chars):
    cpw = row["title_length_chars"] / max(row["title_word_count"], 1) or 6.5
    row["title_length_chars"] = chars
    row["title_word_count"] = max(1, round(chars / max(cpw, 1)))


def _set_description_length(row, chars):
    cpw = row["description_length_chars"] / max(row["description_word_count"], 1) or 6.0
    row["description_length_chars"] = chars
    row["description_word_count"] = max(0, round(chars / max(cpw, 1)))
    row["has_description"] = int(chars > 0)
    if chars == 0:
        row["description_url_count"] = 0
        row["description_has_url"] = 0
        row["description_line_count"] = 1


def _set_tag_count(row, n):
    row["tag_count"] = n
    row["has_tags"] = int(n > 0)


def _set_duration(row, seconds):
    row["duration_seconds"] = seconds
    row["is_short"] = int(seconds <= 60)


def _set_urls(row, n):
    row["description_url_count"] = n
    row["description_has_url"] = int(n > 0)


def _set_caps_word(row, on):
    row["title_has_capitalized_word"] = int(on)
    # an ALL-CAPS word drags the uppercase ratio with it
    row["title_uppercase_ratio"] = min(1.0, row["title_uppercase_ratio"] + 0.15) if on else max(
        0.0, row["title_uppercase_ratio"] - 0.15
    )


def _set_publish(row, dow, hour):
    row["publish_day_of_week"] = dow
    row["publish_hour"] = hour
    row["publish_is_weekend"] = int(dow in (5, 6))


# ---------------------------------------------------------------------------
# knob definitions: (key, label, values, setter, formatter, empirical_feature)
# ---------------------------------------------------------------------------
KNOBS = [
    ("title_length_chars", "Title length", list(range(20, 121, 5)), _set_title_length,
     lambda v: f"{v} characters", "title_length_chars"),
    # 0 is deliberately absent from these two. "Delete all your tags" / "delete
    # your description" is never actionable advice, and it would contradict the
    # missing-tags panel sitting right next to it in the UI.
    ("tag_count", "Number of tags", [3, 5, 8, 12, 16, 20, 25, 30, 40], _set_tag_count,
     lambda v: f"{v} tags", "tag_count"),
    ("description_length_chars", "Description length",
     [100, 250, 500, 1000, 2000, 4000], _set_description_length,
     lambda v: f"{v:,} characters", "description_length_chars"),
    ("duration_seconds", "Video length",
     [30, 60, 120, 300, 600, 900, 1200, 1800, 2700, 3600], _set_duration,
     lambda v: f"{v // 60} min" if v >= 60 else f"{v}s", "duration_seconds"),
    ("description_url_count", "Links in description", [0, 1, 2, 3, 5, 8], _set_urls,
     lambda v: f"{v} links", "description_url_count"),
    ("title_has_number", "Number in the title", [0, 1],
     lambda r, v: r.__setitem__("title_has_number", v),
     lambda v: "include a number" if v else "no number", "title_has_number"),
    ("title_has_bracket", "Brackets in the title", [0, 1],
     lambda r, v: r.__setitem__("title_has_bracket", v),
     lambda v: "use (brackets)" if v else "no brackets", "title_has_bracket"),
    ("title_has_capitalized_word", "ALL-CAPS word", [0, 1], _set_caps_word,
     lambda v: "add an ALL-CAPS word" if v else "no ALL-CAPS words", "title_has_capitalized_word"),
    ("title_emoji_count", "Emoji in the title", [0, 1, 2, 3],
     lambda r, v: r.__setitem__("title_emoji_count", v),
     lambda v: f"{v} emoji", "title_emoji_count"),
    ("title_exclamation_count", "Exclamation marks", [0, 1, 2],
     lambda r, v: r.__setitem__("title_exclamation_count", v),
     lambda v: f"{v} exclamation marks", "title_exclamation_count"),
    ("title_question_count", "Question marks", [0, 1],
     lambda r, v: r.__setitem__("title_question_count", v),
     lambda v: f"{v} question marks", "title_question_count"),
]


# ---------------------------------------------------------------------------
# empirical lookup
# ---------------------------------------------------------------------------
def empirical_hit_rate(reference, feature, value):
    """Real-world hit rate for videos whose `feature` sat near `value`."""
    curve = reference.get("empirical_curves", {}).get(feature)
    if not curve:
        return None, 0

    if curve["kind"] == "discrete":
        vals = np.asarray(curve["values"], dtype=float)
        i = int(np.argmin(np.abs(vals - float(value))))
        return curve["hit_rate"][i], curve["n"][i]

    edges = np.asarray(curve["edges"], dtype=float)
    i = int(np.clip(np.digitize([float(value)], edges[1:-1], right=False)[0], 0, len(edges) - 2))
    rate = curve["hit_rate"][i]
    return (None if rate is None or (isinstance(rate, float) and np.isnan(rate)) else rate), curve["n"][i]


# ---------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------
def _candidate_rows(base: dict):
    """Every (knob, value) pair as a feature row. Batched into one predict call."""
    rows, meta = [], []
    for key, label, values, setter, fmt, emp_feat in KNOBS:
        for v in values:
            row = dict(base)
            setter(row, v)
            rows.append(row)
            meta.append((key, label, v, fmt, emp_feat))
    return rows, meta


def recommendations(models, reference, feat: pd.DataFrame, top_n: int = 6):
    """Ranked, cross-checked suggestions. Empty list is a valid answer."""
    base = feat.iloc[0].to_dict()
    headline, content = models["hit_classifier"], models["content_classifier"]

    current_p = float(headline.predict_proba(feat)[:, 1][0])
    current_content_p = float(content.predict_proba(feat)[:, 1][0])

    rows, meta = _candidate_rows(base)
    batch = pd.DataFrame(rows)
    probs = headline.predict_proba(batch)[:, 1]
    content_probs = content.predict_proba(batch)[:, 1]

    by_knob = {}
    for (key, label, value, fmt, emp_feat), p, cp in zip(meta, probs, content_probs):
        by_knob.setdefault(key, {"label": label, "formatter": fmt, "empirical_feature": emp_feat,
                                 "points": []})["points"].append((value, float(p), float(cp)))

    # Pick the *knee*, not the peak. Taking the argmax makes every tip land on
    # the end of its sweep range ("use 40 tags", "add 8 links"), which reads as
    # "more is always better" and is barely actionable. Instead take the
    # smallest change from where the user already is that still captures 90% of
    # the available gain.
    best = {}
    for key, info in by_knob.items():
        pts = info["points"]
        peak = max(p for _, p, _ in pts)
        if peak <= current_p:
            continue
        target = current_p + 0.9 * (peak - current_p)
        current_value = float(base.get(key, 0))
        viable = [(v, p, cp) for v, p, cp in pts if p >= target]
        value, prob, content_prob = min(viable, key=lambda t: abs(float(t[0]) - current_value))
        best[key] = {
            "knob": key,
            "label": info["label"],
            "value": value,
            "prob": prob,
            "content_prob": content_prob,
            "formatter": info["formatter"],
            "empirical_feature": info["empirical_feature"],
        }

    out = []
    for key, b in best.items():
        model_delta = b["prob"] - current_p
        if model_delta < MIN_DELTA:
            continue
        if b["value"] == base.get(key):
            continue  # already optimal -- nothing to suggest

        cur_rate, cur_n = empirical_hit_rate(reference, b["empirical_feature"], base.get(key, 0))
        new_rate, new_n = empirical_hit_rate(reference, b["empirical_feature"], b["value"])
        if cur_rate is None or new_rate is None:
            continue

        emp_delta = new_rate - cur_rate
        thin = min(cur_n, new_n) < MIN_BIN_SUPPORT
        content_delta = b["content_prob"] - current_content_p

        # both routes must point the same way, and the thin-data case is dropped
        if emp_delta <= 0 or thin or content_delta <= 0:
            continue

        out.append(
            {
                "knob": key,
                "label": b["label"],
                "current": b["formatter"](base.get(key, 0)),
                "suggested": b["formatter"](b["value"]),
                "model_delta": model_delta,
                "empirical_delta": emp_delta,
                "points": round(model_delta * 100, 1),
                "support": int(min(cur_n, new_n)),
            }
        )

    out.sort(key=lambda d: -d["model_delta"])
    return out[:top_n], current_p


def best_publish_time(models, reference, feat: pd.DataFrame):
    """All 168 day x hour combinations in a single batched prediction."""
    base = feat.iloc[0].to_dict()
    rows = []
    for dow in range(7):
        for hour in range(24):
            row = dict(base)
            _set_publish(row, dow, hour)
            rows.append(row)

    probs = models["hit_classifier"].predict_proba(pd.DataFrame(rows))[:, 1]
    grid = probs.reshape(7, 24)

    dow, hour = np.unravel_index(int(np.argmax(grid)), grid.shape)
    current = float(
        models["hit_classifier"].predict_proba(feat)[:, 1][0]
    )
    return {
        "grid": grid,
        "days": DAYS,
        "best_day": DAYS[dow],
        "best_hour": int(hour),
        "best_prob": float(grid[dow, hour]),
        "current_prob": current,
        "gain": float(grid[dow, hour]) - current,
    }


def response_curve(models, reference, feat: pd.DataFrame, knob_key: str):
    """P(hit) across one knob's range -- the data behind a response chart."""
    spec = next((k for k in KNOBS if k[0] == knob_key), None)
    if spec is None:
        return None
    _, label, values, setter, fmt, emp_feat = spec

    base = feat.iloc[0].to_dict()
    rows = []
    for v in values:
        row = dict(base)
        setter(row, v)
        rows.append(row)

    probs = models["hit_classifier"].predict_proba(pd.DataFrame(rows))[:, 1]
    empirical = [empirical_hit_rate(reference, emp_feat, v)[0] for v in values]
    return pd.DataFrame(
        {"value": values, "model": probs, "empirical": empirical, "label": [fmt(v) for v in values]}
    )


def missing_tags(tag_vocab: dict, category: str, current_tags: str, limit: int = 12):
    """High-lift tags used by trending videos in this category that you don't have."""
    from features import split_tags

    have = set(split_tags(current_tags))
    pool = tag_vocab.get(category) or tag_vocab.get("_global", [])
    return [t for t in pool if t["tag"] not in have][:limit]


def category_fit(models, reference, feat: pd.DataFrame):
    """How this exact content would score under every category."""
    base = feat.iloc[0].to_dict()
    cats = reference["categories"]
    rows = []
    for c in cats:
        row = dict(base)
        row["video_category"] = c
        rows.append(row)

    probs = models["hit_classifier"].predict_proba(pd.DataFrame(rows))[:, 1]
    stats = reference.get("category_stats", {})
    df = pd.DataFrame(
        {
            "category": cats,
            "probability": probs,
            "n_in_data": [stats.get(c, {}).get("n", 0) for c in cats],
        }
    )
    # categories with a handful of videos can't support a claim
    return df[df["n_in_data"] >= 30].sort_values("probability", ascending=False)
