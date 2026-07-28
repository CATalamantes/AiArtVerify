"""HTTP layer over the existing inference/advice/narrative modules.

None of the ML logic lives here -- this only translates HTTP <-> the same
Python functions the Streamlit build called directly. Runs on Hugging Face
Spaces (Docker); the frontend (GitHub Pages, static) calls it via fetch().

    uvicorn api:app --reload --port 8000
"""

from datetime import date, datetime, timezone

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import advice
import config as C
import narrative
from inference import VideoInput, growth_curve, linear_contributions, load_artifacts, predict, to_features

app = FastAPI(title="Hit or Flop API")

# Public, stateless, read-only prediction endpoint -- no auth, no cookies, no
# state-changing actions, nothing sensitive to protect. Open CORS is the
# simplest correct choice here rather than pinning to one GitHub Pages origin
# that would need updating if the repo/username ever changes.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Loaded once at process start, not per-request.
MODELS, REFERENCE, METRICS = load_artifacts()
with open(C.TAG_VOCAB_JSON, encoding="utf-8") as _fh:
    import json as _json

    TAG_VOCAB = _json.load(_fh)

# Channel-size presets: subscriber/view/video counts are real dataset
# quantiles (see build_features.py's channel_quantiles); channel age is a
# reasonable monotonic heuristic per tier, not a measured quantile -- it's a
# secondary feature and not worth a second artifact just for this.
#
# Labels are generated from the ACTUAL resolved numbers, not hardcoded
# round-number claims. Every video in this dataset already made Trending, so
# even the 10th percentile channel has ~18K subscribers -- a label like
# "Nano (under 1K subs)" would be flatly false against a preset that actually
# sends 18,300. Percentile framing also matches the "leaderboard of winners"
# story used throughout: these presets are where you'd sit among channels
# that already trended, not a universal small/medium/large scale. Anyone
# testing a genuinely tiny channel still can, via the custom fields.
CHANNEL_TIERS = [
    ("Bottom 10% of trending channels", 0.1, 180),
    ("25th percentile", 0.25, 500),
    ("Median", 0.5, 1200),
    ("75th percentile", 0.75, 2200),
    ("Top 10% of trending channels", 0.9, 3500),
]


def _human(n: float) -> str:
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= div:
            return f"{n / div:.1f}{suf}"
    return f"{n:,.0f}"


def _preset_values(tier_label: str) -> dict:
    q, age_days = next((q, age) for label, q, age in CHANNEL_TIERS if label == tier_label)
    quant = REFERENCE["channel_quantiles"][str(q)]
    return {
        "subscribers": quant["subscribers"],
        "channel_views": quant["views"],
        "channel_videos": quant["videos"],
        "channel_age_days": age_days,
    }


CHANNEL_PRESETS = {label: None for label, _, _ in CHANNEL_TIERS}  # preset names, for validation


class PredictRequest(BaseModel):
    title: str = ""
    description: str = ""
    tags: str = ""
    category: str = "Gaming"
    duration_minutes: float = Field(10.0, gt=0, le=600)
    publish_date: date = Field(default_factory=lambda: datetime.now(timezone.utc).date())
    publish_hour: int = Field(17, ge=0, le=23)
    horizon_hours: float = 48.0
    channel_country: str = "United States"

    # Either a preset name, or all four custom fields -- validated below.
    channel_size_preset: str | None = None
    subscribers: float | None = None
    channel_views: float | None = None
    channel_videos: float | None = None
    channel_age_days: float | None = None


def _resolve_channel(req: PredictRequest) -> dict:
    if req.channel_size_preset and req.channel_size_preset in CHANNEL_PRESETS:
        return _preset_values(req.channel_size_preset)
    if None not in (req.subscribers, req.channel_views, req.channel_videos, req.channel_age_days):
        return {
            "subscribers": req.subscribers,
            "channel_views": req.channel_views,
            "channel_videos": req.channel_videos,
            "channel_age_days": req.channel_age_days,
        }
    raise HTTPException(422, "Provide channel_size_preset or all four custom channel fields.")


def _build_input(req: PredictRequest) -> VideoInput:
    channel = _resolve_channel(req)
    publish_dt = datetime.combine(req.publish_date, datetime.min.time(), tzinfo=timezone.utc).replace(
        hour=req.publish_hour
    )
    return VideoInput(
        title=req.title,
        description=req.description,
        tags=req.tags,
        category=req.category,
        duration_seconds=req.duration_minutes * 60,
        publish_dt=publish_dt,
        horizon_hours=req.horizon_hours,
        channel_country=req.channel_country,
        **channel,
    )


@app.get("/reference")
def get_reference():
    """Everything the frontend needs to build the form and the "where you
    land" histogram without hardcoding it. dist_log_views is ~58 KB of plain
    floats -- fetched once at page load, not per-prediction."""
    presets = {}
    for label, _, _ in CHANNEL_TIERS:
        v = _preset_values(label)
        presets[label] = v | {"display": f"{label} (~{_human(v['subscribers'])} subs)"}
    return {
        "categories": REFERENCE["categories"],
        "channel_countries": REFERENCE["channel_countries"],
        "channel_presets": presets,
        "base_rate": REFERENCE["base_rate"],
        "dist_log_views": REFERENCE["dist_log_views"],
    }


@app.get("/tags/{category}")
def get_tags(category: str, limit: int = 20):
    """Latin-script, lift-ranked tags for the click-to-add chip picker."""
    pool = TAG_VOCAB.get(category) or TAG_VOCAB.get("_global", [])
    return {"category": category, "tags": pool[:limit]}


@app.post("/predict")
def post_predict(req: PredictRequest):
    inp = _build_input(req)
    feat = to_features(inp, REFERENCE)

    result = predict(MODELS, REFERENCE, feat)
    recs, _ = advice.recommendations(MODELS, REFERENCE, feat)
    timing = advice.best_publish_time(MODELS, REFERENCE, feat)
    cats = advice.category_fit(MODELS, REFERENCE, feat)
    growth = growth_curve(MODELS, feat)
    contribs = linear_contributions(MODELS, feat, top_n=None)
    narrative_lines = narrative.build_narrative(contribs, top_n=4)
    tags = advice.missing_tags(TAG_VOCAB, req.category, req.tags, limit=12)

    return {
        "prediction": result,
        "narrative": narrative_lines,
        "recommendations": recs,
        "best_time": {
            "grid": timing["grid"].tolist(),
            "days": timing["days"],
            "best_day": timing["best_day"],
            "best_hour": timing["best_hour"],
            "best_prob": timing["best_prob"],
            "current_prob": timing["current_prob"],
            "gain": timing["gain"],
        },
        "category_fit": cats.to_dict("records"),
        "growth_curve": {
            "hours": growth["hours"].tolist() if hasattr(growth["hours"], "tolist") else list(growth["hours"]),
            "views": growth["views"].tolist(),
            "likes": growth["likes"].tolist(),
            "comments": growth["comments"].tolist(),
        },
        "missing_tags": tags,
    }


@app.get("/health")
def health():
    return {"status": "ok"}
