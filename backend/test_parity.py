"""Proof that serving features == training features.

Takes real videos, strips them down to only what the web form collects, rebuilds
them through inference.py, and asserts the resulting feature row matches
features_v2.csv exactly. This is the one test that catches training/serving
skew -- the failure mode that produces a confidently wrong number rather than a
crash.

Also exercises the ugly inputs a real user will actually paste in.

    python test_parity.py
"""

import sys

import numpy as np
import pandas as pd

import config as C
from inference import VideoInput, load_artifacts, predict, to_features

TOL = 1e-9
N_SAMPLE = 20

raw = pd.read_csv(C.CLEANED_DATA)
feat_ref = pd.read_csv(C.FEATURES_V2)
models, reference, _ = load_artifacts()

failures = []

# ---------------------------------------------------------------------------
# 1. ROUND-TRIP PARITY
# ---------------------------------------------------------------------------
print(f"1. Round-trip parity on {N_SAMPLE} real videos")
sample = raw.sample(N_SAMPLE, random_state=7).index

for i in sample:
    ref_row = feat_ref.loc[i]
    r = raw.loc[i]

    inp = VideoInput(
        title=str(r["video_title"]),
        description="" if pd.isna(r["video_description"]) else str(r["video_description"]),
        tags="" if pd.isna(r["video_tags"]) else str(r["video_tags"]),
        category=str(r["video_category_id"]),
        duration_seconds=float(ref_row["duration_seconds"]),
        publish_dt=pd.to_datetime(r["video_published_at"], utc=True, format="mixed"),
        horizon_hours=float(ref_row["hours_since_publish"]),
        trending_country=str(r["video_trending_country"]),
        channel_country="Unknown" if pd.isna(r["channel_country"]) else str(r["channel_country"]),
        subscribers=float(r["channel_subscriber_count"]),
        channel_views=float(r["channel_view_count"]),
        channel_videos=float(r["channel_video_count"]),
        channel_age_days=float(ref_row["channel_age_days_at_publish"]),
        channel_description_length=int(ref_row["channel_description_length"]),
        channel_has_custom_url=bool(ref_row["channel_has_custom_url"]),
        channel_hidden_subscribers=bool(ref_row["channel_have_hidden_subscribers"]),
        definition_hd=bool(ref_row["video_definition_hd"]),
        licensed_content=bool(ref_row["video_licensed_content"]),
    )

    got = to_features(inp, reference).iloc[0]
    for col in C.ALL_FEATURES:
        a, b = got[col], ref_row[col]
        if col in C.CATEGORICAL_FEATURES:
            if str(a) != str(b):
                failures.append(f"  row {i} {col}: got {a!r} expected {b!r}")
        elif not np.isclose(float(a), float(b), rtol=0, atol=TOL):
            failures.append(f"  row {i} {col}: got {a!r} expected {b!r} (diff {abs(float(a)-float(b)):.3e})")

print(f"   {'PASS' if not failures else 'FAIL'} -- {len(failures)} mismatches across "
      f"{N_SAMPLE * len(C.ALL_FEATURES):,} feature values")
for f in failures[:10]:
    print(f)

# ---------------------------------------------------------------------------
# 2. KNOWN-VIDEO SANITY: real hits should outscore real flops
# ---------------------------------------------------------------------------
print("\n2. Known-video sanity check")


def score_index(i):
    ref_row, r = feat_ref.loc[i], raw.loc[i]
    inp = VideoInput(
        title=str(r["video_title"]),
        description="" if pd.isna(r["video_description"]) else str(r["video_description"]),
        tags="" if pd.isna(r["video_tags"]) else str(r["video_tags"]),
        category=str(r["video_category_id"]),
        duration_seconds=float(ref_row["duration_seconds"]),
        publish_dt=pd.to_datetime(r["video_published_at"], utc=True, format="mixed"),
        horizon_hours=float(ref_row["hours_since_publish"]),
        trending_country=str(r["video_trending_country"]),
        channel_country="Unknown" if pd.isna(r["channel_country"]) else str(r["channel_country"]),
        subscribers=float(r["channel_subscriber_count"]),
        channel_views=float(r["channel_view_count"]),
        channel_videos=float(r["channel_video_count"]),
        channel_age_days=float(ref_row["channel_age_days_at_publish"]),
        channel_description_length=int(ref_row["channel_description_length"]),
        channel_has_custom_url=bool(ref_row["channel_has_custom_url"]),
        channel_hidden_subscribers=bool(ref_row["channel_have_hidden_subscribers"]),
        definition_hd=bool(ref_row["video_definition_hd"]),
        licensed_content=bool(ref_row["video_licensed_content"]),
    )
    return predict(models, reference, to_features(inp, reference))["hit_probability"]


top = feat_ref.nlargest(10, "hit_score").index
bottom = feat_ref.nsmallest(10, "hit_score").index
top_mean = np.mean([score_index(i) for i in top])
bottom_mean = np.mean([score_index(i) for i in bottom])
print(f"   top-10 real videos    -> mean P(hit) {top_mean:.3f}")
print(f"   bottom-10 real videos -> mean P(hit) {bottom_mean:.3f}")
sanity_ok = top_mean > bottom_mean + 0.25
print(f"   {'PASS' if sanity_ok else 'FAIL'} -- separation {top_mean - bottom_mean:+.3f}")
if not sanity_ok:
    failures.append("known-video sanity check: insufficient separation")

# ---------------------------------------------------------------------------
# 3. EDGE CASES -- no crashes, no NaN reaching the screen
# ---------------------------------------------------------------------------
print("\n3. Edge cases")
cases = {
    "empty everything": VideoInput(title="", description="", tags=""),
    "no description": VideoInput(title="My video", description="", tags="a,b,c"),
    "zero tags": VideoInput(title="My video", description="hello", tags=""),
    "200-char title": VideoInput(title="A" * 200, description="x", tags="a"),
    "emoji-only title": VideoInput(title="\U0001F525\U0001F600\U0001F4AF", description="x", tags="a"),
    "unseen category": VideoInput(title="Test", category="Nonprofits & Activism"),
    "unseen country": VideoInput(title="Test", trending_country="Atlantis", channel_country="Atlantis"),
    "zero-sub channel": VideoInput(title="Test", subscribers=0, channel_views=0, channel_videos=0),
    "horizon 0": VideoInput(title="Test", horizon_hours=0),
    "newline-heavy desc": VideoInput(title="Test", description="a\n" * 500),
}

for name, inp in cases.items():
    try:
        out = predict(models, reference, to_features(inp, reference))
        bad = [k for k, v in out.items() if isinstance(v, float) and (np.isnan(v) or np.isinf(v))]
        status = "PASS" if not bad else f"FAIL (NaN/inf in {bad})"
        if bad:
            failures.append(f"edge case {name}: {bad}")
        print(f"   {status:8s} {name:22s} P(hit)={out['hit_probability']:.3f} "
              f"views={out['views']:,.0f} band={out['band']}")
    except Exception as exc:  # noqa: BLE001 - we want any failure surfaced
        failures.append(f"edge case {name}: {type(exc).__name__}: {exc}")
        print(f"   FAIL     {name:22s} {type(exc).__name__}: {exc}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
if failures:
    print(f"FAILED -- {len(failures)} problem(s)")
    sys.exit(1)
print("ALL CHECKS PASSED")
