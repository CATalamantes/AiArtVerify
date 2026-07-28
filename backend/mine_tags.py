"""Which tags actually correlate with placing high on the trending page?

Pure data, no model. For each category we compute the hit-rate of videos
carrying each tag and express it as a lift over that category's own base rate,
so "Gaming tags are compared against Gaming".

A minimum support threshold matters more than it looks: without it a tag used by
one lucky video scores a perfect 100% hit rate and tops the list.

    python mine_tags.py  ->  artifacts/tag_vocab.json
"""

import json
import re
from collections import defaultdict

import pandas as pd

import config as C
from features import split_tags

MIN_SUPPORT = 10  # a tag must appear on this many videos to be rankable
TOP_PER_CATEGORY = 40
MIN_CATEGORY_SIZE = 30  # categories smaller than this get the global list instead

# The dataset is global across 109 countries, so plenty of mined tags are
# Cyrillic/Arabic/Devanagari/etc -- real data, but a chip an English-speaking
# creator can't read isn't "foolproof," it's confusing. Restrict to tags a
# Latin-alphabet user can read and click. Plenty survive: Gaming keeps 20/40,
# Music keeps 36/40.
LATIN_ONLY = re.compile(r"^[\x20-\x7E]+$")

print("Loading ...")
raw = pd.read_csv(C.CLEANED_DATA)
feat = pd.read_csv(C.FEATURES_V2)

hits = feat[C.TARGET_HIT].to_numpy()
categories = feat["video_category"].to_numpy()

# (category, tag) -> [n_videos, n_hits]; plus a global rollup
per_cat = defaultdict(lambda: [0, 0])
overall = defaultdict(lambda: [0, 0])

for i, tags_raw in enumerate(raw["video_tags"]):
    for tag in set(split_tags(tags_raw)):
        if len(tag) < 2 or len(tag) > 40 or not LATIN_ONLY.match(tag):
            continue
        per_cat[(categories[i], tag)][0] += 1
        per_cat[(categories[i], tag)][1] += hits[i]
        overall[tag][0] += 1
        overall[tag][1] += hits[i]

print(f"  {len(overall):,} distinct tags across {len(raw):,} videos")

category_base = feat.groupby("video_category")[C.TARGET_HIT].mean().to_dict()
category_size = feat["video_category"].value_counts().to_dict()
global_base = float(feat[C.TARGET_HIT].mean())


def rank(entries, base_rate):
    """Rank tags by hit-rate lift over the relevant base rate."""
    out = []
    for tag, (n, n_hits) in entries:
        if n < MIN_SUPPORT:
            continue
        hit_rate = n_hits / n
        out.append(
            {
                "tag": tag,
                "n": int(n),
                "hit_rate": round(float(hit_rate), 4),
                "lift": round(float(hit_rate / base_rate) if base_rate > 0 else 0.0, 3),
            }
        )
    out.sort(key=lambda d: (-d["lift"], -d["n"]))
    return out[:TOP_PER_CATEGORY]


vocab = {"_global": rank(overall.items(), global_base), "_global_base_rate": global_base}

by_cat = defaultdict(list)
for (cat, tag), counts in per_cat.items():
    by_cat[cat].append((tag, counts))

for cat, entries in by_cat.items():
    # Small categories can't support a per-category ranking -- Pets & Animals has
    # exactly one video. Those fall back to the global list in the app.
    if category_size.get(cat, 0) < MIN_CATEGORY_SIZE:
        continue
    ranked = rank(entries, category_base.get(cat, global_base))
    if ranked:
        vocab[cat] = ranked

C.ARTIFACTS.mkdir(exist_ok=True)
with open(C.TAG_VOCAB_JSON, "w", encoding="utf-8") as fh:
    json.dump(vocab, fh, ensure_ascii=False)

print(f"\nSaved {C.TAG_VOCAB_JSON.name} ({C.TAG_VOCAB_JSON.stat().st_size / 1024:.0f} KB)")
for cat in list(vocab):
    if cat.startswith("_"):
        continue
    top = ", ".join(f"{d['tag']} ({d['lift']:.1f}x)" for d in vocab[cat][:5])
    # Windows consoles are cp1252; tags are frequently not
    line = f"  {cat:22s} n_tags={len(vocab[cat]):3d}  {top}"
    print(line.encode("ascii", "replace").decode("ascii"))
