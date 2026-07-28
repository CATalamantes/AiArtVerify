# HIT OR FLOP

**A pre-publish YouTube performance predictor — trained entirely on videos that already won.**

Type in a title, description, tags, category and channel size. Get back a Hit/Flop verdict,
predicted views/likes/comments, and a ranked list of things to change before you upload.

```bash
# backend
cd backend
python -m venv ../.venv && ../.venv/Scripts/activate      # Windows
pip install -r requirements.txt
uvicorn api:app --reload --port 8000

# frontend, in a second terminal
cd docs
python -m http.server 5500
# open http://localhost:5500
```

---

## The problem, and why it became the product

The dataset is 6,668 videos scraped from YouTube Trending across 109 countries — **all of them
captured on a single snapshot day, and every single one already trending.**

That means there is no negative class. *"Will this video go viral?"* is not a learnable question
from this data, and any model claiming to answer it would be lying. We could have buried that in a
disclaimer. Instead it became the framing:

> **The dataset is a leaderboard of winners.** So the question isn't "will you go viral" — it's
> **"if your video breaks into Trending, do you top that pack or scrape the bottom of it?"**

That's a much harder bar than it sounds, and it's an honest one. Average *here* is elite anywhere
else. The app says so on the front page.

### What a "hit" means, precisely

```python
score  = 0.5 * percentile(views) + 0.5 * percentile(engagement_rate)
is_hit = score >= 0.75          # top quartile of the trending pack
```

Reach alone would just re-label "has a big channel." Engagement alone would crown tiny videos with
loyal audiences. The blend resists both. Base rate is 25.0% by construction.

---

## Results

Everything below is **5-fold cross-validated**, not a single lucky split.

| Model | Metric | Score | Baseline |
|---|---|---|---|
| Hit classifier (Random Forest) | AUC | **0.849** | 0.500 |
| Content-only classifier | AUC | **0.731** | 0.500 |
| log views (Random Forest) | R² | **0.691** | 0.000 |
| log likes | R² | 0.570 | 0.000 |
| log comments | R² | 0.552 | 0.000 |
| engagement rate (Linear Regression) | R² | 0.263 | 0.000 |

View predictions land within **1.8×** of the true figure for a typical video — a more meaningful
statement than log-scale MAE, which means nothing to a user.

### Two gates decided what actually shipped

Both ran *before* any UI work, because both change what the UI contains.

**Gate A — do channel stats earn their place?** Ablated the 9 channel features and re-measured.

| | AUC |
|---|---|
| With channel features | **0.849** |
| Without | 0.752 |
| **Gain** | **+0.097** (threshold to keep: +0.03) |

Comfortably over the line, so the form asks for channel size and the model uses it.

**Gate B — is there real signal in content alone?** A classifier using only title, description,
tags, timing and duration — no channel information at all — scored **AUC 0.731** against a 0.58
threshold. It passed, which is the only reason the "How to improve it" panel exists. Below that
line the plan was to delete it and ship plain category benchmarks instead of inventing advice.

### A third, smaller decision made mid-build: which country fields to keep

The first UI asked for both "Target Market" (which country's Trending page the video appeared on)
and "Channel Country." Sitting with real screenshots surfaced that "Target Market" was a field
with no correct answer — most creators don't know or control which country's Trending page they'll
land on, and the dropdown didn't even list the US or UK. **Cut**, at zero measured cost (AUC ticked
up slightly, 0.847 → 0.849, within noise). **Channel country was kept** — a creator's own location
is a real, answerable proxy for where their core audience sits — reframed in the form as *"Where
are you based?"*

---

## How the advice engine avoids making things up

Every suggestion has to survive **two independent checks that can disagree**:

1. **Model counterfactual** — hold everything else fixed, sweep one knob through the classifier.
2. **Empirical hit-rate curve** — what actually happened to real trending videos in that bucket.
   Straight from the data, no model involved.

If the two disagree on direction, the tip is dropped. If the supporting bucket has fewer than 40
videos, it's dropped. If the gain is under 1 point, it's dropped. **An empty suggestions list is a
valid, honest output** and the UI says so rather than padding with noise.

Two further details that matter more than they look:

- **Knobs move in coupled groups.** Changing `title_length_chars` without `title_word_count` would
  ask the model about a title that cannot exist.
- **Suggestions take the knee, not the peak.** Taking the argmax made every tip land on the end of
  its sweep range ("use 40 tags", "8 links") — technically the model's answer, but it reads as
  "more is always better" and is barely actionable. Instead we take the *smallest* change from
  where you already are that still captures 90% of the available gain.

Tags are mined per category by hit-rate lift with a **minimum support of 10 videos** — without that
floor, a tag used by one lucky video scores a perfect 100% hit rate and tops the list. They're also
**filtered to Latin script** — the dataset is global across 109 countries, so plenty of mined tags
are Cyrillic/Arabic/Devanagari. Real data, but not something an English-speaking creator can read
or click, so they're excluded before ranking.

### "What's driving this" is text, not a chart — and that took a real fix

Standardized Linear Regression coefficients power a plain-English explanation instead of a bar
chart (`backend/narrative.py`). Two bugs surfaced by testing, not by inspection, before this was
trustworthy:

1. **Channel-authority coefficients are 5–30× larger than any content coefficient**
   (`log_channel_view_count` ≈ 0.33 vs `title_length_chars` ≈ 0.01). A naive "top-4 by absolute
   value" selection surfaced four channel sentences, every time, regardless of what was actually
   typed — the exact opposite of what replacing the chart was supposed to achieve. Fixed by
   partitioning into an actionable pool (title/description/tags/duration) and a context pool
   (channel/timing), leading with actionable content and closing with one context note.
2. **A feature's coefficient sign doesn't reliably match intuition.** `has_description`'s fitted
   coefficient is *negative* — the model learned that, all else equal, having no description
   very slightly correlates with a higher engagement *rate*, plausibly because heavily-described
   videos skew toward passive, informational content. A template that said "Having a description
   is helping" would be flatly wrong whenever the description was empty but the contribution still
   came out positive. Every sentence is now built from the row's actual state (present/absent,
   above/below the training mean) plus the contribution's sign — never from a hardcoded assumption
   about which direction is "good."

---

## The horizon trick

`hours_to_trending` was the most dangerous feature in the original pipeline. Because the data is a
single-day snapshot, it's really *"how old the video was when we measured it"* — a fresh upload
would get `0`, far off-distribution, and it was one of the strongest predictors in the model.

Rather than dropping it, it's renamed `hours_since_publish`, kept as an input, and **exposed in the
UI as a prediction horizon**: "views at 24h / 48h / 7 days." Sweeping it 0→168h draws the predicted
growth curve. The leak became the app's most interactive feature.

---

## Architecture

A real backend/frontend split — Streamlit was dropped because you don't own the HTML it renders,
which made the purple, scroll-reveal design in the mockups effectively unbuildable in it.

```
GitHub repo (single source of truth)
 │
 ├─ backend/     FastAPI — POST /predict, GET /reference, GET /tags/{category}
 │               deploys to Hugging Face Spaces (Docker SDK, free tier)
 │
 └─ docs/        static HTML/CSS/JS, no build step, no framework
                 deploys to GitHub Pages, fetch()'s the backend directly (CORS-enabled)
                 -> a real github.io URL, not an iframe. Named docs/ (not frontend/) because
                    GitHub Pages' branch-deploy UI only offers "/" or "/docs" as folder choices.
```

```
backend/
  cleaned_data.csv, features_v2.csv     (repo root -- shared with the v1 pipeline)
   │
   ├─ build_features.py   ──►  features_v2.csv  +  artifacts/reference.json
   │      38 pre-publish-safe features, 4 targets, the is_hit label,
   │      empirical hit-rate curves, distributions the "where you land" chart plots
   │
   ├─ mine_tags.py        ──►  artifacts/tag_vocab.json     (Latin-script filtered)
   │
   ├─ train_v2.py         ──►  artifacts/models.pkl.gz  +  artifacts/metrics.json
   │      6 fitted pipelines, both gates, cross-validated metrics
   │
   └─ api.py              ──►  the HTTP layer (FastAPI)
          inference.py    (form -> prediction; VideoInput, predict(), growth_curve())
          advice.py       (counterfactuals + empirical cross-check)
          narrative.py    (linear coefficients -> plain-English sentences)
          features.py     (feature extraction -- shared by training AND serving)
          pipeline.py     (encode -> scale -> model; SERVE_FEATURES)
          config.py       (feature groups, constants, paths)

docs/
  index.html
  css/   base.css  form.css  results.css  animations.css
  js/    api.js  charts.js  form.js  results.js  main.js
```

### Why serving can't drift from training

`features.py::extract_features()` is the **only** place features are built. `build_features.py`
calls it on the full dataset; `inference.py` calls it on a one-row frame assembled from the API
request, synthesising `video_trending__date` as publish time + horizon.

`test_parity.py` proves it — it takes 20 real videos, strips them to only what the form collects,
rebuilds them through the serving path, and asserts every feature matches to `1e-9`:

```
1. Round-trip parity on 20 real videos
   PASS -- 0 mismatches across 760 feature values
2. Known-video sanity check
   top-10 real videos    -> mean P(hit) 0.620
   bottom-10 real videos -> mean P(hit) 0.091
3. Edge cases
   PASS  empty everything / no description / zero tags / 200-char title /
         emoji-only title / unseen category / unseen country / zero-sub
         channel / horizon 0 / newline-heavy description
```

Training/serving skew is the failure mode that produces a *confidently wrong number* rather than a
crash, which is exactly why it gets its own test.

The v1 pipeline used `pd.get_dummies(drop_first=True)` plus a manual column list, so serving would
have meant hand-rebuilding a 78-column vector in exact order with silently-dropped baselines. v2
fits the encoder as part of an sklearn `Pipeline`, with `handle_unknown="ignore"` so an unseen
category degrades to all-zeros instead of raising.

### The frontend has no framework and no build step

Four hand-rolled SVG charts (`docs/js/charts.js`) — a gauge, a growth-curve area chart, a
day/hour heatmap, a category bar list, and the "where you land" histogram — with zero external
charting library. Plotly or Chart.js would be the heaviest thing on the page for four chart types
this simple. Tags are a click-to-add chip picker (`GET /tags/{category}`, mined and Latin-filtered
server-side) rather than free text, so getting a reasonable tag set takes zero typing.

**Scroll-reveal:** only the form is visible on load; submitting reveals the results section and
scrolls it to the top of the viewport, so the verdict number and gauge become center stage — the
results section can't `display:none` mid-transition, so the reveal removes that class, forces a
reflow, then triggers the opacity/transform transition on the next frame (`docs/js/main.js`).

---

## Reproducing

```bash
cd backend
python -m venv ../.venv && ../.venv/Scripts/activate      # Windows
pip install -r requirements.txt

python build_features.py     # ~10s   -> features_v2.csv, reference.json
python mine_tags.py          # ~10s   -> tag_vocab.json
python train_v2.py           # ~2min  -> models.pkl.gz, metrics.json  (runs both gates)
python test_parity.py        # ~30s   -> must print ALL CHECKS PASSED

uvicorn api:app --reload --port 8000
```

Then in a second terminal, `cd docs && python -m http.server 5500` and open
`http://localhost:5500`.

`cleaning.py` reads `data/youtube_trending_videos_global_daily.parquet`, which is **not in the
repo**. The v2 pipeline starts from the committed `cleaned_data.csv`, so you only need the parquet
if you want to redo the cleaning step.

### Model size

The v1 config (`n_estimators=300, max_depth=None, min_samples_leaf=3`) produced a **243 MB**
artifact. A size-vs-quality sweep picked `150 / 12 / 10`:

| Config | AUC | R² views | Size |
|---|---|---|---|
| 300 / None / 3 | 0.859 | 0.708 | 22 MB per forest |
| **150 / 12 / 10** | **0.849** | **0.691** | **~4 MB per forest** |

Trading ~0.01 AUC for a 4–5× smaller artifact was the right call, even after moving to a real
backend server — the gzipped bundle (6 fitted pipelines) is **9.6 MB**, comfortably inside a free
Hugging Face Space.

`CalibratedClassifierCV` was tested and **rejected**: isotonic calibration moved Brier only
0.1340 → 0.1320 while *lowering* AUC and costing 2.2× the size. The raw forest probabilities are
already about as calibrated as they get.

---

## Deploying

**Backend → Hugging Face Spaces** (free, Docker SDK):
1. Create a Space, choose **Docker** as the SDK.
2. Push the contents of `backend/` to the Space's git remote (or link the GitHub repo for
   auto-sync) — `backend/README.md`'s frontmatter (`sdk: docker`, `app_port: 7860`) is what HF
   reads to build it.
3. The Space sleeps after ~48h idle and wakes on the next request — not a hard shutdown, just a
   30s–2min cold start on the first hit after a long gap. Worth waking it before a live demo.

**Frontend → GitHub Pages** (free, static, no sleep):
1. Set `API_BASE_URL` in `docs/js/api.js` to the deployed Space's URL
   (`https://<user>-hitorflop.hf.space`).
2. Settings → Pages → deploy from branch, folder `/docs` — the only two folder choices GitHub
   Pages' branch-deploy UI offers are `/` and `/docs`, which is why this project uses the latter
   instead of a `frontend/` name.
3. Result: `https://<username>.github.io/HitOrFlop` — a real, directly-hosted page. The backend's
   CORS is wide open (`allow_origins=["*"]`) since this is a stateless, public, read-only API with
   no auth and nothing sensitive to protect.

---

## Honest limitations

- **Every training row already trended.** The model ranks *within* winners. It cannot tell you
  whether you'll break into Trending in the first place — nothing could, from this data.
- **One snapshot day** (2026-07-27), publish dates spanning ~37 days. No seasonality is learnable.
- **Category imbalance is severe**: Gaming 3,895 and Music 1,684 of 6,668 rows; Pets & Animals has
  exactly 1. Charts and advice suppress categories with fewer than 30 examples.
- **Correlation, not causation.** Every suggestion describes what tends to co-occur with high
  placement among videos that already trended. Adding brackets to your title does not cause views.
- **No thumbnail, no audio, no watch-time** — the three things that probably matter most are all
  absent from the dataset. Given that, an AUC around 0.85 from metadata alone is close to the
  realistic ceiling for this problem, not a shortfall — published work on YouTube virality from
  metadata typically lands in the 0.75–0.85 range.
- The channel-size dropdown's tiers are labeled by **percentile among channels that already
  trended** ("Bottom 10%," "Median," "Top 10%"), not universal small/medium/large labels — even
  the 10th-percentile trending channel has ~18K subscribers, so a label like "Nano (under 1K
  subs)" would have been false. Anyone testing a genuinely tiny channel still can, via the custom
  fields.

---

## The v1 pipeline

`cleaning.py`, `eda.py`, `featureExtraction.py`, `preprocess.py`, `training1.py` and `explaine.py`
are the original exploratory pipeline and are left untouched at the repo root. v2 reuses
`cleaned_data.csv` from it and ports the feature-extraction logic verbatim.
