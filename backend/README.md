---
title: Hit or Flop API
emoji: 🔥
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

FastAPI backend for [Hit or Flop](https://github.com/CATalamantes/HitOrFlop) — a pre-publish
YouTube performance predictor. The frontend (GitHub Pages, static HTML/CSS/JS) calls this API's
`/predict`, `/reference`, and `/tags/{category}` endpoints via `fetch()`.

This Space has no UI of its own — hit `/health` to confirm it's awake, or see the full project
README in the main repo for the model, the two evaluation gates, and how everything fits together.
