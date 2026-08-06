/* Renders a /predict response into the results section. */

const BAND_COLOR_VAR = {
  FLOP: "--critical",
  MID: "--serious",
  HIT: "--warning",
  VIRAL: "--good",
};

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function renderResults(data, formSnapshot) {
  const { prediction, narrative, recommendations, best_time, category_fit, growth_curve, missing_tags } = data;

  const bandColorVar = BAND_COLOR_VAR[prediction.band] || "--violet";
  const bandColor = cssVar(bandColorVar);
  document.documentElement.style.setProperty("--band-color", bandColor);

  document.getElementById("verdict-number").textContent = prediction.hit_score;
  document.getElementById("verdict-band").textContent = prediction.band;
  document.getElementById("base-rate-inline").textContent = Math.round(prediction.base_rate * 100);

  renderGauge(document.getElementById("gauge-svg"), {
    score: prediction.hit_score,
    bandColor,
    baseRate: prediction.base_rate * 100,
  });

  renderTiles(prediction);

  renderViewsHistogram(document.getElementById("histogram-svg"), {
    distLogViews: App.reference.dist_log_views,
    myViews: prediction.views,
  });

  renderGrowthCurve(document.getElementById("growth-svg"), {
    hours: growth_curve.hours,
    views: growth_curve.views,
    horizonHours: formSnapshot.horizon_hours,
  });

  const HORIZON_LABEL = { 24: "24 hours", 48: "48 hours", 168: "7 days" };
  document.getElementById("growth-note").textContent =
    `Views projected across the ${HORIZON_LABEL[formSnapshot.horizon_hours] || formSnapshot.horizon_hours + " hours"} you selected.`;

  const dayIdx = new Date(formSnapshot.publish_date + "T00:00:00Z").getUTCDay();
  // JS getUTCDay(): 0=Sunday..6=Saturday; backend DAYS starts Monday=0 -- convert.
  const mondayFirstIdx = (dayIdx + 6) % 7;
  renderHeatmap(document.getElementById("heatmap-svg"), {
    grid: best_time.grid,
    days: best_time.days,
    currentDay: mondayFirstIdx,
    currentHour: formSnapshot.publish_hour,
  });
  const gainPts = Math.round(best_time.gain * 100);
  document.getElementById("best-time-note").textContent =
    gainPts > 0
      ? `Peak: ${best_time.best_day} ${String(best_time.best_hour).padStart(2, "0")}:00 UTC, worth +${gainPts} points over your current slot.`
      : `You're already at (or near) the best time for this video.`;

  renderNarrative(narrative);
  renderAdvice(recommendations);
  renderMissingTags(missing_tags);

  renderCategoryBars(document.getElementById("category-svg"), {
    categories: category_fit,
    currentCategory: formSnapshot.category,
  });
}

function renderTiles(prediction) {
  const tiles = [
    { label: "Views", value: humanNumber(prediction.views), pct: prediction.views_percentile },
    { label: "Likes", value: humanNumber(prediction.likes), pct: prediction.likes_percentile },
    { label: "Comments", value: humanNumber(prediction.comments), pct: prediction.comments_percentile },
  ];
  const row = document.getElementById("tiles-row");
  row.innerHTML = tiles.map((t) => `
    <div class="tile">
      <div class="tile-label">${t.label}</div>
      <div class="tile-value">${t.value}</div>
      <div class="tile-percentile">beats ${Math.round(t.pct)}% of trending</div>
    </div>
  `).join("");
}

function renderNarrative(lines) {
  const list = document.getElementById("narrative-list");
  if (!lines.length) {
    list.innerHTML = `<li>Nothing stands out as a strong driver either way for this one.</li>`;
    return;
  }
  list.innerHTML = lines.map((l) => `<li>${escapeHtml(l)}</li>`).join("");
}

function renderAdvice(recs) {
  const container = document.getElementById("advice-list");
  if (!recs.length) {
    container.innerHTML = `<p class="advice-empty">No suggestion passed the cross-check for this video.
      Every tip has to be backed by both the model and the raw data before it shows up here, so an empty
      list means the honest answer is "nothing reliable to change."</p>`;
    return;
  }
  container.innerHTML = recs.map((r) => `
    <div class="advice-item">
      <span class="advice-points">+${r.points} pts</span><span class="advice-label">${escapeHtml(r.label)}</span>
      <div class="advice-detail">
        ${escapeHtml(String(r.current))} &rarr; <b>${escapeHtml(String(r.suggested))}</b>
        &middot; confirmed by ${r.support.toLocaleString()} real videos
      </div>
    </div>
  `).join("");
}

function renderMissingTags(tags) {
  const container = document.getElementById("missing-tags-chips");
  if (!tags.length) {
    container.innerHTML = `<span class="hint">No high-lift tags left for this category. You've got the good ones.</span>`;
    return;
  }
  // Static chip look straight from the reference's missingTags binding
  // (line 333 of the Claude Design reference) -- these don't toggle, so
  // unlike the pool/selected chips there's no on/off state to compute.
  container.innerHTML = tags.map((t) => `
    <span style="display: inline-flex; align-items: center; gap: 5px; background: oklch(0.99 0.005 260); border: 1px solid oklch(0.85 0.012 260); border-radius: 999px; padding: 5px 11px; font-size: 12px; color: oklch(0.4 0.02 260);">${escapeHtml(t.tag)} <span style="font-size: 10.5px; color: oklch(0.6 0.02 260);">${t.lift.toFixed(1)}&times;</span></span>
  `).join("");
}
