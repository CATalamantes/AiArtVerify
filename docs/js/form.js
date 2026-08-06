/* Form wiring: populates selects from /reference, the click-to-add tag chip
   picker, the channel-size preset <-> custom toggle, the horizon segmented
   control, and building the /predict request payload. */

const App = {
  reference: null,
  selectedTags: new Set(),
  horizonHours: 48,
};
// `const` doesn't attach to `window` in a classic script, but main.js's
// renderHeroChart() checks `window.App` -- expose it explicitly so that
// check actually sees the object instead of always bailing out.
window.App = App;

const els = {
  category: document.getElementById("category"),
  channelSize: document.getElementById("channel-size"),
  channelCountry: document.getElementById("channel-country"),
  customChannelFields: document.getElementById("custom-channel-fields"),
  publishHour: document.getElementById("publish-hour"),
  publishHourValue: document.getElementById("publish-hour-value"),
  horizonGroup: document.getElementById("horizon-group"),
  tagChipPool: document.getElementById("tag-chip-pool"),
  tagChipSelected: document.getElementById("tag-chip-selected"),
  tagsCustom: document.getElementById("tags-custom"),
  publishDate: document.getElementById("publish-date"),
  formError: document.getElementById("form-error"),
};

async function initForm() {
  App.reference = await Api.getReference();

  els.category.innerHTML = App.reference.categories
    .map((c) => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`)
    .join("");
  if (App.reference.categories.includes("Gaming")) els.category.value = "Gaming";

  els.channelCountry.innerHTML = App.reference.channel_countries
    .map((c) => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`)
    .join("");
  if (App.reference.channel_countries.includes("United States")) els.channelCountry.value = "United States";

  const presetNames = Object.keys(App.reference.channel_presets);
  els.channelSize.innerHTML =
    presetNames.map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(App.reference.channel_presets[name].display)}</option>`).join("") +
    `<option value="__custom__">Custom...</option>`;
  // Default to the median tier -- the most representative "typical trending
  // channel" starting point.
  if (presetNames.includes("Median")) els.channelSize.value = "Median";

  els.publishDate.valueAsDate = new Date();

  await loadTagPool(els.category.value);

  els.category.addEventListener("change", () => loadTagPool(els.category.value));
  els.channelSize.addEventListener("change", onChannelSizeChange);
  els.publishHour.addEventListener("input", () => {
    els.publishHourValue.textContent = `${String(els.publishHour.value).padStart(2, "0")}:00`;
  });
  els.horizonGroup.addEventListener("click", onHorizonClick);
  els.tagsCustom.addEventListener("keydown", onTagsCustomKeydown);
  els.tagsCustom.addEventListener("blur", () => mergeCustomTagsInput());

  onChannelSizeChange();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

/* -------------------------- tag chip picker ---------------------------- */

// Exact on/off colors from the Claude Design reference's suggestedTags binding
// (Hit or Flop Landing.dc.html renderVals()) -- kept as literal oklch values,
// not the --border/--ink-2 tokens, because those tokens are tuned for card
// borders (0.88 lightness) and don't match the reference's control-level
// border (0.85) or off-state text (0.4) used specifically for chips.
function chipStyle(isOn) {
  const bg = isOn ? "oklch(0.75 0.15 220)" : "oklch(0.97 0.008 260)";
  const color = isOn ? "oklch(0.99 0.005 260)" : "oklch(0.4 0.02 260)";
  const border = isOn ? "oklch(0.75 0.15 220)" : "oklch(0.85 0.012 260)";
  return `display: inline-flex; align-items: center; gap: 6px; background: ${bg}; color: ${color}; border: 1px solid ${border}; border-radius: 999px; padding: 6px 12px; font-size: 12.5px; cursor: pointer; font-family: inherit;`;
}
function chipWeightColor(isOn) {
  return isOn ? "oklch(0.9 0.03 220)" : "oklch(0.6 0.02 260)";
}

async function loadTagPool(category) {
  els.tagChipPool.innerHTML = `<span class="hint">Loading tags…</span>`;
  const { tags } = await Api.getTags(category, 20);
  els.tagChipPool.innerHTML = "";
  tags.forEach((t) => {
    const chip = document.createElement("button");
    chip.type = "button";
    const isOn = App.selectedTags.has(t.tag);
    chip.className = "chip" + (isOn ? " is-selected" : "");
    chip.style.cssText = chipStyle(isOn);
    chip.innerHTML = `${escapeHtml(t.tag)} <span style="font-size: 11px; color: ${chipWeightColor(isOn)};">${t.lift.toFixed(1)}&times;</span>`;
    chip.addEventListener("click", () => toggleTag(t.tag, chip));
    els.tagChipPool.appendChild(chip);
  });
  if (!tags.length) {
    els.tagChipPool.innerHTML = `<span class="hint">No mined tags for this category yet -- just type your own below.</span>`;
  }
}

function toggleTag(tag, chipEl) {
  if (App.selectedTags.has(tag)) {
    App.selectedTags.delete(tag);
    if (chipEl) {
      chipEl.classList.remove("is-selected");
      chipEl.style.cssText = chipStyle(false);
    }
  } else {
    App.selectedTags.add(tag);
    if (chipEl) {
      chipEl.classList.add("is-selected");
      chipEl.style.cssText = chipStyle(true);
    }
  }
  renderSelectedTags();
}

function renderSelectedTags() {
  els.tagChipSelected.innerHTML = "";
  App.selectedTags.forEach((tag) => {
    const chip = document.createElement("span");
    chip.className = "chip is-selected";
    chip.style.cssText = chipStyle(true);
    chip.innerHTML = `${escapeHtml(tag)} <span style="font-weight: 700; opacity: 0.85;">&times;</span>`;
    chip.addEventListener("click", () => {
      App.selectedTags.delete(tag);
      renderSelectedTags();
      // reflect removal in the pool chips too, if the tag is currently shown there
      els.tagChipPool.querySelectorAll(".chip").forEach((c) => {
        if (c.textContent.trim().startsWith(tag)) {
          c.classList.remove("is-selected");
          c.style.cssText = chipStyle(false);
        }
      });
    });
    els.tagChipSelected.appendChild(chip);
  });
}

function mergeCustomTagsInput() {
  const raw = els.tagsCustom.value;
  if (!raw.trim()) return;
  raw.split(",").map((t) => t.trim().toLowerCase()).filter(Boolean).forEach((t) => App.selectedTags.add(t));
  els.tagsCustom.value = "";
  renderSelectedTags();
}

function onTagsCustomKeydown(e) {
  if (e.key === "Enter" || e.key === ",") {
    e.preventDefault();
    mergeCustomTagsInput();
  }
}

/* -------------------------- channel size preset ------------------------ */

function onChannelSizeChange() {
  const isCustom = els.channelSize.value === "__custom__";
  els.customChannelFields.classList.toggle("hidden", !isCustom);
}

/* -------------------------- horizon segmented control ------------------- */

// Exact on/off colors from the reference's windowOptions binding (opt() in
// renderVals()) -- same reasoning as chipStyle() above.
function segmentedColors(isOn) {
  return {
    background: isOn ? "oklch(0.75 0.15 220)" : "oklch(0.99 0.005 260)",
    color: isOn ? "oklch(0.99 0.005 260)" : "oklch(0.4 0.02 260)",
    borderColor: isOn ? "oklch(0.75 0.15 220)" : "oklch(0.85 0.012 260)",
  };
}

function onHorizonClick(e) {
  const btn = e.target.closest(".segmented-option");
  if (!btn) return;
  App.horizonHours = Number(btn.dataset.value);
  els.horizonGroup.querySelectorAll(".segmented-option").forEach((b) => {
    const isOn = b === btn;
    b.classList.toggle("is-selected", isOn);
    Object.assign(b.style, segmentedColors(isOn));
  });
}

/* -------------------------- payload ------------------------------------- */

function buildPredictPayload() {
  mergeCustomTagsInput(); // catch anything left in the free-text box unsubmitted

  const title = document.getElementById("title").value.trim();
  if (!title) {
    throw new Error("Give it a title -- even a rough one -- so there's something to score.");
  }

  const payload = {
    title,
    description: document.getElementById("description").value,
    tags: Array.from(App.selectedTags).join(", "),
    category: els.category.value,
    duration_minutes: Number(document.getElementById("duration").value) || 10,
    publish_date: els.publishDate.value,
    publish_hour: Number(els.publishHour.value),
    horizon_hours: App.horizonHours,
    channel_country: els.channelCountry.value,
  };

  if (els.channelSize.value === "__custom__") {
    payload.subscribers = Number(document.getElementById("subscribers").value) || 0;
    payload.channel_views = Number(document.getElementById("channel-views").value) || 0;
    payload.channel_videos = Number(document.getElementById("channel-videos").value) || 0;
    payload.channel_age_days = Number(document.getElementById("channel-age").value) || 0;
  } else {
    payload.channel_size_preset = els.channelSize.value;
  }

  return payload;
}
