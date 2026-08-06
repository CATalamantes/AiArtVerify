/* Wires the form, the API, and the results renderer together, plus the
   scroll-reveal interaction: only the form is visible on load; submitting
   reveals the results section and scrolls it to the top of the viewport so
   the verdict number + gauge become center stage. */

document.addEventListener("DOMContentLoaded", () => {
  initForm()
    .then(renderHeroChart)
    .catch((err) => showFormError(`Couldn't reach the backend: ${err.message}`));

  document.getElementById("video-form").addEventListener("submit", onSubmit);
  document.getElementById("edit-again-btn").addEventListener("click", onEditAgain);

  // Nav/hero CTAs that link to #predictor also carry data-focus="title" --
  // after the anchor scroll lands, put the cursor straight into the title
  // field so "Try it free" actually starts the form instead of just scrolling to it.
  document.querySelectorAll("[data-focus]").forEach((el) => {
    el.addEventListener("click", () => {
      const field = document.getElementById(el.dataset.focus);
      window.setTimeout(() => field && field.focus({ preventScroll: true }), 550);
    });
  });

  initStickyNav();
});

// Nav hides while scrolling down (so it doesn't sit over the predictor form/
// results), reappears the moment you scroll up even a little, and picks up
// a background/blur once it's no longer over the transparent hero so nav
// links stay legible against whatever section is scrolled underneath.
function initStickyNav() {
  const nav = document.getElementById("site-nav");
  const brand = document.getElementById("nav-brand");
  if (!nav) return;

  let lastY = window.scrollY;
  const HIDE_AFTER_PX = 80;

  window.addEventListener("scroll", () => {
    const y = window.scrollY;
    nav.classList.toggle("nav-scrolled", y > 10);
    if (y > lastY && y > HIDE_AFTER_PX) {
      nav.classList.add("nav-hidden");
    } else {
      nav.classList.remove("nav-hidden");
    }
    lastY = y;
  }, { passive: true });

  if (brand) {
    const scrollToTop = () => window.scrollTo({ top: 0, behavior: "smooth" });
    brand.addEventListener("click", scrollToTop);
    brand.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        scrollToTop();
      }
    });
  }
}

async function onSubmit(e) {
  e.preventDefault();
  clearFormError();

  const submitBtn = document.getElementById("submit-btn");
  let payload;
  try {
    payload = buildPredictPayload();
  } catch (err) {
    showFormError(err.message);
    return;
  }

  submitBtn.classList.add("is-loading");
  submitBtn.disabled = true;

  try {
    const data = await Api.predict(payload);
    renderResults(data, payload);
    revealResults();
  } catch (err) {
    showFormError(`Something went wrong scoring that: ${err.message}`);
  } finally {
    submitBtn.classList.remove("is-loading");
    submitBtn.disabled = false;
  }
}

function revealResults() {
  const resultsSection = document.getElementById("results-section");
  const formSection = document.getElementById("form-section");

  resultsSection.classList.remove("hidden");
  formSection.classList.add("is-collapsed");

  // Force a reflow so the removed `hidden` (display:none) takes effect
  // before the opacity/transform transition starts -- otherwise the
  // element can't animate from a display:none state.
  void resultsSection.offsetHeight;
  resultsSection.classList.add("is-visible");

  requestAnimationFrame(() => {
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  });
}

function onEditAgain() {
  const resultsSection = document.getElementById("results-section");
  const formSection = document.getElementById("form-section");

  resultsSection.classList.remove("is-visible");
  formSection.classList.remove("is-collapsed");
  setTimeout(() => resultsSection.classList.add("hidden"), 400);

  formSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function showFormError(message) {
  const el = document.getElementById("form-error");
  el.textContent = message;
  el.classList.remove("hidden");
}

function clearFormError() {
  document.getElementById("form-error").classList.add("hidden");
}

// Draws the hero's ambient background chart from the same dist_log_views
// array form.js already fetched into App.reference -- no extra API call.
function renderHeroChart() {
  const svg = document.getElementById("hero-chart-svg");
  if (!svg || !window.App || !App.reference) return;
  renderGhostHistogram(svg, App.reference.dist_log_views);
}
