/* Wires the form, the API, and the results renderer together, plus the
   scroll-reveal interaction: only the form is visible on load; submitting
   reveals the results section and scrolls it to the top of the viewport so
   the verdict number + gauge become center stage. */

document.addEventListener("DOMContentLoaded", () => {
  initForm().catch((err) => showFormError(`Couldn't reach the backend: ${err.message}`));

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
});

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
