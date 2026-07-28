/* Thin fetch wrappers around the FastAPI backend.

   Change API_BASE_URL to the deployed Hugging Face Space before publishing
   this to GitHub Pages -- e.g. "https://<user>-hitorflop.hf.space". Left as
   localhost for local development against `uvicorn api:app --reload`.
*/

const API_BASE_URL = (() => {
  if (location.hostname === "localhost" || location.hostname === "127.0.0.1") {
    return "http://localhost:8000";
  }
  // TODO: set this to your deployed Hugging Face Space URL before publishing.
  return "https://YOUR-USERNAME-hitorflop.hf.space";
})();

const Api = {
  async getReference() {
    const res = await fetch(`${API_BASE_URL}/reference`);
    if (!res.ok) throw new Error(`GET /reference failed (${res.status})`);
    return res.json();
  },

  async getTags(category, limit = 20) {
    const res = await fetch(`${API_BASE_URL}/tags/${encodeURIComponent(category)}?limit=${limit}`);
    if (!res.ok) throw new Error(`GET /tags/${category} failed (${res.status})`);
    return res.json();
  },

  async predict(payload) {
    const res = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail ? JSON.stringify(detail.detail) : `POST /predict failed (${res.status})`);
    }
    return res.json();
  },
};
