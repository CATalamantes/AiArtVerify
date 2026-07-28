# Hugging Face Spaces, Docker SDK. Build context is this backend/ directory --
# when synced to a Space, backend/ becomes that Space's repo root.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces routes traffic to this exact port regardless of app type.
EXPOSE 7860

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
