FROM python:3.12-slim
WORKDIR /app
COPY backend/pyproject.toml ./
RUN pip install --no-cache-dir -e ".[providers]" || pip install --no-cache-dir .
COPY backend/app ./app
ENV RT_CORPUS_DIR=/data/corpus RT_TELEMETRY_URL=sqlite:////data/telemetry.db
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
