FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && python -c "import nltk; nltk.download('vader_lexicon')"
COPY . .
EXPOSE 8000
# serves the API and the web UI (/) on one port
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
