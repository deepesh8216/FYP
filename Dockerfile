# Week 8: container for the web API. Mount your checkpoint at runtime.
# Build:  docker build -t fyp-fakenews .
# Run (Week 5 checkpoint example):
#   docker run --rm -p 8000:8000 \
#     -v "$(pwd)/outputs/week5/attention/seed_1337/best_model.pt:/model.pt:ro" \
#     -e FAKE_NEWS_CHECKPOINT=/model.pt \
#     fyp-fakenews
# Week 6: mount .../outputs/week6/finetune_attention/seed_1337/best_model.pt instead.

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

ENV PYTHONPATH=/app
ENV FAKE_NEWS_CHECKPOINT=/model.pt

EXPOSE 8000

CMD ["uvicorn", "src.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
