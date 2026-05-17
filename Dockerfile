FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg \
    g++ build-essential

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove g++ build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY data/ ./data/

RUN python -c "\
from insightface.app import FaceAnalysis; \
a = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); \
a.prepare(ctx_id=-1, det_size=(640,640)); \
b = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); \
b.prepare(ctx_id=-1, det_size=(1280,1280))"

ENV PORT=8000
EXPOSE 8000

CMD sh -c "python -m uvicorn backend.main:app --host 0.0.0.0 --port $PORT"
