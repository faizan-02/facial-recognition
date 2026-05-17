FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    g++ build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get purge -y --auto-remove g++ build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY data/ ./data/

# Pre-download insightface models at build time
RUN python -c "from insightface.app import FaceAnalysis; app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); app.prepare(ctx_id=-1, det_size=(640,640))"

EXPOSE 8000

CMD python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
