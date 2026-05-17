# HydraBytes Face Recognition System

CPU-optimized face detection & recognition pipeline built for real-world conditions: small faces, blur, multi-angle, public-place video.

**Stack:** InsightFace (SCRFD + ArcFace) | ONNX Runtime CPU | FastAPI | Vanilla JS

## Features

- **Face Detection** — SCRFD detector handles small, blurred, and angled faces
- **Face Recognition** — ArcFace embeddings with cosine similarity matching
- **Video Processing** — Frame-by-frame detection with configurable skip for speed
- **Age & Gender** — Estimated for each detected face
- **Web UI** — Register faces, upload images/videos, view annotated results
- **REST API** — Full API with Swagger docs at `/docs`
- **CPU Only** — No GPU required, optimized for deployment on standard servers

## Quick Start (Local)

```bash
# Clone the repo
git clone https://github.com/HydraBytes/facial-recognition.git
cd facial-recognition

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r backend/requirements.txt

# Run the server
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000 in your browser.

## Deploy to Railway

1. Push this repo to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Select this repo — Railway auto-detects the Dockerfile
4. Done! Railway provides a public URL

> **Note:** First deploy takes ~5 min as it downloads the InsightFace models (~300MB).

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check + registered face count |
| POST | `/api/register` | Register a face (name + image) |
| POST | `/api/recognize/image` | Detect & identify faces in an image |
| POST | `/api/recognize/video` | Process a video file |
| GET | `/api/faces` | List all registered faces |
| DELETE | `/api/faces/{name}` | Remove a registered face |
| GET | `/api/files/{filename}` | Retrieve processed results |

## Architecture

```
SCRFD (Detection) → Crop Face → ArcFace (512-d Embedding) → Cosine Similarity → Identity
```

- **Detection:** SCRFD from InsightFace buffalo_l pack — multi-scale, handles faces as small as 10x10px
- **Recognition:** ArcFace produces 512-dimensional embeddings, compared via cosine similarity (threshold: 0.4)
- **Inference:** ONNX Runtime CPUExecutionProvider — optimized for server-grade CPUs

## Built by [HydraBytes](https://www.hydrabytes.tech)
