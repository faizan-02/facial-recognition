# HydraBytes Face Recognition System v2.0

Production-grade, CPU-optimized face recognition pipeline built for **real-world public-place video** — small faces, blur, multi-angle, CCTV footage.

**Stack:** InsightFace (SCRFD + ArcFace) | ONNX Runtime CPU | FastAPI | Vanilla JS

---

## Key Capabilities

| Requirement | How It's Handled |
|---|---|
| **CPU-only, highly optimized** | ONNX Runtime with thread tuning (`intra_op_num_threads`, graph optimization), no GPU dependency |
| **Small faces in public video** | Multi-scale detection: high-res (1280x1280) + image upscaling + tiled detection with 2x zoom per quadrant |
| **Blurred faces** | CLAHE contrast enhancement + bilateral denoising + sharpening pre-processing |
| **Any-angle detection** | SCRFD handles yaw/pitch up to ~90 degrees; tiled detection catches edge cases |
| **Any-angle recognition** | Multi-pose enrollment (frontal + left + right), pose-aware similarity matching, top-k averaging |
| **Video processing** | IoU-based face tracking across frames, identity persistence, async processing with progress |

## Architecture

```
Input Image/Frame
    │
    ├── CLAHE Enhancement + Bilateral Denoise + Sharpen
    │
    ├── Scale 1: SCRFD @ 1280x1280 (high-res detection)
    ├── Scale 2: Upscale to 1920px + SCRFD (catches very small faces)
    ├── Scale 3: 4-tile detection with 2x zoom (catches tiny faces in large images)
    │
    ├── NMS Merge (removes duplicate detections across scales)
    │
    ├── ArcFace 512-d Embedding per face
    │
    ├── Pose-Aware Matching (top-k avg, pose bonus)
    │
    └── Identity + Confidence + Age + Gender + Pose
```

### Video Pipeline

```
Frame → Enhance → SCRFD Detect → ArcFace Embed → Identify → IoU Tracker → Annotate
                                                                │
                                                    Identity persists across
                                                    frames via tracking
```

## Quick Start (Local)

```bash
git clone https://github.com/faizan-02/facial-recognition.git
cd facial-recognition

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r backend/requirements.txt

cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000

## Deploy to Railway

1. Push to GitHub
2. [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Select this repo — auto-detects Dockerfile
4. First build ~5 min (downloads InsightFace models ~300MB)

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Status + identity/sample counts |
| POST | `/api/register` | Register face (name + single image) |
| POST | `/api/register/multi` | Register face (name + multiple images for multi-angle) |
| POST | `/api/recognize/image` | Multi-scale detect & identify in image |
| POST | `/api/recognize/video` | Async video processing (returns job_id) |
| GET | `/api/recognize/video/{job_id}` | Poll video processing progress |
| GET | `/api/faces` | List registered faces with pose coverage |
| DELETE | `/api/faces/{name}` | Remove identity |
| GET | `/api/files/{filename}` | Download processed results |

## Optimizations

- **ONNX Runtime**: `ORT_ENABLE_ALL` graph optimization, thread count auto-tuned to CPU cores
- **Multi-scale detection**: 3 detection passes catch faces from 10x10px to full frame
- **Image enhancement**: CLAHE + bilateral filter + sharpening before detection
- **Frame skip**: Configurable for speed/accuracy tradeoff in video
- **Face tracking**: IoU-based tracker avoids redundant recognition on every frame
- **Pose-aware matching**: Bonus for same-pose comparison, top-k averaging for robustness

## Built by [HydraBytes](https://www.hydrabytes.tech)
