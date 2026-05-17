import uuid
import shutil
from pathlib import Path
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from face_engine import FaceEngine, UPLOADS_DIR

engine: FaceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = FaceEngine(det_size=(640, 640), det_thresh=0.3)
    yield


app = FastAPI(
    title="HydraBytes Face Recognition API",
    description="CPU-optimized face detection & recognition pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


def _read_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image file")
    return img


@app.get("/")
async def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "registered_faces": len(engine.known_embeddings) if engine else 0,
    }


@app.post("/api/register")
async def register_face(
    name: str = Form(...),
    file: UploadFile = File(...),
):
    contents = await file.read()
    image = _read_image(contents)
    result = engine.register_face(name, image)
    if not result["success"]:
        raise HTTPException(400, result["message"])
    return result


@app.post("/api/recognize/image")
async def recognize_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = _read_image(contents)
    faces = engine.recognize_image(image)

    annotated = engine.annotate_frame(image, faces)
    out_name = f"result_{uuid.uuid4().hex[:8]}.jpg"
    out_path = UPLOADS_DIR / out_name
    cv2.imwrite(str(out_path), annotated)

    return {
        "faces": faces,
        "total_faces": len(faces),
        "annotated_image": f"/api/files/{out_name}",
    }


@app.post("/api/recognize/video")
async def recognize_video(
    file: UploadFile = File(...),
    frame_skip: int = Form(2),
):
    suffix = Path(file.filename).suffix or ".mp4"
    in_name = f"input_{uuid.uuid4().hex[:8]}{suffix}"
    in_path = UPLOADS_DIR / in_name
    out_name = f"result_{uuid.uuid4().hex[:8]}.mp4"
    out_path = UPLOADS_DIR / out_name

    with open(in_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = engine.recognize_video(str(in_path), str(out_path), frame_skip=frame_skip)
    in_path.unlink(missing_ok=True)

    if not result["success"]:
        raise HTTPException(400, result["message"])

    result["annotated_video"] = f"/api/files/{out_name}"
    return result


@app.get("/api/files/{filename}")
async def get_file(filename: str):
    safe_name = Path(filename).name
    filepath = UPLOADS_DIR / safe_name
    if not filepath.exists():
        raise HTTPException(404, "File not found")
    media_type = "video/mp4" if safe_name.endswith(".mp4") else "image/jpeg"
    return FileResponse(str(filepath), media_type=media_type)


@app.get("/api/faces")
async def list_faces():
    return engine.get_registered_names()


@app.delete("/api/faces/{name}")
async def delete_face(name: str):
    if engine.delete_face(name):
        return {"message": f"Deleted '{name}'"}
    raise HTTPException(404, f"Face '{name}' not found")
