import uuid
import shutil
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

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
executor = ThreadPoolExecutor(max_workers=2)
video_jobs: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = FaceEngine(det_thresh=0.25)
    yield
    executor.shutdown(wait=False)


app = FastAPI(
    title="HydraBytes Face Recognition API",
    description="CPU-optimized face detection & recognition for public-place video — small, blurred, multi-angle faces",
    version="2.0.0",
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
    faces = engine.get_registered_names() if engine else []
    total_samples = sum(f["samples"] for f in faces)
    return {
        "status": "ok",
        "registered_identities": len(faces),
        "total_samples": total_samples,
        "active_video_jobs": sum(1 for j in video_jobs.values() if j["status"] == "processing"),
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


@app.post("/api/register/multi")
async def register_face_multi(
    name: str = Form(...),
    files: list[UploadFile] = File(...),
):
    results = []
    for file in files:
        contents = await file.read()
        image = _read_image(contents)
        result = engine.register_face(name, image)
        results.append(result)
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    return {
        "total": len(results),
        "registered": len(successes),
        "failed": len(failures),
        "results": results,
    }


@app.post("/api/recognize/image")
async def recognize_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = _read_image(contents)
    faces = engine.recognize_image(image)

    annotated = engine.annotate_frame(image, faces)
    out_name = f"result_{uuid.uuid4().hex[:8]}.jpg"
    out_path = UPLOADS_DIR / out_name
    cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])

    return {
        "faces": faces,
        "total_faces": len(faces),
        "annotated_image": f"/api/files/{out_name}",
    }


def _process_video_sync(job_id: str, in_path: str, out_path: str, frame_skip: int):
    def progress_cb(pct):
        video_jobs[job_id]["progress"] = round(pct * 100, 1)

    try:
        result = engine.recognize_video(in_path, out_path, frame_skip=frame_skip, progress_callback=progress_cb)
        Path(in_path).unlink(missing_ok=True)
        if result["success"]:
            video_jobs[job_id]["status"] = "done"
            video_jobs[job_id]["result"] = result
        else:
            video_jobs[job_id]["status"] = "error"
            video_jobs[job_id]["error"] = result["message"]
    except Exception as e:
        video_jobs[job_id]["status"] = "error"
        video_jobs[job_id]["error"] = str(e)


@app.post("/api/recognize/video")
async def recognize_video(
    file: UploadFile = File(...),
    frame_skip: int = Form(2),
):
    suffix = Path(file.filename).suffix or ".mp4"
    job_id = uuid.uuid4().hex[:12]
    in_name = f"input_{job_id}{suffix}"
    in_path = UPLOADS_DIR / in_name
    out_name = f"result_{job_id}.mp4"
    out_path = UPLOADS_DIR / out_name

    with open(in_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    video_jobs[job_id] = {
        "status": "processing",
        "progress": 0.0,
        "result": None,
        "error": None,
        "output_file": f"/api/files/{out_name}",
    }

    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, _process_video_sync, job_id, str(in_path), str(out_path), frame_skip)

    return {"job_id": job_id, "status": "processing"}


@app.get("/api/recognize/video/{job_id}")
async def get_video_status(job_id: str):
    job = video_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    response = {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
    }
    if job["status"] == "done":
        response["result"] = job["result"]
        response["annotated_video"] = job["output_file"]
    elif job["status"] == "error":
        response["error"] = job["error"]
    return response


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
