import pickle
import uuid
import numpy as np
import cv2
from pathlib import Path
from insightface.app import FaceAnalysis
import onnxruntime as ort

ort.set_default_logger_severity(3)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
KNOWN_FACES_DIR = DATA_DIR / "known_faces"
UPLOADS_DIR = DATA_DIR / "uploads"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.pkl"

KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


class FaceEngine:
    def __init__(self, det_size=(640, 640), det_thresh=0.3):
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=-1, det_size=det_size, det_thresh=det_thresh)
        self.known_embeddings: dict[str, list[np.ndarray]] = {}
        self.similarity_threshold = 0.4
        self._load_embeddings()

    def _load_embeddings(self):
        if EMBEDDINGS_FILE.exists():
            with open(EMBEDDINGS_FILE, "rb") as f:
                self.known_embeddings = pickle.load(f)

    def _save_embeddings(self):
        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(self.known_embeddings, f)

    def detect_faces(self, image: np.ndarray) -> list[dict]:
        faces = self.app.get(image)
        results = []
        for face in faces:
            bbox = face.bbox.astype(int).tolist()
            result = {
                "bbox": bbox,
                "confidence": float(face.det_score),
                "age": int(face.age) if face.age is not None else None,
                "gender": "M" if face.gender == 1 else "F" if face.gender is not None else None,
            }
            if face.embedding is not None:
                result["embedding"] = face.embedding
            results.append(result)
        return results

    def register_face(self, name: str, image: np.ndarray) -> dict:
        faces = self.detect_faces(image)
        if not faces:
            return {"success": False, "message": "No face detected in the image"}
        if len(faces) > 1:
            faces = sorted(
                faces,
                key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]),
                reverse=True,
            )

        embedding = faces[0].get("embedding")
        if embedding is None:
            return {"success": False, "message": "Could not extract face embedding"}

        if name not in self.known_embeddings:
            self.known_embeddings[name] = []
        self.known_embeddings[name].append(embedding)
        self._save_embeddings()

        save_path = KNOWN_FACES_DIR / f"{name}_{len(self.known_embeddings[name])}.jpg"
        cv2.imwrite(str(save_path), image)

        return {
            "success": True,
            "message": f"Face registered for '{name}' ({len(self.known_embeddings[name])} samples)",
            "face_count": len(faces),
            "bbox": faces[0]["bbox"],
        }

    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)
        return float(np.dot(emb1_norm, emb2_norm))

    def identify_face(self, embedding: np.ndarray) -> tuple[str | None, float]:
        best_name = None
        best_score = -1.0
        for name, embeddings in self.known_embeddings.items():
            for known_emb in embeddings:
                score = self._compute_similarity(embedding, known_emb)
                if score > best_score:
                    best_score = score
                    best_name = name
        if best_score >= self.similarity_threshold:
            return best_name, best_score
        return None, best_score

    def recognize_image(self, image: np.ndarray) -> list[dict]:
        faces = self.detect_faces(image)
        results = []
        for face in faces:
            embedding = face.get("embedding")
            identity = None
            similarity = 0.0
            if embedding is not None and self.known_embeddings:
                identity, similarity = self.identify_face(embedding)
            results.append({
                "bbox": face["bbox"],
                "confidence": face["confidence"],
                "age": face["age"],
                "gender": face["gender"],
                "identity": identity,
                "similarity": round(similarity, 3),
            })
        return results

    def recognize_video(self, video_path: str, output_path: str, frame_skip: int = 2) -> dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"success": False, "message": "Could not open video file"}

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        processed = 0
        total_faces = 0
        last_results = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % (frame_skip + 1) == 0:
                last_results = self.recognize_image(frame)
                processed += 1
            total_faces += len(last_results)
            annotated = self.annotate_frame(frame, last_results)
            out.write(annotated)
            frame_idx += 1

        cap.release()
        out.release()

        return {
            "success": True,
            "total_frames": frame_idx,
            "processed_frames": processed,
            "avg_faces_per_frame": round(total_faces / max(processed, 1), 1),
            "output_path": output_path,
        }

    def annotate_frame(self, frame: np.ndarray, faces: list[dict]) -> np.ndarray:
        annotated = frame.copy()
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            identity = face.get("identity")
            similarity = face.get("similarity", 0)
            confidence = face.get("confidence", 0)

            color = (0, 200, 0) if identity else (0, 140, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label_parts = []
            if identity:
                label_parts.append(f"{identity} ({similarity:.0%})")
            else:
                label_parts.append("Unknown")
            label_parts.append(f"{confidence:.0%}")
            if face.get("age"):
                label_parts.append(f"{face['age']}y")
            if face.get("gender"):
                label_parts.append(face["gender"])

            label = " | ".join(label_parts)
            font_scale = 0.5
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        return annotated

    def get_registered_names(self) -> list[dict]:
        return [
            {"name": name, "samples": len(embs)}
            for name, embs in self.known_embeddings.items()
        ]

    def delete_face(self, name: str) -> bool:
        if name in self.known_embeddings:
            del self.known_embeddings[name]
            self._save_embeddings()
            for f in KNOWN_FACES_DIR.glob(f"{name}_*"):
                f.unlink()
            return True
        return False
