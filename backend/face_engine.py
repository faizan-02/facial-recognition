import os
import pickle
import numpy as np
import cv2
from pathlib import Path
from insightface.app import FaceAnalysis
import onnxruntime as ort

ort.set_default_logger_severity(3)

NUM_THREADS = max(1, os.cpu_count() or 4)
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
KNOWN_FACES_DIR = DATA_DIR / "known_faces"
UPLOADS_DIR = DATA_DIR / "uploads"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.pkl"

KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def enhance_image(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=40, sigmaSpace=40)
    return enhanced


def sharpen_image(image: np.ndarray) -> np.ndarray:
    kernel = np.array([
        [0, -0.5, 0],
        [-0.5, 3, -0.5],
        [0, -0.5, 0],
    ], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)


def upscale_image(image: np.ndarray, target_short_edge: int = 1280) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    short_edge = min(h, w)
    if short_edge >= target_short_edge:
        return image, 1.0
    scale = target_short_edge / short_edge
    scale = min(scale, 3.0)
    new_w = int(w * scale)
    new_h = int(h * scale)
    upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return upscaled, scale


def nms_boxes(detections: list[dict], iou_thresh: float = 0.4) -> list[dict]:
    if not detections:
        return []
    boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
    scores = np.array([d["confidence"] for d in detections], dtype=np.float32)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        remaining = np.where(iou <= iou_thresh)[0]
        order = order[remaining + 1]

    return [detections[i] for i in keep]


class FaceTracker:
    def __init__(self, max_disappeared: int = 15, iou_thresh: float = 0.3):
        self.next_id = 0
        self.tracks: dict[int, dict] = {}
        self.disappeared: dict[int, int] = {}
        self.max_disappeared = max_disappeared
        self.iou_thresh = iou_thresh

    def _iou(self, box_a, box_b):
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    def update(self, faces: list[dict]) -> list[dict]:
        if not self.tracks:
            for face in faces:
                face["track_id"] = self.next_id
                self.tracks[self.next_id] = face.copy()
                self.disappeared[self.next_id] = 0
                self.next_id += 1
            return faces

        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[tid]["bbox"] for tid in track_ids]

        if not faces:
            for tid in list(self.disappeared.keys()):
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    del self.tracks[tid]
                    del self.disappeared[tid]
            return []

        det_boxes = [f["bbox"] for f in faces]
        iou_matrix = np.zeros((len(track_boxes), len(det_boxes)))
        for i, tb in enumerate(track_boxes):
            for j, db in enumerate(det_boxes):
                iou_matrix[i, j] = self._iou(tb, db)

        matched_tracks = set()
        matched_dets = set()
        results = []

        while True:
            if iou_matrix.size == 0:
                break
            max_iou = iou_matrix.max()
            if max_iou < self.iou_thresh:
                break
            i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            tid = track_ids[i]

            faces[j]["track_id"] = tid
            if faces[j].get("identity") is None and self.tracks[tid].get("identity") is not None:
                faces[j]["identity"] = self.tracks[tid]["identity"]
                faces[j]["similarity"] = self.tracks[tid]["similarity"]

            self.tracks[tid] = faces[j].copy()
            self.disappeared[tid] = 0
            matched_tracks.add(i)
            matched_dets.add(j)
            iou_matrix[i, :] = 0
            iou_matrix[:, j] = 0

        for j, face in enumerate(faces):
            if j not in matched_dets:
                face["track_id"] = self.next_id
                self.tracks[self.next_id] = face.copy()
                self.disappeared[self.next_id] = 0
                self.next_id += 1

        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    del self.tracks[tid]
                    del self.disappeared[tid]

        return faces


class FaceEngine:
    def __init__(self, det_thresh=0.25):
        self.app_highres = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition", "genderage"],
        )
        self.app_highres.prepare(
            ctx_id=-1,
            det_size=(1280, 1280),
            det_thresh=det_thresh,
        )

        self.app_lowres = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition", "genderage"],
        )
        self.app_lowres.prepare(
            ctx_id=-1,
            det_size=(640, 640),
            det_thresh=det_thresh,
        )

        self.known_embeddings: dict[str, list[np.ndarray]] = {}
        self.known_poses: dict[str, list[str]] = {}
        self.similarity_threshold = 0.35
        self._load_embeddings()

    def _load_embeddings(self):
        if EMBEDDINGS_FILE.exists():
            with open(EMBEDDINGS_FILE, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict) and "embeddings" in data:
                self.known_embeddings = data["embeddings"]
                self.known_poses = data.get("poses", {})
            else:
                self.known_embeddings = data
                self.known_poses = {}

    def _save_embeddings(self):
        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump({
                "embeddings": self.known_embeddings,
                "poses": self.known_poses,
            }, f)

    def _estimate_pose(self, face) -> str:
        if face.pose is None and (not hasattr(face, "landmark_2d_106") or face.landmark_2d_106 is None):
            kps = getattr(face, "kps", None)
            if kps is not None and len(kps) >= 5:
                left_eye = kps[0]
                right_eye = kps[1]
                nose = kps[2]
                eye_center_x = (left_eye[0] + right_eye[0]) / 2
                eye_dist = abs(right_eye[0] - left_eye[0])
                if eye_dist < 1:
                    return "frontal"
                nose_offset = (nose[0] - eye_center_x) / eye_dist
                if abs(nose_offset) < 0.15:
                    return "frontal"
                elif nose_offset < -0.15:
                    return "left"
                else:
                    return "right"
            return "frontal"

        if face.pose is not None:
            yaw = face.pose[1] if len(face.pose) > 1 else 0
            pitch = face.pose[0] if len(face.pose) > 0 else 0
            if abs(yaw) < 20:
                return "frontal"
            elif yaw < -20:
                return "left"
            elif yaw > 20:
                return "right"
            if pitch < -20:
                return "up"
            elif pitch > 20:
                return "down"
        return "frontal"

    def detect_faces_multiscale(self, image: np.ndarray, use_enhance: bool = True) -> list[dict]:
        if use_enhance:
            image = enhance_image(image)

        all_detections = []

        faces_hr = self.app_highres.get(image)
        for face in faces_hr:
            all_detections.append(self._face_to_dict(face))

        h, w = image.shape[:2]
        if min(h, w) > 500:
            upscaled, scale = upscale_image(image, target_short_edge=1920)
            if scale > 1.1:
                sharpened = sharpen_image(upscaled)
                faces_up = self.app_highres.get(sharpened)
                for face in faces_up:
                    d = self._face_to_dict(face)
                    d["bbox"] = [
                        int(d["bbox"][0] / scale),
                        int(d["bbox"][1] / scale),
                        int(d["bbox"][2] / scale),
                        int(d["bbox"][3] / scale),
                    ]
                    all_detections.append(d)

        tile_detections = self._detect_tiles(image)
        all_detections.extend(tile_detections)

        merged = nms_boxes(all_detections, iou_thresh=0.4)
        return merged

    def _detect_tiles(self, image: np.ndarray) -> list[dict]:
        h, w = image.shape[:2]
        if max(h, w) < 1000:
            return []

        results = []
        tile_h, tile_w = h // 2, w // 2
        overlap = 100

        tiles = [
            (0, 0, tile_w + overlap, tile_h + overlap),
            (tile_w - overlap, 0, w, tile_h + overlap),
            (0, tile_h - overlap, tile_w + overlap, h),
            (tile_w - overlap, tile_h - overlap, w, h),
        ]

        for (tx1, ty1, tx2, ty2) in tiles:
            tx1, ty1 = max(0, tx1), max(0, ty1)
            tx2, ty2 = min(w, tx2), min(h, ty2)
            tile = image[ty1:ty2, tx1:tx2]

            tile_up = cv2.resize(tile, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            tile_up = sharpen_image(tile_up)

            faces = self.app_lowres.get(tile_up)
            for face in faces:
                d = self._face_to_dict(face)
                d["bbox"] = [
                    int(d["bbox"][0] / 2.0) + tx1,
                    int(d["bbox"][1] / 2.0) + ty1,
                    int(d["bbox"][2] / 2.0) + tx1,
                    int(d["bbox"][3] / 2.0) + ty1,
                ]
                results.append(d)

        return results

    def _face_to_dict(self, face) -> dict:
        bbox = face.bbox.astype(int).tolist()
        result = {
            "bbox": bbox,
            "confidence": float(face.det_score),
            "age": int(face.age) if face.age is not None else None,
            "gender": "M" if face.gender == 1 else "F" if face.gender is not None else None,
            "pose_label": self._estimate_pose(face),
        }
        if face.embedding is not None:
            result["_embedding"] = face.embedding
        return result

    def register_face(self, name: str, image: np.ndarray) -> dict:
        enhanced = enhance_image(image)
        faces = self.app_highres.get(enhanced)
        if not faces:
            faces = self.app_highres.get(image)
        if not faces:
            return {"success": False, "message": "No face detected in the image"}

        if len(faces) > 1:
            faces = sorted(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                reverse=True,
            )

        face = faces[0]
        if face.embedding is None:
            return {"success": False, "message": "Could not extract face embedding"}

        pose_label = self._estimate_pose(face)

        if name not in self.known_embeddings:
            self.known_embeddings[name] = []
            self.known_poses[name] = []

        self.known_embeddings[name].append(face.embedding)
        self.known_poses[name].append(pose_label)
        self._save_embeddings()

        save_path = KNOWN_FACES_DIR / f"{name}_{len(self.known_embeddings[name])}.jpg"
        cv2.imwrite(str(save_path), image)

        existing_poses = set(self.known_poses[name])
        recommended = {"frontal", "left", "right"} - existing_poses
        tip = ""
        if recommended:
            tip = f" Tip: also register {', '.join(recommended)} angle(s) for better accuracy."

        return {
            "success": True,
            "message": f"Face registered for '{name}' — {pose_label} pose ({len(self.known_embeddings[name])} samples).{tip}",
            "face_count": len(faces),
            "bbox": face.bbox.astype(int).tolist(),
            "pose": pose_label,
            "total_samples": len(self.known_embeddings[name]),
            "poses_registered": list(existing_poses | {pose_label}),
        }

    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
        return float(np.dot(emb1_norm, emb2_norm))

    def identify_face(self, embedding: np.ndarray, pose_label: str = "frontal") -> tuple[str | None, float]:
        best_name = None
        best_score = -1.0

        for name, embeddings in self.known_embeddings.items():
            scores = []
            for i, known_emb in enumerate(embeddings):
                score = self._compute_similarity(embedding, known_emb)
                known_pose = self.known_poses.get(name, [])
                if i < len(known_pose) and known_pose[i] == pose_label:
                    score += 0.03
                scores.append(score)

            scores.sort(reverse=True)
            top_k = scores[:min(3, len(scores))]
            avg_score = sum(top_k) / len(top_k)

            if avg_score > best_score:
                best_score = avg_score
                best_name = name

        if best_score >= self.similarity_threshold:
            return best_name, min(best_score, 1.0)
        return None, best_score

    def recognize_image(self, image: np.ndarray) -> list[dict]:
        faces = self.detect_faces_multiscale(image)
        results = []
        for face in faces:
            embedding = face.pop("_embedding", None)
            identity = None
            similarity = 0.0
            if embedding is not None and self.known_embeddings:
                identity, similarity = self.identify_face(embedding, face.get("pose_label", "frontal"))
            results.append({
                "bbox": face["bbox"],
                "confidence": face["confidence"],
                "age": face["age"],
                "gender": face["gender"],
                "pose": face.get("pose_label", "unknown"),
                "identity": identity,
                "similarity": round(similarity, 3),
            })
        return results

    def recognize_video(
        self,
        video_path: str,
        output_path: str,
        frame_skip: int = 2,
        progress_callback=None,
    ) -> dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"success": False, "message": "Could not open video file"}

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        tracker = FaceTracker(max_disappeared=int(fps * 0.5))
        frame_idx = 0
        processed = 0
        total_faces = 0
        last_results = []
        unique_identities = set()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % (frame_skip + 1) == 0:
                enhanced = enhance_image(frame)
                raw_faces = self.app_lowres.get(enhanced)
                face_dicts = []
                for face in raw_faces:
                    d = self._face_to_dict(face)
                    embedding = d.pop("_embedding", None)
                    if embedding is not None and self.known_embeddings:
                        identity, similarity = self.identify_face(embedding, d.get("pose_label", "frontal"))
                        d["identity"] = identity
                        d["similarity"] = round(similarity, 3)
                    else:
                        d["identity"] = None
                        d["similarity"] = 0.0
                    face_dicts.append(d)

                last_results = tracker.update(face_dicts)
                processed += 1
            else:
                for track in last_results:
                    track.setdefault("identity", None)
                    track.setdefault("similarity", 0.0)

            total_faces += len(last_results)
            for r in last_results:
                if r.get("identity"):
                    unique_identities.add(r["identity"])

            annotated = self.annotate_frame(frame, last_results)
            out.write(annotated)
            frame_idx += 1

            if progress_callback and total_frames > 0:
                progress_callback(frame_idx / total_frames)

        cap.release()
        out.release()

        return {
            "success": True,
            "total_frames": frame_idx,
            "processed_frames": processed,
            "avg_faces_per_frame": round(total_faces / max(processed, 1), 1),
            "unique_identities": list(unique_identities),
            "output_path": output_path,
        }

    def annotate_frame(self, frame: np.ndarray, faces: list[dict]) -> np.ndarray:
        annotated = frame.copy()
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            identity = face.get("identity")
            similarity = face.get("similarity", 0)
            confidence = face.get("confidence", 0)

            if identity:
                color = (0, 200, 0)
            elif confidence > 0.5:
                color = (0, 140, 255)
            else:
                color = (0, 100, 200)

            thickness = max(1, min(3, (x2 - x1) // 40))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

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
            if face.get("pose") and face["pose"] != "frontal":
                label_parts.append(face["pose"])

            label = " | ".join(label_parts)
            face_width = x2 - x1
            font_scale = max(0.3, min(0.7, face_width / 200))
            thickness_text = 1
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness_text)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness_text)

        return annotated

    def get_registered_names(self) -> list[dict]:
        results = []
        for name, embs in self.known_embeddings.items():
            poses = self.known_poses.get(name, [])
            results.append({
                "name": name,
                "samples": len(embs),
                "poses": list(set(poses)),
                "coverage": self._pose_coverage(poses),
            })
        return results

    def _pose_coverage(self, poses: list[str]) -> str:
        required = {"frontal", "left", "right"}
        have = set(poses)
        missing = required - have
        if not missing:
            return "excellent"
        if len(missing) == 1:
            return "good"
        if "frontal" in have:
            return "fair"
        return "poor"

    def delete_face(self, name: str) -> bool:
        if name in self.known_embeddings:
            del self.known_embeddings[name]
            self.known_poses.pop(name, None)
            self._save_embeddings()
            for f in KNOWN_FACES_DIR.glob(f"{name}_*"):
                f.unlink()
            return True
        return False
