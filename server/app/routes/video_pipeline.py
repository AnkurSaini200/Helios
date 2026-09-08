"""
HELIOS Video Pipeline — Multi-Model Priority Inference with Frame Division
===========================================================================

Accepts a video upload, extracts frames via OpenCV, divides frames among
models by percentage (not all models on every frame), then runs priority
cascade: Waterlogging → Accident → Pothole → Vehicle Detection.

Frame Distribution (percentage-based):
  - Waterlogging:       ~17%  of total frames
  - Accident:           ~33%  of total frames
  - Pothole:            ~33%  of total frames
  - Vehicle Detection:  ~17%  of total frames

Priority Logic:
  1. Run waterlogging on its frames → if any detection ≥ 60% conf → WINNER
  2. Run accident on its frames     → if any detection ≥ 60% conf → WINNER
  3. Run pothole on its frames      → if any detection ≥ 60% conf → WINNER
  4. Vehicle detection ALWAYS runs on its frames (results always returned)

If a higher-priority model wins, lower-priority hazard models are skipped.
"""

import asyncio
import io
import json
import os
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
# pyrefly: ignore [missing-import]
from fastapi.responses import StreamingResponse
# pyrefly: ignore [missing-import]
from PIL import Image

from app.database.session import SessionLocal
from app.database.models import BusModel, IncidentModel

router = APIRouter(prefix="/detect", tags=["video-pipeline"])

# ── In-memory job store (auto-expires after 30 min) ──
_jobs: Dict[str, Dict[str, Any]] = {}
_JOB_TTL_SEC = 1800  # 30 minutes

# ── Confidence threshold for "winner" detection ──
CONFIDENCE_THRESHOLD = 0.60

# ── Frame distribution percentages ──
FRAME_PCT = {
    "waterlogging": 0.17,
    "accident": 0.33,
    "pothole": 0.33,
    "vehicle": 0.17,
}

# ── Max video file size: 100 MB ──
MAX_VIDEO_BYTES = 100 * 1024 * 1024

# ── Target sampling FPS ──
TARGET_FPS = 2.0


# ═══════════════════════════════════════════════════════════
#  MODEL LOADING (reuse cached globals from detect.py)
# ═══════════════════════════════════════════════════════════

def _get_helios_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_waterlogging_model():
    """Load waterlogging YOLO-Seg model."""
    from app.routes.detect import get_waterlog_model
    return get_waterlog_model()


def _load_accident_model():
    """Load accident YOLO model."""
    from app.routes.detect import get_yolo_model
    return get_yolo_model()


def _load_pothole_model():
    """Load pothole YOLO model."""
    from app.routes.detect import get_pothole_model
    return get_pothole_model()


def _load_traffic_model():
    """Load vehicle/traffic YOLO model."""
    from app.routes.detect import get_traffic_model
    return get_traffic_model()


# ═══════════════════════════════════════════════════════════
#  PER-FRAME INFERENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════

def _infer_waterlogging(model, frame_bgr: np.ndarray) -> Dict[str, Any]:
    """Run waterlogging segmentation on a single frame."""
    h, w = frame_bgr.shape[:2]
    roi_top = int(h * 0.25)
    roi_bottom = int(h * 0.95)
    road_pixels = (roi_bottom - roi_top) * w

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = model.predict(source=frame_rgb, conf=0.42, imgsz=640, verbose=False)
    result = results[0]

    full_mask = np.zeros((h, w), dtype=np.uint8)
    confidence_scores = []

    if result.masks is not None and result.boxes is not None:
        boxes_conf = result.boxes.conf.cpu().numpy()
        for idx, mask_t in enumerate(result.masks.data):
            m = cv2.resize(
                mask_t.cpu().numpy().astype(np.uint8), (w, h),
                interpolation=cv2.INTER_NEAREST,
            )
            if np.count_nonzero(m[roi_top:roi_bottom, :]) >= 8000:
                full_mask = np.bitwise_or(full_mask, m)
                confidence_scores.append(float(boxes_conf[idx]))

    road_mask = full_mask[roi_top:roi_bottom, :]
    water_px = int(np.count_nonzero(road_mask))
    coverage_pct = round((water_px / road_pixels) * 100.0 if road_pixels > 0 else 0.0, 2)
    mean_conf = round(float(np.mean(confidence_scores)), 4) if confidence_scores else 0.0

    w_score = round(coverage_pct * mean_conf, 2)
    detected = coverage_pct > 3.0 and mean_conf > 0.0

    if w_score >= 40:
        severity = "critical"
    elif w_score >= 20:
        severity = "high"
    elif w_score >= 5:
        severity = "medium"
    else:
        severity = "low"

    # Create annotated frame
    overlay = frame_bgr.copy()
    overlay[full_mask == 1] = [34, 197, 94]
    annotated = cv2.addWeighted(overlay, 0.45, frame_bgr, 0.55, 0)

    return {
        "model": "waterlogging",
        "detected": detected,
        "confidence": mean_conf,
        "severity": severity,
        "coverage_pct": coverage_pct,
        "hazard_score": w_score,
        "annotated_frame": annotated,
    }


def _infer_accident(model, frame_bgr: np.ndarray) -> Dict[str, Any]:
    """Run accident detection on a single frame."""
    from app.routes.detect import ACCIDENT_CLASS_NAMES

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    results = model.predict(source=image, conf=0.25, verbose=False)
    result = results[0]
    boxes = result.boxes

    crash_detected = False
    max_conf = 0.0
    det_count = 0

    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        cls_name = result.names.get(cls_id, "")
        
        # Stricter class filtering (must match accident classes)
        valid_accident_classes = set(ACCIDENT_CLASS_NAMES.values())
        if cls_name not in valid_accident_classes and "crash" not in cls_name.lower() and "collision" not in cls_name.lower():
            continue
            
        crash_detected = True
        det_count += 1
        if conf > max_conf:
            max_conf = conf

    if max_conf >= 0.85:
        severity = "critical"
    elif max_conf >= 0.75:
        severity = "high"
    elif max_conf >= 0.50:
        severity = "medium"
    else:
        severity = "low"

    annotated_bgr = result.plot()

    return {
        "model": "accident",
        "detected": crash_detected,
        "confidence": max_conf,
        "severity": severity,
        "detections": det_count,
        "annotated_frame": annotated_bgr,
    }


def _infer_pothole(model, frame_bgr: np.ndarray) -> Dict[str, Any]:
    """Run pothole detection on a single frame."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    results = model.predict(source=image, imgsz=640, conf=0.25, verbose=False)
    result = results[0]
    boxes = result.boxes

    pothole_detected = False
    max_conf = 0.0
    det_count = 0

    if boxes is not None:
        for box in boxes:
            conf = float(box.conf[0].item())
            pothole_detected = True
            det_count += 1
            if conf > max_conf:
                max_conf = conf

    if max_conf >= 0.90:
        severity = "critical"
    elif max_conf >= 0.75:
        severity = "high"
    elif max_conf >= 0.50:
        severity = "medium"
    else:
        severity = "low"

    annotated_bgr = result.plot()

    return {
        "model": "pothole",
        "detected": pothole_detected,
        "confidence": max_conf,
        "severity": severity,
        "detections": det_count,
        "annotated_frame": annotated_bgr,
    }


def _infer_vehicle(model, frame_bgr: np.ndarray) -> Dict[str, Any]:
    """Run vehicle detection and traffic density on a single frame."""
    from app.routes.detect import VEHICLE_CLASSES, VEHICLE_PCU_WEIGHTS, _compute_traffic_density

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    results = model.predict(
        source=image,
        classes=list(VEHICLE_PCU_WEIGHTS.keys()),
        conf=0.30, iou=0.45, imgsz=1024, verbose=False,
    )
    result = results[0]
    boxes = result.boxes

    detected_vehicles = []
    breakdown = {name: 0 for name in VEHICLE_CLASSES.values()}

    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = [round(float(c), 1) for c in box.xyxy[0].tolist()]
            bw = xyxy[2] - xyxy[0]
            bh = xyxy[3] - xyxy[1]
            area = bw * bh
            cls_name = VEHICLE_CLASSES.get(cls_id, "vehicle")
            if cls_name in breakdown:
                breakdown[cls_name] += 1
            detected_vehicles.append({
                "class_id": cls_id, "class_name": cls_name,
                "confidence": round(conf, 4), "bbox": xyxy,
                "area": round(area, 1),
            })

    img_shape = frame_bgr.shape[:2]
    density_pct, total_pcu, congestion_status = _compute_traffic_density(
        detected_vehicles, img_shape
    )

    annotated_bgr = result.plot()

    return {
        "model": "vehicle_detection",
        "vehicles_detected": len(detected_vehicles),
        "density_pct": density_pct,
        "total_pcu": total_pcu,
        "congestion_status": congestion_status,
        "breakdown": breakdown,
        "annotated_frame": annotated_bgr,
    }


# ═══════════════════════════════════════════════════════════
#  VIDEO FRAME EXTRACTION
# ═══════════════════════════════════════════════════════════

def _extract_frames(video_path: str, target_fps: float = TARGET_FPS) -> List[np.ndarray]:
    """Extract frames from video at the target sampling FPS."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, int(round(native_fps / target_fps)))

    frames = []
    curr = 0
    while curr < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr)
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frames.append(frame)
        curr += frame_step

    cap.release()
    return frames


def _divide_frames(total: int) -> Dict[str, List[int]]:
    """
    Divide frame indices among models by percentage.
    Example: 60 frames → waterlogging gets ~10, accident ~20, pothole ~20, vehicle ~10.
    Uses round-robin assignment to distribute evenly across the video timeline.
    """
    n_waterlog = max(1, round(total * FRAME_PCT["waterlogging"]))
    n_accident = max(1, round(total * FRAME_PCT["accident"]))
    n_pothole = max(1, round(total * FRAME_PCT["pothole"]))
    n_vehicle = max(1, total - n_waterlog - n_accident - n_pothole)

    # Evenly space indices across the timeline for each model
    def spaced_indices(count: int) -> List[int]:
        if count >= total:
            return list(range(total))
        step = total / count
        return [min(total - 1, int(round(i * step))) for i in range(count)]

    return {
        "waterlogging": spaced_indices(n_waterlog),
        "accident": spaced_indices(n_accident),
        "pothole": spaced_indices(n_pothole),
        "vehicle": spaced_indices(n_vehicle),
    }


# ═══════════════════════════════════════════════════════════
#  BACKGROUND PROCESSING TASK
# ═══════════════════════════════════════════════════════════

def _save_incident_from_winner(job_id: str, bus_id: str, bus_gps: dict, winner_data: dict) -> str:
    """Save the detected winner as a real incident in the DB and broadcast via websocket."""
    incident_id = f"INC-{uuid.uuid4().hex[:8].upper()}"
    db = SessionLocal()
    try:
        new_inc = IncidentModel(
            id=incident_id,
            bus_id=bus_id or "BUS-HYD-VID",
            event_type=winner_data["model"],
            confidence=winner_data["confidence"],
            severity=winner_data.get("severity", "medium"),
            lat=bus_gps["lat"],
            lng=bus_gps["lng"],
            timestamp=datetime.utcnow(),
            camera="front",
            image_url=winner_data.get("annotated_image_url"),
            video_url=None,
            model="video-pipeline",
            status="detected",
            notes=f"Detected via Video Pipeline Simulator Job {job_id}",
            metadata_json=None
        )
        db.add(new_inc)
        db.commit()
        db.refresh(new_inc)
        
        # Broadcast via websocket
        from app.websocket.manager import manager
        async def _broadcast():
            await manager.broadcast("incident_created", {
                "id": new_inc.id,
                "bus_id": new_inc.bus_id,
                "event_type": new_inc.event_type,
                "confidence": new_inc.confidence,
                "severity": new_inc.severity,
                "gps": {"lat": new_inc.lat, "lng": new_inc.lng},
                "timestamp": new_inc.timestamp.isoformat(),
                "camera": new_inc.camera,
                "image_url": new_inc.image_url,
                "video_url": new_inc.video_url,
                "model": new_inc.model,
                "status": new_inc.status,
                "notes": new_inc.notes,
            })
            
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(_broadcast(), loop)
        except Exception:
            pass
            
        return incident_id
    finally:
        db.close()


def _run_pipeline(job_id: str, video_path: str):
    """
    Synchronous pipeline function run inside a thread via asyncio.to_thread.
    Updates _jobs[job_id] with progress so the SSE endpoint can stream it.
    """
    job = _jobs.get(job_id)
    if job is None:
        # Job was lost (e.g., server reload wiped _jobs). Nothing to do.
        try:
            os.remove(video_path)
        except OSError:
            pass
        return

    start = time.time()

    helios_root = _get_helios_root()
    media_dir = helios_root / "server" / "media"
    os.makedirs(media_dir, exist_ok=True)

    try:
        # Fetch actual GPS from DB based on bus_id
        bus_gps = {"lat": 17.4422, "lng": 78.3923} # default fallback
        db = SessionLocal()
        try:
            if job.get("bus_id"):
                bus = db.query(BusModel).filter(BusModel.id == job["bus_id"]).first()
                if bus:
                    bus_gps = {"lat": bus.lat, "lng": bus.lng}
        finally:
            db.close()

        # ── Phase 1: Extract frames ──
        job["phase"] = "extracting_frames"
        frames = _extract_frames(video_path)
        total = len(frames)
        if total == 0:
            job["status"] = "failed"
            job["error"] = "No frames could be extracted from video"
            return

        job["total_frames"] = total
        job["phase"] = "dividing_frames"
        division = _divide_frames(total)

        job["frame_division"] = {
            k: len(v) for k, v in division.items()
        }

        # ── Phase 2: Run models in priority order ──
        winner = None
        processed = 0

        # --- 2a. Waterlogging ---
        job["phase"] = "running_waterlogging"
        job["current_model"] = "waterlogging"
        wl_model = _load_waterlogging_model()
        best_wl = None

        for idx in division["waterlogging"]:
            res = _infer_waterlogging(wl_model, frames[idx])
            processed += 1
            job["processed_frames"] = processed

            if res["detected"] and res["confidence"] >= CONFIDENCE_THRESHOLD:
                if best_wl is None or res["confidence"] > best_wl["confidence"]:
                    best_wl = {**res, "frame_index": idx}

        if best_wl is not None:
            # Save annotated winner frame
            ts = int(time.time())
            fname = f"video_winner_wl_{ts}.jpg"
            cv2.imwrite(str(media_dir / fname), best_wl["annotated_frame"])
            best_wl.pop("annotated_frame", None)
            best_wl["annotated_image_url"] = f"http://localhost:8000/media/{fname}"
            
            # GPS for the video upload tester
            best_wl["gps"] = bus_gps
            
            # Save incident to database for map/dashboard tracking
            best_wl["incident_id"] = _save_incident_from_winner(job_id, job.get("bus_id"), bus_gps, best_wl)
            
            winner = best_wl
            job["winner"] = winner
            job["phase"] = "winner_found"
            # Skip accident and pothole
        else:
            # --- 2b. Accident ---
            job["phase"] = "running_accident"
            job["current_model"] = "accident"
            ac_model = _load_accident_model()
            best_ac = None
            
            # Temporal tracking across sampled frames
            ac_history = []

            for idx in division["accident"]:
                res = _infer_accident(ac_model, frames[idx])
                processed += 1
                job["processed_frames"] = processed
                
                conf = res.get("confidence", 0.0)
                ac_history.append(conf)

                if res["detected"] and conf >= CONFIDENCE_THRESHOLD:
                    if best_ac is None or conf > best_ac["confidence"]:
                        best_ac = {**res, "frame_index": idx}
            
            # Temporal Gating: Need at least 2 strong frames (since we are sampling 33%, a 5s video might only give us 3 frames)
            high_conf_frames = sum(1 for c in ac_history if c >= 0.70)
            avg_conf = sum(ac_history) / len(ac_history) if ac_history else 0.0
            
            if best_ac is not None and (high_conf_frames >= 2 or (len(ac_history) <= 3 and high_conf_frames >= 1)) and avg_conf >= 0.40:
                ts = int(time.time())
                fname = f"video_winner_ac_{ts}.jpg"
                if "annotated_frame" in best_ac and best_ac["annotated_frame"] is not None:
                    cv2.imwrite(str(media_dir / fname), best_ac["annotated_frame"])
                best_ac.pop("annotated_frame", None)
                best_ac["annotated_image_url"] = f"http://localhost:8000/media/{fname}"
                
                # Update confidence to reflect temporal reality
                fused_conf = (best_ac["confidence"] * 0.6) + (avg_conf * 0.4)
                best_ac["confidence"] = min(0.99, fused_conf)
                
                # GPS for the video upload tester
                best_ac["gps"] = bus_gps
                
                # Save incident to database for map/dashboard tracking
                best_ac["incident_id"] = _save_incident_from_winner(job_id, job.get("bus_id"), bus_gps, best_ac)
                
                winner = best_ac
                job["winner"] = winner
                job["phase"] = "winner_found"
                # Skip pothole
            else:
                # --- 2c. Pothole ---
                job["phase"] = "running_pothole"
                job["current_model"] = "pothole"
                ph_model = _load_pothole_model()
                best_ph = None

                for idx in division["pothole"]:
                    res = _infer_pothole(ph_model, frames[idx])
                    processed += 1
                    job["processed_frames"] = processed

                    if res["detected"] and res["confidence"] >= CONFIDENCE_THRESHOLD:
                        if best_ph is None or res["confidence"] > best_ph["confidence"]:
                            best_ph = {**res, "frame_index": idx}

                if best_ph is not None:
                    ts = int(time.time())
                    fname = f"video_winner_ph_{ts}.jpg"
                    cv2.imwrite(str(media_dir / fname), best_ph["annotated_frame"])
                    best_ph.pop("annotated_frame", None)
                    best_ph["annotated_image_url"] = f"http://localhost:8000/media/{fname}"
                    
                    # GPS for the video upload tester
                    best_ph["gps"] = bus_gps
                    
                    # Save incident to database for map/dashboard tracking
                    best_ph["incident_id"] = _save_incident_from_winner(job_id, job.get("bus_id"), bus_gps, best_ph)
                    
                    winner = best_ph
                    job["winner"] = winner
                    job["phase"] = "winner_found"

        # --- 2d. Vehicle Detection (ALWAYS runs) ---
        job["phase"] = "running_vehicle_detection"
        job["current_model"] = "vehicle_detection"
        vd_model = _load_traffic_model()

        vd_results = []
        best_vd_frame = None
        best_vd_density = -1.0

        for idx in division["vehicle"]:
            res = _infer_vehicle(vd_model, frames[idx])
            processed += 1
            job["processed_frames"] = processed
            vd_results.append(res)

            if res["density_pct"] > best_vd_density:
                best_vd_density = res["density_pct"]
                best_vd_frame = res

        # Aggregate vehicle stats
        if vd_results:
            avg_density = round(np.mean([r["density_pct"] for r in vd_results]), 2)
            avg_pcu = round(np.mean([r["total_pcu"] for r in vd_results]), 1)
            total_vehicles = sum(r["vehicles_detected"] for r in vd_results)
            agg_breakdown = {}
            for r in vd_results:
                for vtype, cnt in r["breakdown"].items():
                    agg_breakdown[vtype] = agg_breakdown.get(vtype, 0) + cnt

            # Determine overall congestion from average density
            if avg_density >= 75:
                cong = "Heavy (Congestion)"
            elif avg_density >= 40:
                cong = "Moderate"
            else:
                cong = "Low (Free Flow)"

            # Save best traffic frame
            ts = int(time.time())
            vd_fname = f"video_traffic_{ts}.jpg"
            if best_vd_frame and best_vd_frame.get("annotated_frame") is not None:
                cv2.imwrite(str(media_dir / vd_fname), best_vd_frame["annotated_frame"])

            vehicle_summary = {
                "avg_density_pct": avg_density,
                "avg_pcu": avg_pcu,
                "total_vehicles_seen": total_vehicles,
                "congestion_status": cong,
                "breakdown": agg_breakdown,
                "annotated_image_url": f"http://localhost:8000/media/{vd_fname}",
            }
        else:
            vehicle_summary = {
                "avg_density_pct": 0, "avg_pcu": 0,
                "total_vehicles_seen": 0, "congestion_status": "No Data",
                "breakdown": {}, "annotated_image_url": None,
            }

        # ── Phase 3: Finalize ──
        elapsed_ms = round((time.time() - start) * 1000, 1)

        job["status"] = "completed"
        job["phase"] = "done"
        job["current_model"] = None
        job["vehicle_summary"] = vehicle_summary
        job["processing_time_ms"] = elapsed_ms
        job["completed_at"] = datetime.utcnow().isoformat()

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        job["phase"] = "error"
    finally:
        # Clean up temp video file
        try:
            os.remove(video_path)
        except OSError:
            pass


# ═══════════════════════════════════════════════════════════
#  API ENDPOINTS
# ═══════════════════════════════════════════════════════════

@router.post("/video/upload")
async def upload_video(
    file: UploadFile = File(...),
    bus_id: Optional[str] = Form(None),
):
    """
    Upload a video file to start the multi-model priority inference pipeline.
    Returns a job_id to poll/stream status via SSE.
    """
    # Validate file type
    content_type = file.content_type or ""
    if not content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video (MP4, AVI, MOV, WebM)")

    # Read and validate size
    content = await file.read()
    if len(content) > MAX_VIDEO_BYTES:
        raise HTTPException(status_code=413, detail="Video file exceeds 100 MB limit")

    # Save to temp file OUTSIDE the server/ directory to avoid triggering
    # uvicorn --reload when the video is written to disk.
    ext = Path(file.filename or "video.mp4").suffix or ".mp4"
    job_id = f"vid_{uuid.uuid4().hex[:12]}"
    temp_fd, temp_path = tempfile.mkstemp(suffix=ext, prefix=f"{job_id}_")
    try:
        os.write(temp_fd, content)
    finally:
        os.close(temp_fd)

    # Initialize job state
    _jobs[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "phase": "initializing",
        "current_model": None,
        "total_frames": 0,
        "processed_frames": 0,
        "frame_division": {},
        "winner": None,
        "vehicle_summary": None,
        "processing_time_ms": 0,
        "error": None,
        "bus_id": bus_id or "BUS-HYD-VID",
        "filename": file.filename,
        "created_at": time.time(),
        "completed_at": None,
    }

    # Purge expired jobs (older than 30 min)
    now = time.time()
    expired = [k for k, v in _jobs.items()
               if isinstance(v.get("created_at"), (int, float))
               and (now - v["created_at"]) > _JOB_TTL_SEC]
    for k in expired:
        _jobs.pop(k, None)

    # Launch processing in background thread (non-blocking)
    asyncio.get_event_loop().run_in_executor(None, _run_pipeline, job_id, temp_path)

    return {"job_id": job_id, "status": "processing", "message": "Video pipeline started"}


@router.get("/video/status/{job_id}")
async def stream_video_status(job_id: str):
    """
    SSE endpoint that streams progress updates for a video processing job.
    Client connects with EventSource and receives real-time updates.
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    async def event_generator():
        last_sent = ""
        while True:
            job = _jobs.get(job_id)
            if not job:
                yield f"data: {json.dumps({'status': 'not_found'})}\n\n"
                break

            # Build snapshot (exclude annotated_frame numpy arrays)
            snapshot = {k: v for k, v in job.items()
                        if k != "annotated_frame"}

            snapshot_str = json.dumps(snapshot, default=str)

            if snapshot_str != last_sent:
                yield f"data: {snapshot_str}\n\n"
                last_sent = snapshot_str

            if job["status"] in ("completed", "failed"):
                # Send final state and close
                yield f"data: {json.dumps(snapshot, default=str)}\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/video/result/{job_id}")
async def get_video_result(job_id: str):
    """
    Get the final result of a completed video processing job.
    Use this as a fallback if SSE is not available.
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return {k: v for k, v in job.items() if k != "annotated_frame"}
