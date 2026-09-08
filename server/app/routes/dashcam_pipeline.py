import asyncio
import time
import uuid
import threading
from collections import deque
from typing import Dict, Any, Optional

import cv2
import numpy as np

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.database.session import SessionLocal
from app.database.models import BusModel

from app.routes.video_pipeline import (
    _load_waterlogging_model,
    _load_accident_model,
    _load_pothole_model,
    _load_traffic_model,
    _infer_waterlogging,
    _infer_accident,
    _infer_pothole,
    _infer_vehicle,
    _save_incident_from_winner,
)


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/dashcam",
    tags=["dashcam"]
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_FPS = 4.0
SAMPLE_INTERVAL = 1.0 / TARGET_FPS

JPEG_QUALITY = 75

# General fallback confidence threshold.
# Individual detectors should ideally have their own threshold.
DEFAULT_CONFIDENCE_THRESHOLD = 0.50

# Accident requires stronger confirmation because a single
# high-confidence frame is NOT enough to declare an accident.
ACCIDENT_CONFIDENCE_THRESHOLD = 0.60

# Pothole threshold
POTHOLE_CONFIDENCE_THRESHOLD = 0.60

# Waterlogging threshold
WATERLOG_CONFIDENCE_THRESHOLD = 0.60

# Number of recent frames used for temporal confirmation.
TEMPORAL_WINDOW = 5

# Number of positive frames required inside the temporal window.
ACCIDENT_MIN_POSITIVE_FRAMES = 3
POTHOLE_MIN_POSITIVE_FRAMES = 2
WATERLOG_MIN_POSITIVE_FRAMES = 2

# DB incident save cooldown
INCIDENT_SAVE_COOLDOWN = 5.0

# Camera reconnect settings
MAX_CAMERA_FAILURES = 20
RECONNECT_DELAY = 2.0


# ============================================================
# GLOBAL MODEL CACHE
# ============================================================

_models: Dict[str, Any] = {
    "waterlogging": None,
    "accident": None,
    "pothole": None,
    "traffic": None,
}

_models_lock = threading.Lock()
_models_loaded = False


# ============================================================
# ACTIVE STREAMS
# ============================================================

_active_streams: Dict[str, bool] = {}

# Each stream has its own state.
_stream_states: Dict[str, Dict[str, Any]] = {}

_stream_lock = threading.Lock()


# ============================================================
# MODEL LOADING
# ============================================================

def load_all_models():
    """
    Load all AI models exactly once.

    This prevents every browser request from loading four
    separate YOLO models again.
    """

    global _models_loaded

    if _models_loaded:
        return _models

    with _models_lock:

        if _models_loaded:
            return _models

        print("\n" + "=" * 70)
        print("[*] Loading AI models...")
        print("=" * 70)

        try:
            print("[1/4] Loading waterlogging model...")
            _models["waterlogging"] = _load_waterlogging_model()
            print("[+] Waterlogging model loaded.")

            print("[2/4] Loading accident model...")
            _models["accident"] = _load_accident_model()
            print("[+] Accident model loaded.")

            print("[3/4] Loading pothole model...")
            _models["pothole"] = _load_pothole_model()
            print("[+] Pothole model loaded.")

            print("[4/4] Loading traffic model...")
            _models["traffic"] = _load_traffic_model()
            print("[+] Traffic model loaded.")

            _models_loaded = True

            print("=" * 70)
            print("[+] ALL AI MODELS LOADED SUCCESSFULLY")
            print("=" * 70 + "\n")

        except Exception as e:
            print("=" * 70)
            print("[!] MODEL LOADING FAILED")
            print(f"[!] Error: {e}")
            print("=" * 70)

            # Reset partially loaded models
            for key in _models:
                _models[key] = None

            raise

    return _models


# ============================================================
# GPS
# ============================================================

def get_bus_gps(bus_id: str):

    # Fallback coordinates.
    gps = {
        "lat": 17.4422,
        "lng": 78.3923,
        "source": "fallback"
    }

    db = SessionLocal()

    try:

        if bus_id:

            bus = (
                db.query(BusModel)
                .filter(BusModel.id == bus_id)
                .first()
            )

            if bus:

                if bus.lat is not None:
                    gps["lat"] = float(bus.lat)

                if bus.lng is not None:
                    gps["lng"] = float(bus.lng)

                gps["source"] = "database"

    except Exception as e:

        print(f"[!] GPS DB lookup failed: {e}")

    finally:

        db.close()

    return gps


# ============================================================
# SAFE INFERENCE WRAPPER
# ============================================================

def safe_infer(
    inference_function,
    model,
    frame,
    detector_name: str
):

    """
    Calls your existing inference functions safely.

    If one model crashes, the other models can continue running.
    """

    if model is None:

        return {
            "detected": False,
            "confidence": 0.0,
            "severity": "NORMAL",
            "model": detector_name,
            "detections": []
        }

    try:

        result = inference_function(model, frame)

        if result is None:

            return {
                "detected": False,
                "confidence": 0.0,
                "severity": "NORMAL",
                "model": detector_name,
                "detections": []
            }

        # Normalize output.
        if not isinstance(result, dict):

            return {
                "detected": False,
                "confidence": 0.0,
                "severity": "NORMAL",
                "model": detector_name,
                "detections": []
            }

        result.setdefault("detected", False)
        result.setdefault("confidence", 0.0)
        result.setdefault("severity", "NORMAL")
        result.setdefault("model", detector_name)
        result.setdefault("detections", [])

        return result

    except Exception as e:

        print(
            f"[!] {detector_name} inference error: {e}"
        )

        return {
            "detected": False,
            "confidence": 0.0,
            "severity": "NORMAL",
            "model": detector_name,
            "detections": [],
            "error": str(e)
        }


# ============================================================
# TEMPORAL CONFIRMATION
# ============================================================

def update_temporal_state(
    state: Dict[str, Any],
    detector_name: str,
    detected: bool,
    confidence: float
):

    """
    Maintains a rolling detection history.

    Example:

        accident:
        [0, 1, 1, 0, 1]

    3 positives inside 5 frames => confirmed.
    """

    if detector_name not in state:

        state[detector_name] = deque(
            maxlen=TEMPORAL_WINDOW
        )

    history = state[detector_name]

    positive = (
        detected and
        confidence >= DEFAULT_CONFIDENCE_THRESHOLD
    )

    history.append(
        {
            "detected": bool(positive),
            "confidence": float(confidence)
        }
    )

    return history


def is_temporally_confirmed(
    history,
    min_positive_frames: int
):

    if not history:
        return False

    positives = sum(
        1
        for item in history
        if item["detected"]
    )

    return positives >= min_positive_frames


# ============================================================
# ACCIDENT CONFIRMATION
# ============================================================

def confirm_accident(
    state: Dict[str, Any],
    accident_result: Dict[str, Any]
):

    confidence = float(
        accident_result.get("confidence", 0.0)
    )

    detected = bool(
        accident_result.get("detected", False)
    )

    # Stronger threshold specifically for accident.
    positive = (
        detected and
        confidence >= ACCIDENT_CONFIDENCE_THRESHOLD
    )

    history = update_temporal_state(
        state,
        "accident",
        positive,
        confidence
    )

    confirmed = is_temporally_confirmed(
        history,
        ACCIDENT_MIN_POSITIVE_FRAMES
    )

    return confirmed


# ============================================================
# POTHOLE CONFIRMATION
# ============================================================

def confirm_pothole(
    state: Dict[str, Any],
    pothole_result: Dict[str, Any]
):

    confidence = float(
        pothole_result.get("confidence", 0.0)
    )

    detected = bool(
        pothole_result.get("detected", False)
    )

    positive = (
        detected and
        confidence >= POTHOLE_CONFIDENCE_THRESHOLD
    )

    history = update_temporal_state(
        state,
        "pothole",
        positive,
        confidence
    )

    return is_temporally_confirmed(
        history,
        POTHOLE_MIN_POSITIVE_FRAMES
    )


# ============================================================
# WATERLOGGING CONFIRMATION
# ============================================================

def confirm_waterlogging(
    state: Dict[str, Any],
    water_result: Dict[str, Any]
):

    confidence = float(
        water_result.get("confidence", 0.0)
    )

    detected = bool(
        water_result.get("detected", False)
    )

    positive = (
        detected and
        confidence >= WATERLOG_CONFIDENCE_THRESHOLD
    )

    history = update_temporal_state(
        state,
        "waterlogging",
        positive,
        confidence
    )

    return is_temporally_confirmed(
        history,
        WATERLOG_MIN_POSITIVE_FRAMES
    )


# ============================================================
# RESULT NORMALIZATION
# ============================================================

def normalize_result(
    result: Dict[str, Any],
    model_name: str
):

    confidence = float(
        result.get("confidence", 0.0) or 0.0
    )

    confidence = max(
        0.0,
        min(1.0, confidence)
    )

    detected = bool(
        result.get("detected", False)
    )

    severity = str(
        result.get("severity", "NORMAL")
    ).upper()

    return {
        "model": model_name,
        "detected": detected,
        "confidence": confidence,
        "severity": severity,
        "detections": result.get(
            "detections",
            []
        ),
        "raw": result
    }


# ============================================================
# WINNER SELECTION
# ============================================================

def select_incident_winner(
    water_result,
    accident_result,
    pothole_result,
    temporal_state
):

    # --------------------------------------------------------
    # FIRST PRIORITY: CONFIRMED ACCIDENT (Temporarily Disabled)
    # --------------------------------------------------------

    # accident_confirmed = confirm_accident(
    #     temporal_state,
    #     accident_result
    # )

    # if accident_confirmed:
    # 
    #     result = dict(accident_result)
    # 
    #     result["detected"] = True
    #     result["confirmed"] = True
    #     result["model"] = "accident"
    # 
    #     return result

    # --------------------------------------------------------
    # SECOND: CONFIRMED WATERLOGGING
    # --------------------------------------------------------

    water_confirmed = confirm_waterlogging(
        temporal_state,
        water_result
    )

    if water_confirmed:

        result = dict(water_result)

        result["detected"] = True
        result["confirmed"] = True
        result["model"] = "waterlogging"

        return result

    # --------------------------------------------------------
    # THIRD: CONFIRMED POTHOLE
    # --------------------------------------------------------

    pothole_confirmed = confirm_pothole(
        temporal_state,
        pothole_result
    )

    if pothole_confirmed:

        result = dict(pothole_result)

        result["detected"] = True
        result["confirmed"] = True
        result["model"] = "pothole"

        return result

    return None


# ============================================================
# HUD
# ============================================================

def get_status_color(
    alert_level: str
):

    level = alert_level.upper()

    if level in ("CRITICAL", "SEVERE", "HIGH"):
        return (0, 0, 255)

    if level in ("MODERATE", "MEDIUM"):
        return (0, 165, 255)

    if level in ("LOW",):
        return (0, 255, 255)

    return (0, 255, 0)


def draw_box(
    image,
    box,
    label,
    color=(0, 0, 255)
):

    try:

        x1, y1, x2, y2 = map(
            int,
            box
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        label_size, _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            1
        )

        label_y = max(
            y1,
            label_size[1] + 5
        )

        cv2.rectangle(
            image,
            (
                x1,
                label_y - label_size[1] - 5
            ),
            (
                x1 + label_size[0] + 6,
                label_y
            ),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            image,
            label,
            (x1 + 3, label_y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA
        )

    except Exception:
        pass


def draw_detector_boxes(
    image,
    result,
    detector_name,
    color
):

    """
    Draws boxes if your inference function returns them.

    Supports common formats:

        detections = [
            {
                "bbox": [x1,y1,x2,y2],
                "confidence": 0.9
            }
        ]

    OR

        detections = [
            {
                "box": [x1,y1,x2,y2],
                "conf": 0.9
            }
        ]
    """

    detections = result.get(
        "detections",
        result.get("raw", {}).get("detected_vehicles", [])
    )

    if not isinstance(detections, list):
        return

    for detection in detections:

        if not isinstance(detection, dict):
            continue

        box = (
            detection.get("bbox")
            or detection.get("box")
        )

        if box is None:
            continue

        confidence = detection.get(
            "confidence",
            detection.get("conf", 0.0)
        )

        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0

        label = (
            f"{detector_name} "
            f"{confidence * 100:.1f}%"
        )

        draw_box(
            image,
            box,
            label,
            color
        )


# ============================================================
# MAIN HUD
# ============================================================

def draw_hud(
    frame,
    winner,
    water_result,
    accident_result,
    pothole_result,
    traffic_result,
    gps,
    frame_number,
    inference_ms
):

    image = frame.copy()

    h, w = image.shape[:2]

    # --------------------------------------------------------
    # Determine overall state
    # --------------------------------------------------------

    if winner:

        model_name = winner.get(
            "model",
            "incident"
        )

        severity = str(
            winner.get(
                "severity",
                "DETECTED"
            )
        ).upper()

        confidence = float(
            winner.get(
                "confidence",
                0.0
            )
        )

        color = get_status_color(
            severity
        )

        status_text = (
            f"ALERT: {severity} "
            f"{model_name.upper()}"
        )

    else:

        status_text = "STATUS: CLEAR"
        severity = "NORMAL"
        confidence = 0.0
        color = (0, 255, 0)

    # --------------------------------------------------------
    # Dark HUD
    # --------------------------------------------------------

    banner_height = 135

    overlay = image.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (
            w - 10,
            min(
                h - 10,
                banner_height
            )
        ),
        (20, 20, 20),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.82,
        image,
        0.18,
        0,
        image
    )

    # Accent bar

    cv2.rectangle(
        image,
        (15, 15),
        (
            23,
            min(
                h - 10,
                banner_height - 5
            )
        ),
        color,
        -1
    )

    # --------------------------------------------------------
    # Line 1
    # --------------------------------------------------------

    cv2.putText(
        image,
        status_text,
        (35, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Line 2
    # --------------------------------------------------------

    if winner:

        cv2.putText(
            image,
            (
                f"Confidence: "
                f"{confidence * 100:.1f}% | "
                f"CONFIRMED"
            ),
            (35, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    else:

        cv2.putText(
            image,
            "No confirmed road hazard",
            (35, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # Detector summary
    # --------------------------------------------------------

    def conf(result):

        try:
            return float(
                result.get(
                    "confidence",
                    0.0
                )
            ) * 100.0
        except Exception:
            return 0.0

    detector_line = (
        f"Pothole: {conf(pothole_result):.0f}% | "
        f"Water: {conf(water_result):.0f}%"
    )

    cv2.putText(
        image,
        detector_line,
        (35, 89),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (210, 210, 210),
        1,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Traffic
    # --------------------------------------------------------

    raw_traffic = traffic_result.get("raw", {})
    vehicle_count = raw_traffic.get(
        "vehicles_detected",
        "?"
    )
    density = raw_traffic.get("density_pct", 0)

    traffic_line = (
        f"Traffic Vehicles: "
        f"{vehicle_count} | "
        f"Density: {density}%"
    )

    cv2.putText(
        image,
        traffic_line,
        (35, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (210, 210, 210),
        1,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # GPS + latency
    # --------------------------------------------------------

    gps_line = (
        f"GPS: "
        f"({gps['lat']:.5f}, {gps['lng']:.5f}) | "
        f"{inference_ms:.0f}ms | "
        f"Frame: {frame_number}"
    )

    # Put GPS on bottom if enough room.
    gps_y = min(
        h - 15,
        banner_height + 20
    )

    cv2.putText(
        image,
        gps_line,
        (15, gps_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (180, 180, 180),
        1,
        cv2.LINE_AA
    )

    return image


# ============================================================
# INCIDENT SAVE
# ============================================================

def save_incident(
    job_id,
    bus_id,
    gps,
    winner,
    state
):

    if not winner:
        return

    now = time.time()

    last_save = state.get(
        "last_incident_save",
        0.0
    )

    # Do not continuously spam DB.
    if now - last_save < INCIDENT_SAVE_COOLDOWN:
        return

    try:

        winner_copy = dict(winner)

        incident_id = _save_incident_from_winner(
            job_id,
            bus_id,
            gps,
            winner_copy
        )

        winner["incident_id"] = incident_id

        state["last_incident_save"] = now

        print(
            f"[+] Incident saved: "
            f"{winner.get('model')} | "
            f"{winner.get('confidence', 0) * 100:.1f}%"
        )

    except Exception as e:

        print(
            f"[!] Could not save incident: {e}"
        )


# ============================================================
# BLANK FRAME
# ============================================================

def create_status_frame(
    text,
    width=640,
    height=480,
    color=(0, 255, 255)
):

    frame = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    cv2.putText(
        frame,
        text,
        (35, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA
    )

    return frame


def encode_frame(frame):

    ret, buffer = cv2.imencode(
        ".jpg",
        frame,
        [
            int(cv2.IMWRITE_JPEG_QUALITY),
            JPEG_QUALITY
        ]
    )

    if not ret:
        return None

    return buffer.tobytes()


def mjpeg_chunk(frame_bytes):

    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n"
        b"Content-Length: "
        + str(len(frame_bytes)).encode()
        + b"\r\n\r\n"
        + frame_bytes
        + b"\r\n"
    )


# ============================================================
# LIVE GENERATOR
# ============================================================

def generate_mjpeg_stream(
    stream_url: str,
    bus_id: str,
    job_id: str
):

    """
    Main live camera pipeline.

    Camera
       |
       v
    OpenCV
       |
       +----> Waterlogging
       |
       +----> Accident
       |
       +----> Pothole
       |
       +----> Traffic
       |
       v
    Temporal confirmation
       |
       v
    Winner
       |
       v
    HUD + MJPEG
       |
       v
    Browser
    """

    cap = None

    # --------------------------------------------------------
    # Per-stream state
    # --------------------------------------------------------

    state = {

        "accident": deque(
            maxlen=TEMPORAL_WINDOW
        ),

        "pothole": deque(
            maxlen=TEMPORAL_WINDOW
        ),

        "waterlogging": deque(
            maxlen=TEMPORAL_WINDOW
        ),

        "last_incident_save": 0.0,

        "frame_number": 0,

        "last_capture_time": 0.0,

    }

    with _stream_lock:

        _stream_states[job_id] = state

    # --------------------------------------------------------
    # GPS
    # --------------------------------------------------------

    gps = get_bus_gps(
        bus_id
    )

    # --------------------------------------------------------
    # Loading frame
    # --------------------------------------------------------

    loading = create_status_frame(
        "CONNECTING TO CAMERA...",
        640,
        480
    )

    encoded = encode_frame(
        loading
    )

    if encoded:

        yield mjpeg_chunk(
            encoded
        )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    try:

        models = load_all_models()

    except Exception as e:

        error_frame = create_status_frame(
            f"MODEL ERROR: {str(e)[:60]}",
            640,
            480,
            (0, 0, 255)
        )

        encoded = encode_frame(
            error_frame
        )

        if encoded:
            yield mjpeg_chunk(
                encoded
            )

        _active_streams.pop(
            job_id,
            None
        )

        return

    # --------------------------------------------------------
    # Camera loop
    # --------------------------------------------------------

    try:

        print(
            f"\n[*] Starting live dashcam:"
            f"\n    URL: {stream_url}"
            f"\n    BUS: {bus_id}"
            f"\n    JOB: {job_id}\n"
        )

        while _active_streams.get(
            job_id,
            False
        ):

            # ------------------------------------------------
            # Open/reopen camera
            # ------------------------------------------------

            cap = cv2.VideoCapture(
                stream_url
            )

            cap.set(
                cv2.CAP_PROP_BUFFERSIZE,
                1
            )

            if not cap.isOpened():

                print(
                    f"[!] Camera unavailable: "
                    f"{stream_url}"
                )

                waiting = create_status_frame(
                    "WAITING FOR CAMERA...",
                    640,
                    480,
                    (0, 165, 255)
                )

                encoded = encode_frame(
                    waiting
                )

                if encoded:

                    yield mjpeg_chunk(
                        encoded
                    )

                cap.release()

                time.sleep(
                    RECONNECT_DELAY
                )

                continue

            print(
                "[+] PHONE CAMERA CONNECTED"
            )

            consecutive_failures = 0

            # ------------------------------------------------
            # Frame loop
            # ------------------------------------------------

            while _active_streams.get(
                job_id,
                False
            ):

                ret, frame = cap.read()

                # --------------------------------------------
                # Camera failure
                # --------------------------------------------

                if (
                    not ret
                    or frame is None
                    or frame.size == 0
                ):

                    consecutive_failures += 1

                    if consecutive_failures >= MAX_CAMERA_FAILURES:

                        print(
                            "[!] Camera disconnected. "
                            "Reconnecting..."
                        )

                        break

                    time.sleep(
                        0.05
                    )

                    continue

                consecutive_failures = 0

                # --------------------------------------------
                # Frame sampling
                # --------------------------------------------

                now = time.time()

                if (
                    now -
                    state["last_capture_time"]
                    < SAMPLE_INTERVAL
                ):

                    # Still send live camera preview
                    preview = frame.copy()

                    cv2.putText(
                        preview,
                        "LIVE CAMERA",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )

                    encoded = encode_frame(
                        preview
                    )

                    if encoded:

                        yield mjpeg_chunk(
                            encoded
                        )

                    continue

                state["last_capture_time"] = now

                state["frame_number"] += 1

                frame_number = state[
                    "frame_number"
                ]

                # --------------------------------------------
                # Inference
                # --------------------------------------------

                inference_start = (
                    time.perf_counter()
                )

                # ============================================
                # 1. WATERLOGGING
                # ============================================

                water_result = safe_infer(
                    _infer_waterlogging,
                    models["waterlogging"],
                    frame,
                    "waterlogging"
                )

                water_result = normalize_result(
                    water_result,
                    "waterlogging"
                )

                # ============================================
                # 2. ACCIDENT (Temporarily Disabled)
                # ============================================

                # accident_result = safe_infer(
                #     _infer_accident,
                #     models["accident"],
                #     frame,
                #     "accident"
                # )
                
                # Mocking empty result to bypass wrong alerts
                accident_result = safe_infer(
                    _infer_accident,
                    None,
                    frame,
                    "accident"
                )

                accident_result = normalize_result(
                    accident_result,
                    "accident"
                )

                # ============================================
                # 3. POTHOLE
                # ============================================

                pothole_result = safe_infer(
                    _infer_pothole,
                    models["pothole"],
                    frame,
                    "pothole"
                )

                pothole_result = normalize_result(
                    pothole_result,
                    "pothole"
                )

                # ============================================
                # 4. TRAFFIC
                # ============================================

                traffic_result = safe_infer(
                    _infer_vehicle,
                    models["traffic"],
                    frame,
                    "traffic"
                )

                traffic_result = normalize_result(
                    traffic_result,
                    "traffic"
                )
                
                # Save to state for frontend polling
                state["latest_traffic"] = traffic_result

                inference_ms = (
                    time.perf_counter()
                    - inference_start
                ) * 1000.0

                # --------------------------------------------
                # Winner selection
                # --------------------------------------------

                winner = select_incident_winner(
                    water_result,
                    accident_result,
                    pothole_result,
                    state
                )

                # --------------------------------------------
                # Alert level
                # --------------------------------------------

                if winner:

                    alert_level = str(
                        winner.get(
                            "severity",
                            "MEDIUM"
                        )
                    ).upper()

                else:

                    alert_level = "NORMAL"

                # --------------------------------------------
                # Draw detection boxes
                # --------------------------------------------

                annotated = frame.copy()

                # Pothole
                draw_detector_boxes(
                    annotated,
                    pothole_result,
                    "POTHOLE",
                    (0, 165, 255)
                )

                # Waterlogging
                draw_detector_boxes(
                    annotated,
                    water_result,
                    "WATER",
                    (255, 191, 0)
                )

                # Traffic
                draw_detector_boxes(
                    annotated,
                    traffic_result,
                    "VEHICLE",
                    (255, 255, 255)
                )

                # --------------------------------------------
                # HUD
                # --------------------------------------------

                annotated = draw_hud(
                    annotated,
                    winner,
                    water_result,
                    accident_result,
                    pothole_result,
                    traffic_result,
                    gps,
                    frame_number,
                    inference_ms
                )

                # --------------------------------------------
                # Encode
                # --------------------------------------------

                encoded = encode_frame(
                    annotated
                )

                if encoded:

                    yield mjpeg_chunk(
                        encoded
                    )

                # --------------------------------------------
                # DB incident save
                # --------------------------------------------

                if winner:

                    save_incident(
                        job_id,
                        bus_id,
                        gps,
                        winner,
                        state
                    )

                # --------------------------------------------
                # Logging
                # --------------------------------------------

                print(
                    f"[FRAME {frame_number:04d}] "
                    f"Accident="
                    f"{accident_result['confidence']:.2f} | "
                    f"Pothole="
                    f"{pothole_result['confidence']:.2f} | "
                    f"Water="
                    f"{water_result['confidence']:.2f} | "
                    f"Traffic="
                    f"{traffic_result['confidence']:.2f} | "
                    f"STATUS={alert_level} | "
                    f"{inference_ms:.0f}ms"
                )

            # ------------------------------------------------
            # Release broken camera
            # ------------------------------------------------

            if cap is not None:

                cap.release()
                cap = None

            if _active_streams.get(
                job_id,
                False
            ):

                print(
                    "[*] Reconnecting to camera..."
                )

                time.sleep(
                    RECONNECT_DELAY
                )

    except GeneratorExit:

        print(
            f"[*] Browser disconnected: "
            f"{job_id}"
        )

    except Exception as e:

        print(
            f"[!] Live stream error: {e}"
        )

    finally:

        if cap is not None:

            cap.release()

        _active_streams.pop(
            job_id,
            None
        )

        with _stream_lock:

            _stream_states.pop(
                job_id,
                None
            )

        print(
            f"[*] Dashcam stream stopped: "
            f"{job_id}"
        )


# ============================================================
# START STREAM
# ============================================================

@router.get("/stream")
async def dashcam_stream(
    url: str = Query(
        ...,
        description=(
            "IP Webcam video URL, "
            "example: http://10.1.83.49:8080/video"
        )
    ),

    bus_id: str = Query(
        "BUS-HYD-VID",
        description="Bus Context ID"
    )
):

    # --------------------------------------------------------
    # Validate URL
    # --------------------------------------------------------

    if not url.startswith(
        ("http://", "https://")
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid camera URL. "
                "Use http://IP:PORT/video"
            )
        )

    # --------------------------------------------------------
    # Create job
    # --------------------------------------------------------

    job_id = (
        f"dash_"
        f"{uuid.uuid4().hex[:10]}"
    )

    _active_streams[
        job_id
    ] = True

    print(
        f"[+] Created dashcam job "
        f"{job_id}"
    )

    # --------------------------------------------------------
    # Streaming response
    # --------------------------------------------------------

    return StreamingResponse(

        generate_mjpeg_stream(
            stream_url=url,
            bus_id=bus_id,
            job_id=job_id
        ),

        media_type=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        ),

        headers={
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Connection": "keep-alive",
        }
    )


# ============================================================
# STOP STREAM
# ============================================================

@router.delete(
    "/stream/{job_id}"
)
async def stop_dashcam_stream(
    job_id: str
):

    if job_id not in _active_streams:

        raise HTTPException(
            status_code=404,
            detail="Stream not found"
        )

    _active_streams[
        job_id
    ] = False

    print(
        f"[*] Stopping dashcam "
        f"job: {job_id}"
    )

    return {
        "status": "stopped",
        "job_id": job_id
    }


# ============================================================
# LIVE DASHCAM STATUS
# ============================================================

@router.get("/status/{job_id}")
async def get_dashcam_status(job_id: str):
    """Return the latest traffic and incident data for the frontend."""
    if job_id not in _active_streams:
        raise HTTPException(status_code=404, detail="Stream not found")
        
    state = _stream_states.get(job_id, {})
    latest_traffic = state.get("latest_traffic")
    
    return {
        "job_id": job_id,
        "active": _active_streams[job_id],
        "vehicle_summary": latest_traffic.get("raw") if latest_traffic else None
    }


# ============================================================
# ACTIVE STREAMS
# ============================================================

@router.get("/streams")
async def get_active_streams():

    return {
        "active_streams": list(
            _active_streams.keys()
        ),
        "count": len(
            _active_streams
        )
    }


# ============================================================
# MODEL STATUS
# ============================================================

@router.get("/models/status")
async def model_status():

    return {
        "loaded": _models_loaded,

        "models": {
            name: model is not None
            for name, model in _models.items()
        }
    }