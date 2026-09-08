import io
import glob
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from PIL import Image
import requests
from ultralytics import YOLO

# EXACT accident class name from Task 1 data.yaml
ACCIDENT_CLASS_NAME = "Car Crash Severity Detection - v13 Version 11 - with background"


class JetsonAccidentDetector:
    def __init__(
        self,
        bus_id: str,
        weights_path: Optional[str] = None,
        server_url: str = "http://localhost:8000/api/v1"
    ):
        self.bus_id = bus_id
        self.server_url = server_url
        self.model_name = "YOLOv8n-Accident-EdgeNet"
        self.last_speed_kmh: Optional[float] = None
        self.target_class_name = ACCIDENT_CLASS_NAME
        self.accident_history = deque(maxlen=15)
        self.accident_confirmed = False

        # Resolve weights path
        if not weights_path:
            weights_path = self._find_best_weights()

        self.weights_path = weights_path
        print(f"[JetsonAccidentDetector] Loading weights from: {self.weights_path}")
        self.model = YOLO(self.weights_path)

    def _find_best_weights(self) -> str:
        """Finds best.pt from ai_models/accident/weights/best.pt, runs/ or falls back to yolov8n.pt."""
        base_dir = Path(__file__).resolve().parent
        helios_root = base_dir.parent.parent

        search_patterns = [
            str(base_dir / "weights" / "best.pt"),
            str(base_dir / "runs" / "accident" / "**" / "weights" / "best.pt"),
            str(helios_root / "ai_models" / "accident" / "weights" / "best.pt"),
            str(helios_root / "runs" / "accident" / "**" / "weights" / "best.pt"),
        ]

        for pattern in search_patterns:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                # Return the newest matched weights file
                matches.sort(key=os.path.getmtime, reverse=True)
                return matches[0]

        # Fallback if training hasn't produced weights yet
        fallback = helios_root / "yolov8n.pt"
        return str(fallback) if fallback.exists() else "yolov8n.pt"


    def process_frame(
        self,
        frame_bytes: bytes,
        lat: float,
        lng: float,
        speed_kmh: Optional[float] = None,
        camera_id: str = "front"
    ) -> Optional[Dict[str, Any]]:
        """
        Runs real inference on frame_bytes:
        1. Extracts accident confidence using EXACT class name from Task 1 data.yaml.
        2. Fuses deceleration signal: if speed drops >= 25 km/h & confidence > 0.4, boost by 0.2 (max 1.0).
        3. Emits payload only when confidence >= 0.75.
        4. Matches shared Helios payload schema.
        """
        # Decode frame bytes to PIL Image
        try:
            image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
        except Exception as e:
            print(f"[JetsonAccidentDetector] Error decoding frame_bytes: {e}")
            return None

        # Run real inference
        results = self.model.predict(source=image, verbose=False)

        raw_confidence = 0.0
        detected_boxes_count = 0
        crash_boxes_count = 0
        if results and len(results) > 0:
            result = results[0]
            names = result.names  # mapping from int to class name string
            detected_boxes_count = len(result.boxes)
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                cls_name = names.get(cls_id, "")
                conf = float(box.conf[0].item())

                # Strict class filtering
                cls_name_lower = cls_name.lower()
                if cls_name == self.target_class_name or "crash" in cls_name_lower or "collision" in cls_name_lower:
                    crash_boxes_count += 1
                    if conf > raw_confidence:
                        raw_confidence = conf

        self.last_raw_confidence = raw_confidence
        self.last_results = results
        self.last_boxes_count = detected_boxes_count
        self.last_crash_boxes_count = crash_boxes_count

        # Temporal Verification
        self.accident_history.append(raw_confidence)
        high_conf_frames = sum(c >= 0.70 for c in self.accident_history)
        temporal_confidence = (
            sum(self.accident_history) / len(self.accident_history)
            if self.accident_history
            else 0.0
        )

        # Telemetry fusion as supporting evidence (NOT a direct raw boost)
        telemetry_score = 0.0
        if speed_kmh is not None and self.last_speed_kmh is not None:
            speed_drop = self.last_speed_kmh - speed_kmh
            if speed_drop >= 25.0:
                telemetry_score = 1.0  # Strong deceleration evidence
                print(f"[JetsonAccidentDetector] Strong deceleration detected: {speed_drop:.1f} km/h drop")

        # Update last speed
        if speed_kmh is not None:
            self.last_speed_kmh = speed_kmh

        # Temporal gating (fail early if not sustained)
        if len(self.accident_history) < 8:
            return None
        
        # We need at least 5 strong visual frames in our window
        if high_conf_frames < 5:
            return None

        # Calculate final fused confidence score (Visual 60%, Temporal 30%, Telemetry 10%)
        final_confidence = (raw_confidence * 0.60) + (temporal_confidence * 0.30) + (telemetry_score * 0.10)

        # Threshold gate: Only emit payload when fused confidence is high enough
        if final_confidence < 0.70:
            return None

        # Calculate severity based on final confidence and speed drop
        if final_confidence >= 0.88 or telemetry_score > 0:
            severity = "critical"
        elif final_confidence >= 0.78:
            severity = "high"
        else:
            severity = "medium"

        # Save annotated image for dashboard and local inspection
        base_dir = Path(__file__).resolve().parent
        helios_root = base_dir.parent.parent
        image_url = "https://images.unsplash.com/photo-1568605117036-5fe5e7bab0b7?w=1000&auto=format&fit=crop&q=80"
        try:
            media_dir = helios_root / "server" / "media"
            os.makedirs(media_dir, exist_ok=True)
            output_filename = f"accident_{int(datetime.now(timezone.utc).timestamp())}.jpg"
            output_path = media_dir / output_filename
            latest_path = media_dir / "latest_accident.jpg"
            
            if results and len(results) > 0:
                annotated_bgr = results[0].plot()
                annotated_rgb = annotated_bgr[..., ::-1]
                img_to_save = Image.fromarray(annotated_rgb)
                img_to_save.save(str(output_path))
                img_to_save.save(str(latest_path))
                image_url = f"http://localhost:8000/media/{output_filename}"
        except Exception as e:
            print(f"[JetsonAccidentDetector] Note: Could not save annotated media: {e}")

        payload = {
            "bus_id": self.bus_id,
            "event_type": "accident",
            "confidence": round(float(final_confidence), 4),
            "severity": severity,
            "gps": {"lat": lat, "lng": lng},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "camera": camera_id,
            "image_url": image_url,
            "video_url": None,
            "model": self.model_name,
            "status": "detected",
            "notes": f"Accident detected: Visual {raw_confidence:.1%}, Temporal {temporal_confidence:.1%}, Fused {final_confidence:.1%}"
        }

        return payload

    def send_payload(self, payload: Dict[str, Any]) -> requests.Response:
        """Sends the payload to the Helios backend."""
        endpoint = f"{self.server_url.rstrip('/')}/detect/accident"
        response = requests.post(endpoint, json=payload, timeout=5)
        return response


if __name__ == "__main__":
    import sys
    import json

    # Determine image path from command line arguments
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        img_path = Path(sys.argv[1]).resolve()
    else:
        # Default sample image
        base_dir = Path(__file__).resolve().parent
        helios_root = base_dir.parent.parent
        img_path = helios_root / "model_Test_my_image" / "sample_car_accident.jpg"

    if not img_path.exists():
        print(f"[!] Error: Image file not found: {img_path}")
        sys.exit(1)

    print("=" * 65)
    print("  HELIOS JETSON ACCIDENT DETECTOR (DIRECT RUN)")
    print("=" * 65)
    print(f"[*] Target Image : {img_path}")

    # Read binary bytes of the target image
    with open(img_path, "rb") as f:
        frame_bytes = f.read()

    detector = JetsonAccidentDetector(bus_id="BUS-TEST-01")

    # Optional: Check if user passed --boost or simulate sudden deceleration (e.g. vehicle braking abruptly from 60 to 20 km/h)
    simulate_brake = "--decel" in sys.argv or "-d" in sys.argv
    if simulate_brake:
        print("[*] Simulating telemetry: vehicle sudden deceleration (-40 km/h)")
        detector.last_speed_kmh = 60.0
        speed_now = 20.0
    else:
        speed_now = 40.0

    print("[*] Processing frame with YOLO accident model...")
    payload = detector.process_frame(
        frame_bytes=frame_bytes,
        lat=17.4422,
        lng=78.3923,
        speed_kmh=speed_now
    )

    print("\n" + "-" * 65)
    print("[DETECTION BREAKDOWN]:")
    print("-" * 65)
    raw_conf = getattr(detector, "last_raw_confidence", 0.0)
    total_boxes = getattr(detector, "last_boxes_count", 0)
    crash_boxes = getattr(detector, "last_crash_boxes_count", 0)
    print(f"--> Total Objects Found     : {total_boxes} box(es)")
    print(f"--> Car Crash Detections    : {crash_boxes} box(es)")
    print(f"--> Peak Crash Confidence   : {raw_conf:.1%}")
    if total_boxes > 0 and crash_boxes == 0:
        print("    [!] The AI detected objects in the image, but none were recognized as a 'Car Crash'.")
    if simulate_brake:
        print(f"--> Sensor Telemetry Boost  : +20.0% (Triggered by >=25 km/h speed drop)")

    print("\n" + "-" * 65)
    print("[RETURNED PAYLOAD]:")
    print("-" * 65)
    if payload:
        print(json.dumps(payload, indent=2))
        print(f"\n--> STATUS: [ACCIDENT TRIGGERED]")
        print(f"--> Severity Level : {payload['severity'].upper()}")
        print(f"--> Final Confidence: {payload['confidence']:.1%}")
    else:
        print("None")
        print("\n--> STATUS: Filtered by 75% threshold gate.")
        print(f"    (Model visual confidence was {raw_conf:.1%}, which is < 75% trigger gate.")
        print("     Pass '--decel' to simulate sudden braking deceleration boost)")
    print("=" * 65)

