import argparse
import json
import math
import os
import signal
import sys
import time
import urllib.request
from pathlib import Path
# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from ultralytics import YOLO

# ======================= CONFIGURATION =======================
PHONE_IP = "10.2.5.75"
PHONE_PORT = "8080"
STREAM_URL = f"http://{PHONE_IP}:{PHONE_PORT}/video"
SENSORS_URL = f"http://{PHONE_IP}:{PHONE_PORT}/sensors.json"

TARGET_FPS = 4.0
SAMPLE_INTERVAL_SEC = 1.0 / TARGET_FPS

CLUSTER_DISTANCE_THRESHOLD_M = 10.0
BASE_CONF_DETECTION = 0.35
MIN_WATERLOG_AREA_PX = 5000
IMG_SIZE = 640

MODEL_PATH = os.path.join("weights", "waterlog_best.pt")
# =============================================================


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0)**2
    return R * (2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a)))


def draw_waterlog_hud(annotated, w, h, severity, coverage, w_score, avg_score, lat, lon, mean_conf, hud_color, latency_ms=None):
    banner_width = min(w - 20, int(w * 0.95))
    banner_height = 110

    overlay = annotated.copy()
    cv2.rectangle(overlay, (10, 10), (10 + banner_width, 10 + banner_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.80, annotated, 0.20, 0, annotated)

    cv2.rectangle(annotated, (15, 16), (23, 10 + banner_height - 6), hud_color, -1)

    font_scale = 0.58 if w >= 720 else 0.45
    sub_font_scale = 0.48 if w >= 720 else 0.38
    text_x = 32

    # Line 1: Status
    cv2.putText(annotated, f"Status: {severity} WATERLOGGING", (text_x, 34),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, hud_color, 2, cv2.LINE_AA)

    # Line 2: Water Coverage
    cv2.putText(annotated, f"Road Water Coverage: {coverage}%", (text_x, 58),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

    # Line 3: Scores
    cv2.putText(annotated, f"Hazard Score: {w_score} | Selection Avg: {avg_score}", (text_x, 82),
                cv2.FONT_HERSHEY_SIMPLEX, sub_font_scale, (220, 220, 220), 1, cv2.LINE_AA)

    # Line 4: GPS & Latency
    latency_str = f" | Latency: {latency_ms:.1f}ms" if latency_ms is not None else ""
    cv2.putText(annotated, f"GPS: ({lat:.4f}, {lon:.4f}){latency_str}", (text_x, 104),
                cv2.FONT_HERSHEY_SIMPLEX, sub_font_scale, (180, 180, 180), 1, cv2.LINE_AA)

    return annotated


class LiveWaterlogManager:
    def __init__(self, output_root="waterlog_live_outputs", target_fps=4.0):
        self.output_root = output_root
        self.target_fps = target_fps
        self.dir_all_frames = os.path.join(output_root, "all_annotated_frames")
        self.dir_patch_best = os.path.join(output_root, "clean_patch_outputs")
        self.dir_global_best = os.path.join(output_root, "global_best_incident")
        self.dir_telemetry = os.path.join(output_root, "json_telemetry")

        for d in [self.dir_all_frames, self.dir_patch_best, self.dir_global_best, self.dir_telemetry]:
            os.makedirs(d, exist_ok=True)

        self.incidents = []
        self.global_best = None
        self.video_writer = None
        self.video_path = os.path.join(self.output_root, "output_waterlog.mp4")
        self.last_known_gps = {"lat": 16.5062, "lon": 80.6480, "speed_kmh": 25.0, "source": "FALLBACK_GPS"}
        self.last_gps_fetch_time = 0.0

    def init_video_writer(self, width, height):
        if self.video_writer is not None:
            return

        # Use H.264 codecs natively recognized by Windows Media Player
        encoders = [
            ('H264', cv2.VideoWriter_fourcc(*'H264')),
            ('avc1', cv2.VideoWriter_fourcc(*'avc1')),
            ('mp4v', cv2.VideoWriter_fourcc(*'mp4v'))
        ]

        for tag, fourcc in encoders:
            writer = cv2.VideoWriter(self.video_path, fourcc, self.target_fps, (width, height))
            if writer.isOpened():
                self.video_writer = writer
                print(f"[+] Recording video with codec '{tag}' -> '{self.video_path}'")
                return

        # Fallback if MP4 container creation fails
        fallback_avi = os.path.join(self.output_root, "output_waterlog.avi")
        self.video_writer = cv2.VideoWriter(fallback_avi, cv2.VideoWriter_fourcc(*'MJPG'), self.target_fps, (width, height))
        print(f"[!] Fallback to MJPG AVI -> '{fallback_avi}'")

    def fetch_phone_gps(self):
        now = time.time()
        if now - self.last_gps_fetch_time < 1.0:
            return self.last_known_gps

        try:
            req = urllib.request.Request(SENSORS_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=0.3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                gps_info = data.get("gps", {}).get("data", [])
                if gps_info:
                    latest = gps_info[-1][1]
                    self.last_known_gps = {
                        "lat": round(float(latest[0]), 6),
                        "lon": round(float(latest[1]), 6),
                        "speed_kmh": round(float(latest[3]) * 3.6, 1) if len(latest) > 3 and latest[3] else 25.0,
                        "source": "PHONE_HARDWARE_GPS"
                    }
                    self.last_gps_fetch_time = now
        except Exception:
            pass
        return self.last_known_gps

    def register_incident(self, frame_img, file_id, avg_score, mean_conf, w_score, road_coverage_pct, lat, lon, alert_lvl, latency_ms):
        if alert_lvl == "NORMAL" or w_score <= 0:
            return

        # 1. Update Global Best Incident
        if (self.global_best is None) or (avg_score > self.global_best["avg_score"]):
            for f in os.listdir(self.dir_global_best):
                try:
                    os.remove(os.path.join(self.dir_global_best, f))
                except OSError:
                    pass

            global_img_name = f"GLOBAL_BEST_{file_id}.jpg"
            cv2.imwrite(os.path.join(self.dir_global_best, global_img_name), frame_img)

            self.global_best = {
                "frame": global_img_name,
                "avg_score": avg_score,
                "conf": mean_conf,
                "score": w_score,
                "coverage": road_coverage_pct,
                "lat": lat,
                "lon": lon,
                "severity": alert_lvl,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(os.path.join(self.dir_global_best, "global_best_meta.json"), "w") as gf:
                json.dump(self.global_best, gf, indent=2)
            print(f"[★ GLOBAL WINNER UPDATED] Score: {avg_score:.2f} @ ({lat}, {lon})")

        # 2. 10m Spatial Deduplication Patch
        matched = None
        for inc in self.incidents:
            dist = haversine_m(inc["best_lat"], inc["best_lon"], lat, lon)
            if dist <= CLUSTER_DISTANCE_THRESHOLD_M:
                matched = inc
                break

        patch_img_name = f"{file_id}.jpg"

        if matched is not None:
            matched["frames_merged"] += 1
            if avg_score > matched["best_avg_score"]:
                old_img = os.path.join(self.dir_patch_best, matched["best_img_name"])
                if os.path.exists(old_img):
                    try:
                        os.remove(old_img)
                    except OSError:
                        pass

                cv2.imwrite(os.path.join(self.dir_patch_best, patch_img_name), frame_img)
                matched["best_avg_score"] = avg_score
                matched["best_conf"] = mean_conf
                matched["best_score"] = w_score
                matched["best_coverage"] = road_coverage_pct
                matched["best_lat"] = lat
                matched["best_lon"] = lon
                matched["best_img_name"] = patch_img_name
                matched["severity"] = alert_lvl
        else:
            new_id = len(self.incidents) + 1
            cv2.imwrite(os.path.join(self.dir_patch_best, patch_img_name), frame_img)
            self.incidents.append({
                "id": new_id,
                "best_avg_score": avg_score,
                "best_conf": mean_conf,
                "best_score": w_score,
                "best_coverage": road_coverage_pct,
                "best_lat": lat,
                "best_lon": lon,
                "best_img_name": patch_img_name,
                "severity": alert_lvl,
                "frames_merged": 1
            })

        # 3. Save telemetry JSON
        iso_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        telemetry = {
            "id": f"INC-LIVE-{file_id}",
            "incident_id": f"WATERLOG_LIVE_{file_id}",
            "bus_id": "BUS-205",
            "event_type": "waterlogging",
            "status": "detected",
            "camera": "phone_live",
            "model": "yolo11n-seg",
            "timestamp": iso_timestamp,
            "frame": f"{file_id}.jpg",
            "confidence": mean_conf,
            "severity": alert_lvl.lower(),
            "classification": {
                "severity": alert_lvl,
                "severity_title": f"{alert_lvl} WATERLOGGING"
            },
            "gps": {
                "lat": lat,
                "lon": lon,
                "speed_kmh": self.last_known_gps["speed_kmh"]
            },
            "metrics": {
                "confidence": mean_conf,
                "water_hazard_score": w_score,
                "road_water_coverage_pct": road_coverage_pct,
                "selection_avg_score": avg_score
            },
            "deduplication_meta": {
                "source": "LIVE_IP_WEBCAM_4FPS",
                "cluster_distance_m": CLUSTER_DISTANCE_THRESHOLD_M
            },
            "metadata_json": json.dumps({"selection_avg_score": avg_score, "latency_ms": round(latency_ms, 2)})
        }

        with open(os.path.join(self.dir_telemetry, f"{file_id}.json"), "w") as jf:
            json.dump(telemetry, jf, indent=2)

    def close(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            print(f"[+] Output MP4 video finalized and closed successfully: '{self.video_path}'")


def run_live_pipeline(stream_url=STREAM_URL, output_root="waterlog_live_outputs", target_fps=TARGET_FPS):
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Model weights not found at: {MODEL_PATH}")
        sys.exit(1)

    print("[*] Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    manager = LiveWaterlogManager(output_root=output_root, target_fps=target_fps)

    # Clean signal handling for Ctrl+C or terminal termination
    def handle_shutdown(signum, frame):
        print("\n[*] Shutdown triggered. Finalizing video...")
        manager.close()
        cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    auto_mode = True
    last_capture_time = 0.0
    frame_idx = 0
    sample_interval = 1.0 / target_fps

    print(f"[*] Connecting to Phone Camera: {stream_url} ...")

    try:
        while True:
            cap = cv2.VideoCapture(stream_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                print("[!] Camera stream unavailable. Retrying in 2 seconds...")
                time.sleep(2)
                continue

            print("[+] Phone stream connected! Active.")
            consecutive_invalid = 0

            while True:
                ret, frame = cap.read()

                # Robust Disconnect & Blank Frame Filter
                # Drop frames if read fails, if frame is empty, or if pixel mean is under 8 (completely black stream)
                if not ret or frame is None or frame.size == 0 or frame.mean() < 8.0:
                    consecutive_invalid += 1
                    if consecutive_invalid >= 4:
                        print("[!] Disconnect / Blank camera feed detected. Stopping detection...")
                        cap.release()
                        time.sleep(1.0)
                        break
                    time.sleep(0.05)
                    continue

                consecutive_invalid = 0
                h, w, _ = frame.shape
                manager.init_video_writer(w, h)

                now = time.time()
                preview = frame.copy()

                status_text = "AUTO 4-FPS RUNNING | [S] Pause | [Q] Quit" if auto_mode else "PAUSED | [S] Resume | [SPACE] Step | [Q] Quit"
                color = (0, 255, 0) if auto_mode else (0, 165, 255)
                cv2.rectangle(preview, (10, 10), (min(w - 10, 740), 50), (20, 20, 20), -1)
                cv2.putText(preview, status_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                cv2.imshow("Phone Dashcam Viewfinder", preview)

                if auto_mode and (now - last_capture_time >= sample_interval):
                    last_capture_time = now
                    frame_idx += 1

                    roi_top, roi_bottom = int(h * 0.25), int(h * 0.95)
                    road_pixels = (roi_bottom - roi_top) * w

                    t0 = time.perf_counter()
                    results = model.predict(source=frame, conf=BASE_CONF_DETECTION, imgsz=IMG_SIZE, verbose=False)[0]
                    latency_ms = (time.perf_counter() - t0) * 1000

                    full_mask = np.zeros((h, w), dtype=np.uint8)
                    confidence_scores = []

                    if results.masks is not None and results.boxes is not None:
                        boxes_conf = results.boxes.conf.cpu().numpy()
                        for idx, mask_t in enumerate(results.masks.data):
                            m = cv2.resize(mask_t.cpu().numpy().astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                            if np.count_nonzero(m[roi_top:roi_bottom, :]) >= MIN_WATERLOG_AREA_PX:
                                full_mask = np.bitwise_or(full_mask, m)
                                confidence_scores.append(float(boxes_conf[idx]))

                    road_mask = full_mask[roi_top:roi_bottom, :]
                    water_px = np.count_nonzero(road_mask)
                    road_coverage_pct = round((water_px / road_pixels) * 100.0 if road_pixels > 0 else 0.0, 2)
                    mean_conf = round(float(np.mean(confidence_scores)), 3) if confidence_scores else 0.0

                    w_score = round(road_coverage_pct * mean_conf, 2)
                    avg_score = round(((mean_conf * 100.0) + w_score) / 2.0, 2)

                    if w_score < 5.0 or road_coverage_pct < 3.0:
                        alert_lvl, hud_color = "NORMAL", (0, 255, 0)
                    elif w_score < 20.0:
                        alert_lvl, hud_color = "LOW", (0, 255, 255)
                    elif w_score < 40.0:
                        alert_lvl, hud_color = "MODERATE", (0, 165, 255)
                    else:
                        alert_lvl, hud_color = "CRITICAL", (0, 0, 255)

                    gps_data = manager.fetch_phone_gps()
                    file_id = f"live_{int(now * 1000)}"

                    # Visual HUD Overlay
                    overlay = frame.copy()
                    overlay[full_mask == 1] = [255, 191, 0]
                    annotated = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

                    annotated = draw_waterlog_hud(
                        annotated, w, h, alert_lvl, road_coverage_pct, w_score, avg_score,
                        gps_data['lat'], gps_data['lon'], mean_conf, hud_color, latency_ms
                    )

                    # 1. Save frame to all_annotated_frames
                    cv2.imwrite(os.path.join(manager.dir_all_frames, f"{file_id}.jpg"), annotated)

                    # 2. Write to MP4 (Only write when valid frame is present)
                    if manager.video_writer is not None:
                        manager.video_writer.write(annotated)

                    # 3. Incident Clustering & Telemetry
                    manager.register_incident(
                        annotated, file_id, avg_score, mean_conf, w_score,
                        road_coverage_pct, gps_data["lat"], gps_data["lon"], alert_lvl, latency_ms
                    )

                    print(f"[+] Live Frame #{frame_idx:04d} | Hazard: {w_score:5.2f} ({alert_lvl}) | {latency_ms:.1f}ms")

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('s'), ord('S')):
                    auto_mode = not auto_mode
                    last_capture_time = now
                    print(f"[*] Auto-Capture toggled: {auto_mode}")
                elif key in (ord('q'), ord('Q')):
                    print("[*] Exit key pressed ('q').")
                    return

    finally:
        cv2.destroyAllWindows()
        manager.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Waterlogging Dashcam")
    parser.add_argument("--url", default=STREAM_URL, help="Stream endpoint URL")
    parser.add_argument("--output_dir", default="waterlog_live_outputs", help="Output directory")
    parser.add_argument("--fps", type=float, default=TARGET_FPS, help="Target FPS sampling")
    args = parser.parse_args()

    run_live_pipeline(stream_url=args.url, output_root=args.output_dir, target_fps=args.fps)