import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import requests
from detector import detect_potholes

# ======================= CONFIGURATION =======================
DEFAULT_PHONE_IP = "10.2.5.75"
DEFAULT_PHONE_PORT = "8080"
DEFAULT_URL = f"http://{DEFAULT_PHONE_IP}:{DEFAULT_PHONE_PORT}/video"
TARGET_FPS = 4.0

GEO_CACHE = {}


def get_road_name(lat, lon):
    key = (round(lat, 4), round(lon, 4))
    if key in GEO_CACHE:
        return GEO_CACHE[key]
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"
        headers = {"User-Agent": "SIH-Pothole-LiveCam/2.0"}
        resp = requests.get(url, headers=headers, timeout=2.0).json()
        road = resp.get("address", {}).get("road") or "Patamata Lanka"
        GEO_CACHE[key] = road
        return road
    except Exception:
        return "Patamata Lanka"


def compute_pothole_severity(detections, img_shape):
    """
    Computes severity metrics based on pothole count, coverage area,
    and confidence density relative to the frame size.
    """
    if not detections:
        return 0.0, "Low (Safe Road)", (46, 204, 113)

    img_h, img_w = img_shape[:2]
    total_frame_area = float(img_h * img_w)
    total_pothole_area = sum(
        float((d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
        for d in detections
    )

    coverage_ratio = total_pothole_area / total_frame_area
    count = len(detections)
    avg_conf = sum(d["confidence"] for d in detections) / count

    # Severity scoring algorithm (0 to 100 scale)
    severity_score = (min(1.0, coverage_ratio * 15.0) * 50.0) + \
                     (min(1.0, count / 5.0) * 30.0) + \
                     (avg_conf * 20.0)

    severity_score = min(99.0, max(5.0, severity_score))

    if severity_score < 35.0:
        status = "Low (Minor Cracks)"
        color_bgr = (46, 204, 113)  # Green
    elif severity_score < 70.0:
        status = "Moderate (Repair Needed)"
        color_bgr = (0, 165, 255)   # Orange
    else:
        status = "Severe (Hazardous)"
        color_bgr = (50, 50, 220)   # Red

    return round(float(severity_score), 2), status, color_bgr


def draw_hud_and_boxes(image_bgr, detections, severity_pct, status, color_bgr, road_name):
    # 1. Draw Bounding Boxes using detector output format
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        conf = det["confidence"]
        label = f"pothole {conf:.2f}"

        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        top_y = max(y1, label_size[1] + 5)
        cv2.rectangle(image_bgr, (x1, top_y - label_size[1] - 4), (x1 + label_size[0] + 4, top_y), (30, 30, 30), -1)
        cv2.putText(image_bgr, label, (x1 + 2, top_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    # 2. Responsive HUD Banner Overlay
    h, w, _ = image_bgr.shape
    banner_width = min(w - 20, int(w * 0.95))
    banner_height = 105

    overlay = image_bgr.copy()
    cv2.rectangle(overlay, (10, 10), (10 + banner_width, 10 + banner_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, image_bgr, 0.25, 0, image_bgr)

    cv2.rectangle(image_bgr, (15, 15), (23, 10 + banner_height - 6), color_bgr, -1)

    font_scale = 0.58 if w >= 720 else 0.45
    sub_font_scale = 0.48 if w >= 720 else 0.38
    text_x = 32

    cv2.putText(image_bgr, f"Status: {status}", (text_x, 34), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_bgr, 2, cv2.LINE_AA)
    cv2.putText(image_bgr, f"Severity Index: {severity_pct}%", (text_x, 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image_bgr, f"Potholes Detected: {len(detections)} | Road: {road_name}", (text_x, 86), cv2.FONT_HERSHEY_SIMPLEX, sub_font_scale, (220, 220, 220), 1, cv2.LINE_AA)

    return image_bgr


def run_live_pothole_cam(stream_url=DEFAULT_URL, output_dir="outputs_pothole", target_fps=TARGET_FPS):
    all_frames_dir = os.path.join(output_dir, "all_annotated_frames")
    global_best_dir = os.path.join(output_dir, "global_best")
    json_telemetry_dir = os.path.join(output_dir, "json_telemetry")

    for d in [all_frames_dir, global_best_dir, json_telemetry_dir]:
        os.makedirs(d, exist_ok=True)

    print("[*] Initializing Pothole Detection Live Stream Pipeline...")

    video_writer = None
    video_out_path = os.path.join(output_dir, "output_potholes.mp4")

    best_incident = {"severity_score": -1.0, "saved_img_path": None, "record": None}

    def close_resources():
        nonlocal video_writer
        if video_writer is not None:
            video_writer.release()
            video_writer = None
            print(f"[+] Output video finalized and closed: '{video_out_path}'")

    def handle_signal(sig, frame):
        print("\n[*] Interrupted by user. Finalizing video...")
        close_resources()
        cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    lat, lon = 16.5062, 80.6480
    road_name = get_road_name(lat, lon)
    sample_interval = 1.0 / target_fps
    last_capture_time = 0.0
    frame_idx = 0

    print(f"[*] Starting Reconnecting Loop for IP Webcam: {stream_url}")

    try:
        while True:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;3000000|max_delay;500000"
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                print("[!] Camera stream unavailable. Retrying in 2s...")
                time.sleep(2)
                continue

            print("[+] Live Camera Connected! Processing pothole feed...")
            consecutive_invalid = 0

            while True:
                ret, frame = cap.read()

                # Filter out blank, corrupt, or dropped frames
                if not ret or frame is None or frame.size == 0 or frame.mean() < 8.0:
                    consecutive_invalid += 1
                    if consecutive_invalid >= 4:
                        print("[!] Feed interrupted / blank frame detected. Reconnecting...")
                        cap.release()
                        time.sleep(1.0)
                        break
                    time.sleep(0.05)
                    continue

                consecutive_invalid = 0
                h, w, _ = frame.shape

                # Initialize Video Writer with Windows H264-compatible encoder
                if video_writer is None:
                    encoders = [
                        ('H264', cv2.VideoWriter_fourcc(*'H264')),
                        ('avc1', cv2.VideoWriter_fourcc(*'avc1')),
                        ('mp4v', cv2.VideoWriter_fourcc(*'mp4v'))
                    ]
                    for tag, fourcc in encoders:
                        wrt = cv2.VideoWriter(video_out_path, fourcc, target_fps, (w, h))
                        if wrt.isOpened():
                            video_writer = wrt
                            print(f"[+] Recording video using codec '{tag}' -> '{video_out_path}'")
                            break

                now = time.time()
                cv2.imshow("Live Pothole Feed (Press 'q' to Quit)", frame)

                if now - last_capture_time >= sample_interval:
                    last_capture_time = now
                    frame_idx += 1

                    # Call your custom image-based detector.py function
                    detections = detect_potholes(frame)
                    severity_pct, status, color_bgr = compute_pothole_severity(detections, frame.shape)

                    annotated = draw_hud_and_boxes(
                        frame, detections, severity_pct, status, color_bgr, road_name
                    )

                    # 1. Save frame to all_annotated_frames
                    file_id = f"pothole_frame_{frame_idx:05d}_{int(now * 1000)}"
                    frame_save_path = os.path.join(all_frames_dir, f"{file_id}.jpg")
                    cv2.imwrite(frame_save_path, annotated)

                    # 2. Write to MP4 output video
                    if video_writer is not None:
                        video_writer.write(annotated)

                    # 3. Save individual JSON telemetry record
                    record = {
                        "frame_index": frame_idx,
                        "timestamp": round(now, 2),
                        "road_name": road_name,
                        "coordinates": {"lat": lat, "lon": lon},
                        "pothole_severity_percent": severity_pct,
                        "status": status,
                        "total_potholes": len(detections),
                        "detections": detections
                    }
                    with open(os.path.join(json_telemetry_dir, f"{file_id}.json"), "w") as jf:
                        json.dump(record, jf, indent=4)

                    # 4. Check & Update Global Peak Incident (Worst Pothole Hazard)
                    if severity_pct > best_incident["severity_score"]:
                        best_incident["severity_score"] = severity_pct
                        best_incident["saved_img_path"] = frame_save_path
                        best_incident["record"] = record

                        cv2.imwrite(os.path.join(global_best_dir, "peak_pothole_incident.jpg"), annotated)
                        with open(os.path.join(global_best_dir, "peak_pothole_incident.json"), "w") as pf:
                            json.dump(record, pf, indent=4)
                        print(f"[★ PEAK HAZARD RECORDED] Severity: {severity_pct}% -> Saved to '{global_best_dir}/'")

                    print(f"[+] Frame #{frame_idx:04d} | Status: {status} ({severity_pct}%) | Potholes Found: {len(detections)}")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[*] 'q' pressed. Exiting...")
                    return

    finally:
        cv2.destroyAllWindows()
        close_resources()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Phone Camera Pothole Detection Pipeline")
    parser.add_argument("--url", default=DEFAULT_URL, help="IP Webcam video endpoint URL")
    parser.add_argument("--output_dir", default="outputs_pothole", help="Base directory for output folders")
    parser.add_argument("--fps", type=float, default=TARGET_FPS, help="Target sampling FPS")
    args = parser.parse_args()

    run_pothole_cam = run_live_pothole_cam
    run_pothole_cam(stream_url=args.url, output_dir=args.output_dir, target_fps=args.fps)