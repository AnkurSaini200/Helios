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
from ultralytics import YOLO

# ======================= CONFIGURATION =======================
DEFAULT_PHONE_IP = "10.2.5.75"
DEFAULT_PHONE_PORT = "8080"
DEFAULT_URL = f"http://{DEFAULT_PHONE_IP}:{DEFAULT_PHONE_PORT}/video"
TARGET_FPS = 4.0

VEHICLE_WEIGHTS = {1: 0.5, 2: 1.0, 3: 0.5, 5: 3.5, 7: 3.0}
COCO_NAMES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CLASS_COLORS = {
    1: (255, 178, 50),
    2: (255, 255, 255),
    3: (0, 215, 255),
    5: (203, 192, 255),
    7: (180, 105, 255)
}

GEO_CACHE = {}


def get_road_name(lat, lon):
    key = (round(lat, 4), round(lon, 4))
    if key in GEO_CACHE:
        return GEO_CACHE[key]
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"
        headers = {"User-Agent": "SIH-Traffic-LiveCam/2.0"}
        resp = requests.get(url, headers=headers, timeout=2.0).json()
        road = resp.get("address", {}).get("road") or "Patamata Lanka"
        GEO_CACHE[key] = road
        return road
    except Exception:
        return "Patamata Lanka"


def compute_traffic_density(detections, img_shape, road_length_m=65.0, lanes=2):
    """
    Adaptive density estimation that computes density relative to the active traffic
    corridor instead of the raw camera frame size, keeping scores synchronized
    between direct dashcam video files and external camera captures.
    """
    if not detections:
        return 0.0, 0.0, "Low (Free Flow)", (46, 204, 113)

    img_h, img_w = img_shape[:2]

    # Calculate active bounding cluster of vehicles to establish effective road area
    min_x = min(d["box"][0] for d in detections)
    max_x = max(d["box"][2] for d in detections)
    min_y = min(d["box"][1] for d in detections)
    max_y = max(d["box"][3] for d in detections)

    cluster_w = max(int(img_w * 0.45), max_x - min_x)
    cluster_h = max(int(img_h * 0.45), max_y - min_y)
    effective_road_area = float(cluster_w * cluster_h)

    detected_classes = [d["class_id"] for d in detections]
    total_pcu = sum(VEHICLE_WEIGHTS.get(c, 1.0) for c in detected_classes)
    n_vehicles = len(detections)

    total_vehicle_pixel_area = sum(d["area"] for d in detections)
    occupancy_ratio = min(1.0, total_vehicle_pixel_area / (effective_road_area * 0.70))

    jam_buffer_m = 7.0
    nominal_pcu_cap = (road_length_m / jam_buffer_m) * lanes
    load_ratio = total_pcu / nominal_pcu_cap

    vertical_span = (max_y - min_y) / float(img_h)

    stress_index = (0.55 * occupancy_ratio) + (0.45 * min(1.6, load_ratio))
    if vertical_span > 0.35 and n_vehicles >= 7:
        stress_index += 0.12

    if stress_index < 0.40:
        density_pct = (stress_index / 0.40) * 38.0
    elif stress_index < 0.75:
        density_pct = 38.0 + ((stress_index - 0.40) / 0.35) * 34.0
    else:
        excess = stress_index - 0.75
        saturation_curve = 1.0 - np.exp(-1.8 * excess)
        density_pct = 78.0 + (saturation_curve * 19.5)

    density_pct = min(98.5, max(5.0, density_pct))

    if density_pct < 40.0:
        status = "Low (Free Flow)"
        color_bgr = (46, 204, 113)
    elif density_pct < 75.0:
        status = "Moderate"
        color_bgr = (0, 165, 255)
    else:
        status = "Heavy (Congestion)"
        color_bgr = (50, 50, 220)

    return round(float(density_pct), 2), round(float(total_pcu), 1), status, color_bgr


def clean_tracked_detections(boxes, img_shape):
    if len(boxes) == 0:
        return []

    img_h, img_w = img_shape[:2]
    min_pixel_area = (img_h * img_w) * 0.0008

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(xyxy)

    candidates = []
    for i in range(len(xyxy)):
        w = xyxy[i, 2] - xyxy[i, 0]
        h = xyxy[i, 3] - xyxy[i, 1]
        area = w * h
        cid = classes[i]
        conf = confs[i]

        if area < min_pixel_area or w < 12 or h < 12:
            continue
        if (h / float(w) > 3.0) and area < (img_h * img_w * 0.005):
            continue

        if cid in (5, 7) and conf < 0.65:
            cid = 2

        candidates.append({
            "box": xyxy[i].astype(int),
            "conf": float(conf),
            "class_id": int(cid),
            "track_id": track_ids[i],
            "area": float(area)
        })

    n = len(candidates)
    if n == 0:
        return []

    # Strict spatial deduplication: merge duplicates without merging adjacent bumper-to-bumper cars
    adj = [[] for _ in range(n)]
    for i in range(n):
        b1 = candidates[i]["box"]
        a1 = candidates[i]["area"]
        for j in range(i + 1, n):
            b2 = candidates[j]["box"]
            a2 = candidates[j]["area"]

            ix1 = max(b1[0], b2[0])
            iy1 = max(b1[1], b2[1])
            ix2 = min(b1[2], b2[2])
            iy2 = min(b1[3], b2[3])

            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                smaller = min(a1, a2)
                iou = inter / float(a1 + a2 - inter)
                if (inter / smaller) > 0.80 or iou > 0.65:
                    adj[i].append(j)
                    adj[j].append(i)

    visited = [False] * n
    merged_detections = []
    for i in range(n):
        if visited[i]:
            continue
        component = []
        stack = [i]
        visited[i] = True
        while stack:
            curr = stack.pop()
            component.append(curr)
            for neighbor in adj[curr]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)

        comp_boxes = [candidates[k]["box"] for k in component]
        x1 = min(b[0] for b in comp_boxes)
        y1 = min(b[1] for b in comp_boxes)
        x2 = max(b[2] for b in comp_boxes)
        y2 = max(b[3] for b in comp_boxes)

        best_conf = max(candidates[k]["conf"] for k in component)
        t_id = next((candidates[k]["track_id"] for k in component if candidates[k]["track_id"] is not None), None)
        comp_classes = [candidates[k]["class_id"] for k in component]
        dominant_cls = 5 if (5 in comp_classes) else (7 if (7 in comp_classes) else comp_classes[0])

        merged_detections.append({
            "box": np.array([x1, y1, x2, y2]),
            "conf": best_conf,
            "class_id": dominant_cls,
            "track_id": t_id,
            "area": float((x2 - x1) * (y2 - y1))
        })
    return merged_detections


def draw_hud_and_boxes(image_bgr, detections, density_pct, status, color_bgr, total_pcu, road_name):
    # 1. Bounding Boxes
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cid = det["class_id"]
        t_id = det["track_id"]

        label = f"{COCO_NAMES.get(cid, 'vehicle')}"
        if t_id is not None:
            label += f" #{t_id}"

        box_color = CLASS_COLORS.get(cid, (255, 255, 255))
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), box_color, 2)

        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        top_y = max(y1, label_size[1] + 5)
        cv2.rectangle(image_bgr, (x1, top_y - label_size[1] - 4), (x1 + label_size[0] + 4, top_y), (30, 30, 30), -1)
        cv2.putText(image_bgr, label, (x1 + 2, top_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)

    # 2. Responsive HUD Banner
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
    cv2.putText(image_bgr, f"Traffic Flow: {density_pct}%", (text_x, 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image_bgr, f"Vehicles: {len(detections)} | PCU: {total_pcu} | Road: {road_name}", (text_x, 86), cv2.FONT_HERSHEY_SIMPLEX, sub_font_scale, (220, 220, 220), 1, cv2.LINE_AA)

    return image_bgr


def run_live_cam(stream_url=DEFAULT_URL, output_dir="outputs", target_fps=TARGET_FPS):
    all_frames_dir = os.path.join(output_dir, "all_annotated_frames")
    global_best_dir = os.path.join(output_dir, "global_best")
    json_telemetry_dir = os.path.join(output_dir, "json_telemetry")

    for d in [all_frames_dir, global_best_dir, json_telemetry_dir]:
        os.makedirs(d, exist_ok=True)

    print("[*] Loading YOLO model: yolo26n.pt ...")
    model = YOLO("yolo26n.pt")

    video_writer = None
    video_out_path = os.path.join(output_dir, "output_traffic.mp4")

    best_incident = {"traffic_flow_percent": -1.0, "saved_img_path": None, "record": None}

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

    print(f"[*] Starting Reconnecting Loop for: {stream_url}")

    try:
        while True:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;3000000|max_delay;500000"
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                print("[!] Camera stream unavailable. Retrying in 2s...")
                time.sleep(2)
                continue

            print("[+] Live Camera Connected! Processing feed...")
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
                cv2.imshow("Live Traffic Feed (Press 'q' to Quit)", frame)

                if now - last_capture_time >= sample_interval:
                    last_capture_time = now
                    frame_idx += 1

                    # Inference with imgsz=960 to detect distant vehicles reliably
                    results = model.track(
                        source=frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        classes=list(VEHICLE_WEIGHTS.keys()),
                        conf=0.25,
                        iou=0.45,
                        imgsz=960,
                        verbose=False
                    )[0]

                    clean_detections = clean_tracked_detections(results.boxes, frame.shape)
                    detected_classes = [d["class_id"] for d in clean_detections]
                    density_pct, total_pcu, status, color_bgr = compute_traffic_density(clean_detections, frame.shape)

                    annotated = draw_hud_and_boxes(
                        frame, clean_detections, density_pct, status, color_bgr, total_pcu, road_name
                    )

                    # 1. Folder: all_annotated_frames
                    file_id = f"frame_{frame_idx:05d}_{int(now * 1000)}"
                    frame_save_path = os.path.join(all_frames_dir, f"{file_id}.jpg")
                    cv2.imwrite(frame_save_path, annotated)

                    # 2. Write to MP4
                    if video_writer is not None:
                        video_writer.write(annotated)

                    # 3. Individual JSON
                    record = {
                        "frame_index": frame_idx,
                        "timestamp": round(now, 2),
                        "road_name": road_name,
                        "coordinates": {"lat": lat, "lon": lon},
                        "traffic_flow_percent": density_pct,
                        "status": status,
                        "total_vehicles": len(clean_detections),
                        "total_pcu": total_pcu,
                        "active_track_ids": [d["track_id"] for d in clean_detections if d["track_id"] is not None],
                        "breakdown": {name: detected_classes.count(cid) for cid, name in COCO_NAMES.items()}
                    }
                    with open(os.path.join(json_telemetry_dir, f"{file_id}.json"), "w") as jf:
                        json.dump(record, jf, indent=4)

                    # 4. Global Best
                    if density_pct > best_incident["traffic_flow_percent"]:
                        best_incident["traffic_flow_percent"] = density_pct
                        best_incident["saved_img_path"] = frame_save_path
                        best_incident["record"] = record

                        cv2.imwrite(os.path.join(global_best_dir, "peak_traffic_incident.jpg"), annotated)
                        with open(os.path.join(global_best_dir, "peak_traffic_incident.json"), "w") as pf:
                            json.dump(record, pf, indent=4)
                        print(f"[★ GLOBAL PEAK TRAFFIC] {density_pct}% -> Saved to '{global_best_dir}/'")

                    print(f"[+] Frame #{frame_idx:04d} | Status: {status} ({density_pct}%) | Active: {len(clean_detections)}")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[*] 'q' pressed. Exiting...")
                    return

    finally:
        cv2.destroyAllWindows()
        close_resources()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Phone Camera Traffic Flow Pipeline")
    parser.add_argument("--url", default=DEFAULT_URL, help="IP Webcam video endpoint URL")
    parser.add_argument("--output_dir", default="outputs", help="Base directory for output folders")
    parser.add_argument("--fps", type=float, default=TARGET_FPS, help="Target sampling FPS")
    args = parser.parse_args()

    run_live_cam(stream_url=args.url, output_dir=args.output_dir, target_fps=args.fps)