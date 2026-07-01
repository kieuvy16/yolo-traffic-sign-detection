"""
Traffic Sign Detection - Optimized for low latency / Raspberry Pi deployment
- Prioritizes newest frame to reduce lag
- Threaded camera capture
- Real processing FPS measurement
- Optional frame skipping and lightweight overlay
- Safe fallbacks for laptop/webcam and Raspberry Pi camera

Original source provided by user: fileciteturn1file0
"""

from __future__ import annotations

import os
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
from ultralytics import YOLO


# =========================
# Configuration
# =========================
MODEL_CANDIDATES = [
    "src/models/yolov8/weights/best.pt",
    "models/yolov8/weights/best.pt",
    "best.pt",
    "yolov8n.pt",  # fallback
]


@dataclass
class AppConfig:
    camera_index: int = 0
    width: int = 640
    height: int = 480
    camera_fps_request: int = 30
    inference_imgsz: int = 416
    conf_threshold: float = 0.45
    iou_threshold: float = 0.45
    max_det: int = 20
    use_half: bool = False          # only meaningful on supported GPU backends
    device: Optional[str] = None    # e.g. "cpu", "0"
    skip_frames: int = 0            # 0 = infer every frame, 1 = every 2nd frame, etc.
    buffer_size: int = 1
    save_dir: str = "captured_images"
    show_window: bool = True
    draw_boxes: bool = True
    log_every_seconds: float = 1.5
    warmup_runs: int = 1
    preferred_backend: Optional[int] = None  # can set cv2.CAP_V4L2 on Linux/RPi


CONFIG = AppConfig(
    camera_index=0,
    width=640,
    height=480,
    camera_fps_request=30,
    inference_imgsz=416,
    conf_threshold=0.45,
    iou_threshold=0.45,
    max_det=20,
    use_half=False,
    device=None,
    skip_frames=0,
    buffer_size=1,
    save_dir="captured_images",
    show_window=True,
    draw_boxes=True,
    log_every_seconds=1.5,
    warmup_runs=1,
    preferred_backend=cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else None,
)


# =========================
# Camera thread
# =========================
class LatestFrameCamera:
    """Continuously grabs frames and keeps only the newest one to minimize latency."""

    def __init__(self, index: int, width: int, height: int, fps: int,
                 buffer_size: int = 1, backend: Optional[int] = None) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size
        self.backend = backend

        self.cap: Optional[cv2.VideoCapture] = None
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_timestamp = 0.0
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.frames_grabbed = 0

    def open(self) -> bool:
        if self.backend is not None:
            self.cap = cv2.VideoCapture(self.index, self.backend)
        else:
            self.cap = cv2.VideoCapture(self.index)

        if not self.cap.isOpened():
            return False

        # Best-effort camera tuning. Some backends may ignore some properties.
        

        # Disable autofocus when available to reduce focus hunting in motion.
        if hasattr(cv2, "CAP_PROP_AUTOFOCUS"):
            try:
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            except Exception:
                pass

        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        return True

    def _reader(self) -> None:
        while self.running and self.cap is not None:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            ts = time.perf_counter()
            with self.lock:
                self.latest_frame = frame
                self.latest_timestamp = ts
                self.frames_grabbed += 1

    def read_latest(self) -> Tuple[bool, Optional[any], float]:
        with self.lock:
            if self.latest_frame is None:
                return False, None, 0.0
            return True, self.latest_frame.copy(), self.latest_timestamp

    def release(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()


# =========================
# Utility helpers
# =========================
def find_model_path() -> str:
    for path in MODEL_CANDIDATES:
        if os.path.exists(path) or path == "yolov8n.pt":
            return path
    return "yolov8n.pt"


def load_model(config: AppConfig) -> Optional[YOLO]:
    model_path = find_model_path()
    try:
        model = YOLO(model_path)
        print(f"✅ Model loaded: {model_path}")
        print(f"📋 Classes: {list(model.names.values())}")
        return model
    except Exception as exc:
        print(f"❌ Failed to load model: {exc}")
        return None


def draw_detections(frame, results, model_names) -> Tuple[any, int]:
    annotated = frame.copy()
    boxes = results[0].boxes
    det_count = 0

    if boxes is None or len(boxes) == 0:
        return annotated, 0

    for box in boxes:
        det_count += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{model_names.get(cls_id, str(cls_id))} {conf:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            lineType=cv2.LINE_AA,
        )

    return annotated, det_count


def put_status(frame, process_fps: float, display_fps: float, latency_ms: float,
               detections: int, paused: bool, infer_size: int, skip_frames: int) -> None:
    h, w = frame.shape[:2]
    status_lines = [
        f"Process FPS: {process_fps:.1f}",
        f"Display FPS: {display_fps:.1f}",
        f"Latency: {latency_ms:.1f} ms",
        f"Detections: {detections}",
        f"imgsz: {infer_size} | skip: {skip_frames}",
        "Q: quit | S: save | P: pause | D: draw on/off | +/-: imgsz | K: skip toggle",
    ]

    y = 25
    for line in status_lines[:-1]:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        y += 28

    footer = status_lines[-1]
    cv2.putText(frame, footer, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    if paused:
        cv2.putText(frame, "PAUSED", (w // 2 - 70, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3, cv2.LINE_AA)


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# =========================
# Main loop
# =========================
def run_camera_detection(config: AppConfig = CONFIG) -> None:
    safe_mkdir(config.save_dir)

    model = load_model(config)
    if model is None:
        return

    print("📷 Opening camera...")
    camera = LatestFrameCamera(
        index=config.camera_index,
        width=config.width,
        height=config.height,
        fps=config.camera_fps_request,
        buffer_size=config.buffer_size,
        backend=config.preferred_backend,
    )

    if not camera.open():
        print("❌ Cannot open camera. Check device index, permissions, or backend.")
        return

    print("✅ Camera ready")
    print("=" * 60)
    print("Tips for speed:")
    print("- Use 640x480 or 960x540")
    print("- Use a small model")
    print("- Increase lighting to reduce blur")
    print("- For Raspberry Pi, prefer newer Pi + accelerator/GPU if available")
    print("=" * 60)

    # Warmup on a dummy frame if available to reduce first inference delay.
    warmed_up = False
    for _ in range(100):
        ok, frame, _ = camera.read_latest()
        if ok:
            for _ in range(config.warmup_runs):
                _ = model.predict(
                    source=frame,
                    conf=config.conf_threshold,
                    iou=config.iou_threshold,
                    imgsz=config.inference_imgsz,
                    device=config.device,
                    half=config.use_half,
                    max_det=config.max_det,
                    verbose=False,
                )
            warmed_up = True
            break
        time.sleep(0.01)
    if warmed_up:
        print("🔥 Warmup complete")

    paused = False
    draw_boxes = config.draw_boxes
    screenshot_count = 0
    processed_frames = 0
    last_log_time = time.perf_counter()
    last_display_time = time.perf_counter()
    process_times = deque(maxlen=30)
    display_times = deque(maxlen=30)

    last_result = None
    last_det_count = 0
    last_annotated = None
    last_frame_ts = 0.0

    try:
        while True:
            loop_start = time.perf_counter()

            if not paused:
                ok, frame, frame_ts = camera.read_latest()
                if not ok:
                    time.sleep(0.002)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), ord('Q')):
                        break
                    continue

                # Avoid re-processing the exact same buffered frame.
                if frame_ts == last_frame_ts:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), ord('Q')):
                        break
                    continue
                last_frame_ts = frame_ts
                processed_frames += 1

                should_infer = (config.skip_frames == 0) or (processed_frames % (config.skip_frames + 1) == 0)

                if should_infer:
                    infer_start = time.perf_counter()
                    last_result = model.predict(
                        source=frame,
                        conf=config.conf_threshold,
                        iou=config.iou_threshold,
                        imgsz=config.inference_imgsz,
                        device=config.device,
                        half=config.use_half,
                        max_det=config.max_det,
                        verbose=False,
                    )
                    infer_dt = time.perf_counter() - infer_start
                    process_times.append(infer_dt)

                    boxes = last_result[0].boxes
                    last_det_count = 0 if boxes is None else len(boxes)
                    

                    if draw_boxes:
                        last_annotated, last_det_count = draw_detections(frame, last_result, model.names)
                    else:
                        last_annotated = frame
                else:
                    # Reuse latest raw frame to keep display responsive even when skipping inference.
                    last_annotated = frame

                if last_annotated is None:
                    last_annotated = frame

                now = time.perf_counter()
                latency_ms = max(0.0, (now - frame_ts) * 1000.0)
                process_fps = (1.0 / (sum(process_times) / len(process_times))) if process_times else 0.0
                dt_display = now - last_display_time
                if dt_display > 0:
                    display_times.append(dt_display)
                last_display_time = now
                display_fps = (1.0 / (sum(display_times) / len(display_times))) if display_times else 0.0

                overlay_frame = last_annotated.copy()
                put_status(
                    overlay_frame,
                    process_fps=process_fps,
                    display_fps=display_fps,
                    latency_ms=latency_ms,
                    detections=last_det_count,
                    paused=paused,
                    infer_size=config.inference_imgsz,
                    skip_frames=config.skip_frames,
                )

                if config.show_window:
                    cv2.imshow("Traffic Sign Detection - Optimized", overlay_frame)

                if now - last_log_time >= config.log_every_seconds:
                    print(
                        f"[INFO] process_fps={process_fps:.1f} | display_fps={display_fps:.1f} | "
                        f"latency={latency_ms:.1f}ms | det={last_det_count} | imgsz={config.inference_imgsz} | skip={config.skip_frames}"
                    )
                    last_log_time = now
            else:
                if last_annotated is not None and config.show_window:
                    paused_frame = last_annotated.copy()
                    put_status(
                        paused_frame,
                        process_fps=(1.0 / (sum(process_times) / len(process_times))) if process_times else 0.0,
                        display_fps=(1.0 / (sum(display_times) / len(display_times))) if display_times else 0.0,
                        latency_ms=0.0,
                        detections=last_det_count,
                        paused=True,
                        infer_size=config.inference_imgsz,
                        skip_frames=config.skip_frames,
                    )
                    cv2.imshow("Traffic Sign Detection - Optimized", paused_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                print("👋 Exiting...")
                break
            elif key in (ord('p'), ord('P')):
                paused = not paused
                print("⏸️ Paused" if paused else "▶️ Resumed")
            elif key in (ord('s'), ord('S')):
                if last_annotated is not None:
                    screenshot_count += 1
                    filename = os.path.join(config.save_dir, f"screenshot_{screenshot_count}.jpg")
                    cv2.imwrite(filename, last_annotated)
                    print(f"📸 Saved: {filename}")
            elif key in (ord('d'), ord('D')):
                draw_boxes = not draw_boxes
                print(f"🖼️ Draw boxes: {'ON' if draw_boxes else 'OFF'}")
            elif key == ord('+') or key == ord('='):
                config.inference_imgsz = min(960, config.inference_imgsz + 32)
                print(f"🔎 imgsz increased to {config.inference_imgsz}")
            elif key == ord('-') or key == ord('_'):
                config.inference_imgsz = max(256, config.inference_imgsz - 32)
                print(f"🔎 imgsz decreased to {config.inference_imgsz}")
            elif key in (ord('k'), ord('K')):
                config.skip_frames = 0 if config.skip_frames > 0 else 1
                print(f"⏭️ skip_frames set to {config.skip_frames}")

            _ = loop_start  # reserved for future profiling hooks

    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        avg_proc_fps = (1.0 / (sum(process_times) / len(process_times))) if process_times else 0.0
        print("✅ Camera released and windows closed")
        print(f"📊 Frames grabbed: {camera.frames_grabbed}")
        print(f"📊 Frames processed: {processed_frames}")
        print(f"📊 Avg process FPS: {avg_proc_fps:.1f}")


if __name__ == "__main__":
    print("=" * 60)
    print("🚦 TRAFFIC SIGN DETECTION - OPTIMIZED CAMERA MODE")
    print("🚀 Low latency / Raspberry Pi oriented")
    print("=" * 60)
    run_camera_detection()
