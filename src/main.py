from __future__ import annotations

from flask import Flask, request, render_template, jsonify, Response
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageFile, ImageDraw, ImageFont
import base64
import io
import os
import threading
import time
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Any

ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(BASE_DIR / "models" / "yolov8" / "best.pt"),
    )
)
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR),
)
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])
app.config["MAX_CONTENT_LENGTH"] = int(
    os.getenv("MAX_CONTENT_LENGTH", str(16 * 1024 * 1024))
)

INFERENCE_IMGSZ = int(os.getenv("INFERENCE_IMGSZ", "416"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.45"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
MAX_DET = int(os.getenv("MAX_DET", "20"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "80"))
RETURN_JPEG = os.getenv("RETURN_JPEG", "1") == "1"
MODEL_DEVICE = "0" if torch.cuda.is_available() else "cpu"
USE_HALF = os.getenv("USE_HALF", "0") == "1"
VERBOSE_LOGS = os.getenv("VERBOSE_LOGS", "0") == "1"
REQUEST_GAP_MS = int(os.getenv("REQUEST_GAP_MS", "0"))

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}

SIGN_NAMES_VIETNAMESE = {
    "W.224": "Đường người đi bộ cắt ngang",
    "W.205c": "Đường giao nhau ngã ba bên phải",
    "P.102": "Cấm đi ngược chiều",
    "R.302a": "Phải đi vòng sang bên phải",
    "W.205a": "Giao nhau với đường đồng cấp",
    "W.207": "Giao nhau với đường không ưu tiên",
    "W.201a": "Chỗ ngoặt nguy hiểm vòng bên trái",
    "P.123a": "Cấm quay đầu",
    "I.434a": "Bến xe buýt",
    "R.303": "Nơi giao nhau chạy theo vòng xuyến",
    "P.130": "Cấm dừng và đỗ xe",
    "I.409": "Chỗ quay xe",
    "R.415a": "Biển gộp làn đường theo phương tiện",
    "W.245a": "Đi chậm",
    "P.106a*Xe tai": "Cấm xe tải",
    "W.203c": "Đường bị thu hẹp về phía phải",
    "P.117*": "Giới hạn chiều cao",
    "P.124a*": "Cấm quay đầu",
    "P.107": "Cấm ô tô khách và ô tô tải",
    "P.124d": "Cấm rẽ phải và quay đầu",
    "P.103a": "Cấm ô tô",
    "W.203b": "Đường bị thu hẹp về phía trái",
    "W.221b": "Gồ giảm tốc phía trước",
    "P.111": "Cấm xe hai và ba bánh",
    "P.129": "Kiểm tra",
    "S.505a*Xe may": "Chỉ dành cho xe máy",
    "W.246a": "Chướng ngại vật phía trước",
    "W.225": "Trẻ em",
    "S.505a*Xe tai va cong": "Xe tải và xe công",
    "P.104": "Cấm mô tô và xe máy",
    "S.505a*Xe tai": "Chỉ dành cho xe tải",
    "Camera": "Đường có camera giám sát",
    "P.123b": "Cấm rẽ phải",
    "W.202b": "Nhiều chỗ ngoặt nguy hiểm liên tiếp, chỗ đầu tiên sang phải",
    "B.8a": "Cấm xe sơ mi rơ moóc",
    "P.137": "Cấm rẽ trái và phải",
    "P.139": "Cấm đi thẳng và rẽ phải",
    "W.205b": "Đường giao nhau ngã ba bên trái",
    "P.127*50": "Giới hạn tốc độ 50 km một giờ",
    "P.127*60": "Giới hạn tốc độ 60 km một giờ",
    "P.127*80": "Giới hạn tốc độ 80 km một giờ",
    "P.127*40": "Giới hạn tốc độ 40 km một giờ",
    "R.301e": "Các xe chỉ được rẽ trái",
    "W.239b*": "Chiều cao tĩnh không thực tế",
    "W.233": "Nguy hiểm khác",
    "I.407a": "Cấm đi ngược chiều",
    "P.131a": "Cấm đỗ xe",
    "P.124b1": "Cấm ô tô quay đầu xe được rẽ trái",
    "W.210": "Giao nhau với đường sắt có rào chắn",
    "P.124c": "Cấm rẽ trái và quay đầu xe",
    "W.201b": "Chỗ ngoặt nguy hiểm vòng bên phải",
    "W.246c": "Chú ý chướng ngại vật vòng tránh sang bên phải",
}

model = None
model_lock = threading.Lock()
request_gap_lock = threading.Lock()
last_request_at = 0.0


def log(*args):
    if VERBOSE_LOGS:
        print(*args)


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def load_model() -> None:
    global model
    try:
        if not MODEL_PATH.exists():
            print(f"❌ Model not found at: {MODEL_PATH}")
            model = None
            return

        model = YOLO(str(MODEL_PATH))
        print(f"✅ Loaded model from: {MODEL_PATH}")
        print(f"📋 Model classes ({len(model.names)}): {list(model.names.values())}")

        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        with model_lock:
            _ = model.predict(
                source=dummy,
                imgsz=320,
                conf=0.25,
                iou=0.45,
                max_det=1,
                verbose=False,
                device=MODEL_DEVICE,
                half=USE_HALF,
            )
        print("🔥 Model warmup completed")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    img_array = np.array(image)
    if image.mode == "L":
        return cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)


def encode_image_for_web(image_bgr: np.ndarray) -> str:
    ext = ".jpg" if RETURN_JPEG else ".png"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY] if RETURN_JPEG else []
    ok, encoded = cv2.imencode(ext, image_bgr, params)
    if not ok:
        raise RuntimeError("Failed to encode image")
    mime_prefix = "data:image/jpeg;base64," if RETURN_JPEG else "data:image/png;base64,"
    return mime_prefix + base64.b64encode(encoded.tobytes()).decode("utf-8")


def get_vietnamese_font(size: int = 22):
    candidates = [
        os.getenv("UNICODE_FONT_PATH", ""),
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def draw_detections(image_bgr: np.ndarray, detections: List[Dict[str, Any]], inference_ms: float) -> np.ndarray:
    annotated = image_bgr.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(annotated_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = get_vietnamese_font(22)
    small_font = get_vietnamese_font(20)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f'{det["display_name"]} {det["confidence"]:.2f}'

        try:
            left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
            text_w = right - left
            text_h = bottom - top
        except Exception:
            text_w = len(label) * 10
            text_h = 24

        tx = max(4, x1)
        ty = y1 - text_h - 12
        if ty < 4:
            ty = min(y1 + 6, max(4, pil_img.height - text_h - 12))

        bg = [
            tx,
            ty,
            min(tx + text_w + 12, pil_img.width - 4),
            min(ty + text_h + 8, pil_img.height - 4),
        ]
        draw.rounded_rectangle(bg, radius=6, fill=(15, 23, 42))
        draw.text((tx + 6, ty + 3), label, font=font, fill=(80, 255, 120))

    summary = f"Infer: {inference_ms:.1f} ms | Dets: {len(detections)} | imgsz: {INFERENCE_IMGSZ}"
    summary_box = [8, 8, min(430, pil_img.width - 8), 42]
    draw.rounded_rectangle(summary_box, radius=6, fill=(15, 23, 42))
    draw.text((16, 13), summary, font=small_font, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def process_image(image: Image.Image) -> Tuple[str, List[Dict[str, Any]], float]:
    if model is None:
        raise RuntimeError("Model not loaded")

    image_bgr = pil_to_bgr(image)
    start = time.perf_counter()
    with model_lock:
        results = model.predict(
            source=image_bgr,
            imgsz=INFERENCE_IMGSZ,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            max_det=MAX_DET,
            verbose=False,
            device=MODEL_DEVICE,
            half=USE_HALF,
        )
    inference_ms = (time.perf_counter() - start) * 1000.0

    if not results:
        return encode_image_for_web(image_bgr), [], inference_ms

    result = results[0]
    detections: List[Dict[str, Any]] = []
    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = model.names.get(cls_id, str(cls_id))

            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            detections.append(
                {
                    "class": class_name,
                    "display_name": SIGN_NAMES_VIETNAMESE.get(class_name, class_name),
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                }
            )

    detections.append(
        {
            "class": class_name,
            "display_name": SIGN_NAMES_VIETNAMESE.get(class_name, class_name),
            "confidence": confidence,
            "bbox": [x1, y1, x2, y2],
        }
    )
    annotated = draw_detections(image_bgr, detections, inference_ms)
    encoded = encode_image_for_web(annotated)
    return encoded, detections, inference_ms


@app.route("/")
def index():
    return render_template("index.html")



@app.route("/detect", methods=["POST"])
def detect():
    global last_request_at

    if REQUEST_GAP_MS > 0:
        with request_gap_lock:
            now_ms = time.time() * 1000.0
            delta = now_ms - last_request_at
            if delta < REQUEST_GAP_MS:
                time.sleep((REQUEST_GAP_MS - delta) / 1000.0)
            last_request_at = time.time() * 1000.0

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Supported: PNG, JPG, JPEG, GIF, BMP, WEBP"}), 400

    try:
        image = Image.open(file.stream)
        result_img, detections, inference_ms = process_image(image)
        return jsonify(
            {
                "success": True,
                "image": result_img,
                "detections": detections,
                "count": len(detections),
                "inference_ms": round(inference_ms, 2),
                "imgsz": INFERENCE_IMGSZ,
            }
        )
    except Exception as e:
        print(f"❌ Detect error: {e}")
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500


@app.route("/tts", methods=["POST"])
def tts():
    try:
        from gtts import gTTS

        data = request.get_json(silent=True) or {}
        text = str(data.get("text", "")).strip()
        if not text:
            return jsonify({"error": "Missing text"}), 400

        mp3_buffer = io.BytesIO()
        gTTS(text=text, lang="vi").write_to_fp(mp3_buffer)
        mp3_buffer.seek(0)

        return Response(
            mp3_buffer.read(),
            mimetype="audio/mpeg",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    except Exception as e:
        print(f"❌ TTS route error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    os.makedirs(STATIC_DIR, exist_ok=True)

    load_model()

    print("🚦 Traffic Sign Detection API ready")
    print(f"📦 MODEL_PATH={MODEL_PATH}")
    print(f"⚙️ INFERENCE_IMGSZ={INFERENCE_IMGSZ}, CONF_THRESHOLD={CONF_THRESHOLD}, DEVICE={MODEL_DEVICE}")

    try:
        from waitress import serve

        print("🌐 Running with Waitress")
        serve(app, host="0.0.0.0", port=5000, threads=4)
    except Exception:
        print("🌐 Running with Flask dev server")
        app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
