"""
Resize tất cả ảnh trong database_goc về 640x640.
YOLO normalized coordinates (0-1) không cần thay đổi khi resize.
"""
import os
import sys
import cv2

sys.stdout.reconfigure(line_buffering=True)

IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database_goc", "images")
TARGET_SIZE = 640


def main():
    files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    print(f"Total images: {len(files)}", flush=True)

    resized = 0
    already_ok = 0
    errors = 0

    for i, f in enumerate(files):
        path = os.path.join(IMAGES_DIR, f)
        img = cv2.imread(path)
        if img is None:
            errors += 1
            continue

        h, w = img.shape[:2]
        if w == TARGET_SIZE and h == TARGET_SIZE:
            already_ok += 1
            continue

        # Resize giữ tỷ lệ với letterbox (padding)
        scale = min(TARGET_SIZE / w, TARGET_SIZE / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Tạo canvas 640x640 với padding màu xám (114)
        canvas = cv2.copyMakeBorder(
            resized_img,
            top=(TARGET_SIZE - new_h) // 2,
            bottom=TARGET_SIZE - new_h - (TARGET_SIZE - new_h) // 2,
            left=(TARGET_SIZE - new_w) // 2,
            right=TARGET_SIZE - new_w - (TARGET_SIZE - new_w) // 2,
            borderType=cv2.BORDER_CONSTANT,
            value=(114, 114, 114)
        )

        # Ghi đè ảnh gốc
        cv2.imwrite(path, canvas)
        resized += 1

        if (i + 1) % 1000 == 0:
            print(f"  Progress: {i+1}/{len(files)} ({resized} resized)", flush=True)

        # Cập nhật label (bounding box) cho letterbox
        label_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "database_goc", "labels",
            os.path.splitext(f)[0] + ".txt"
        )
        if os.path.exists(label_path):
            # Đọc labels gốc
            lines = []
            with open(label_path, 'r') as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = parts[0]
                        cx = float(parts[1])
                        cy = float(parts[2])
                        bw = float(parts[3])
                        bh = float(parts[4])

                        # Chuyển từ tọa độ ảnh gốc sang tọa độ letterbox
                        pad_x = (TARGET_SIZE - new_w) / 2
                        pad_y = (TARGET_SIZE - new_h) / 2

                        # cx, cy, bw, bh đang normalized theo ảnh gốc (w x h)
                        # Chuyển sang pixel coords trên ảnh gốc
                        px = cx * w
                        py = cy * h
                        pw = bw * w
                        ph = bh * h

                        # Scale + pad
                        px_new = px * scale + pad_x
                        py_new = py * scale + pad_y
                        pw_new = pw * scale
                        ph_new = ph * scale

                        # Normalize lại theo 640x640
                        ncx = px_new / TARGET_SIZE
                        ncy = py_new / TARGET_SIZE
                        nbw = pw_new / TARGET_SIZE
                        nbh = ph_new / TARGET_SIZE

                        lines.append(f"{cls_id} {ncx:.6f} {ncy:.6f} {nbw:.6f} {nbh:.6f}")

            with open(label_path, 'w') as lf:
                lf.write('\n'.join(lines) + '\n')

    print(f"\n{'='*50}", flush=True)
    print(f"Done!", flush=True)
    print(f"  Already 640x640: {already_ok}", flush=True)
    print(f"  Resized: {resized}", flush=True)
    print(f"  Errors: {errors}", flush=True)


if __name__ == "__main__":
    main()
