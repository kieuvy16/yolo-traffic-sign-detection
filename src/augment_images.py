"""
Data augmentation cho các class ít bounding box trong database_goc.
Áp dụng: horizontal flip, rotation (±10°, ±15°), brightness adjustment.
Tự động xác định class cần augment dựa trên ngưỡng MIN_BBOX.
"""

import os
import cv2
import numpy as np
from collections import defaultdict
import random
import math

DATABASE_GOC_DIR = os.path.join(os.path.dirname(__file__), "database_goc")
IMAGES_DIR = os.path.join(DATABASE_GOC_DIR, "images")
LABELS_DIR = os.path.join(DATABASE_GOC_DIR, "labels")

# Ngưỡng tối thiểu: class nào có ít hơn số bbox này sẽ được augment
MIN_BBOX = 200

# Mục tiêu: augment lên đến khoảng bao nhiêu bbox
TARGET_BBOX = 300

CLASSES = [
    "W.224", "W.205c", "P.102", "R.302a", "W.205a", "W.207", "W.201a",
    "P.123a", "I.434a", "R.303", "P.130", "I.409", "R.415a", "W.245a",
    "P.106a*Xe tải", "W.203c", "P.117*", "P.124a*", "P.107", "P.124d",
    "P.103a", "W.203b", "W.221b", "P.111", "P.129", "S.505a*Xe máy",
    "W.246a", "W.225", "S.505a*Xe tải và công", "P.104", "S.505a*Xe tải",
    "Camera", "P.123b", "W.202b", "B.8a", "P.137", "P.139", "W.205b",
    "P.127*50", "P.127*60", "P.127*80", "P.127*40", "R.301e", "W.239b*",
    "W.233", "I.407a", "P.131a", "P.124b1", "W.210", "P.124c", "W.201b",
    "W.246c",
]


# ===== AUGMENTATION FUNCTIONS =====

def horizontal_flip(img, bboxes):
    """Lật ngang ảnh và bbox. YOLO: class cx cy w h"""
    flipped = cv2.flip(img, 1)
    new_bboxes = []
    for bbox in bboxes:
        cls_id, cx, cy, w, h = bbox
        new_bboxes.append((cls_id, 1.0 - cx, cy, w, h))
    return flipped, new_bboxes


def adjust_brightness(img, bboxes, factor):
    """Điều chỉnh độ sáng. factor > 1 = sáng hơn, < 1 = tối hơn."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return result, bboxes  # bbox không đổi


def rotate_image_bbox(img, bboxes, angle_deg):
    """
    Xoay ảnh một góc nhỏ và recalculate bbox.
    Giữ nguyên kích thước ảnh, pad đen nếu cần.
    """
    h, w = img.shape[:2]
    center = (w / 2, h / 2)

    # Ma trận xoay
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    # Tính kích thước ảnh mới để chứa toàn bộ ảnh xoay
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)

    # Điều chỉnh translation trong ma trận xoay
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    rotated = cv2.warpAffine(img, M, (new_w, new_h), borderValue=(114, 114, 114))

    # Xoay bounding boxes
    new_bboxes = []
    for bbox in bboxes:
        cls_id, cx, cy, bw, bh = bbox

        # Convert YOLO normalized -> pixel coords (corners)
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h

        # 4 corners
        corners = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ], dtype=np.float32)

        # Xoay từng corner
        ones = np.ones((4, 1), dtype=np.float32)
        corners_h = np.hstack([corners, ones])
        rotated_corners = M.dot(corners_h.T).T  # shape (4, 2)

        # Tính axis-aligned bbox mới
        rx_min = rotated_corners[:, 0].min()
        rx_max = rotated_corners[:, 0].max()
        ry_min = rotated_corners[:, 1].min()
        ry_max = rotated_corners[:, 1].max()

        # Clamp vào giới hạn ảnh mới
        rx_min = max(0, rx_min)
        ry_min = max(0, ry_min)
        rx_max = min(new_w, rx_max)
        ry_max = min(new_h, ry_max)

        # Convert lại YOLO normalized
        ncx = ((rx_min + rx_max) / 2) / new_w
        ncy = ((ry_min + ry_max) / 2) / new_h
        nbw = (rx_max - rx_min) / new_w
        nbh = (ry_max - ry_min) / new_h

        # Bỏ qua bbox quá nhỏ hoặc ngoài ảnh
        if nbw > 0.01 and nbh > 0.01 and nbw < 1.0 and nbh < 1.0:
            new_bboxes.append((cls_id, ncx, ncy, nbw, nbh))

    return rotated, new_bboxes


# Danh sách augmentation sẽ áp dụng
AUGMENTATIONS = [
    ("flip", lambda img, bb: horizontal_flip(img, bb)),
    ("bright_up", lambda img, bb: adjust_brightness(img, bb, 1.4)),
    ("bright_down", lambda img, bb: adjust_brightness(img, bb, 0.6)),
    ("rot_p10", lambda img, bb: rotate_image_bbox(img, bb, 10)),
    ("rot_n10", lambda img, bb: rotate_image_bbox(img, bb, -10)),
    ("rot_p15", lambda img, bb: rotate_image_bbox(img, bb, 15)),
    ("rot_n15", lambda img, bb: rotate_image_bbox(img, bb, -15)),
    ("flip_bright", lambda img, bb: adjust_brightness(*horizontal_flip(img, bb), 1.3)),
]


def parse_label_file(path):
    """Đọc file label YOLO -> list of (class_id, cx, cy, w, h)"""
    bboxes = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                bboxes.append((cls_id, cx, cy, w, h))
    return bboxes


def write_label_file(path, bboxes):
    """Ghi file label YOLO"""
    with open(path, 'w') as f:
        for bbox in bboxes:
            cls_id, cx, cy, w, h = bbox
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def find_image_for_label(label_file):
    """Tìm ảnh tương ứng với label file"""
    base = os.path.splitext(label_file)[0]
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        img_path = os.path.join(IMAGES_DIR, base + ext)
        if os.path.exists(img_path):
            return img_path
    return None


def main():
    # === 1. Đếm bbox per class ===
    print("=== Đếm bounding box theo class ===")
    label_files = [f for f in os.listdir(LABELS_DIR) if f.endswith('.txt')]

    bbox_per_class = defaultdict(int)
    images_per_class = defaultdict(list)  # class_id -> list of label filenames

    for lf in label_files:
        bboxes = parse_label_file(os.path.join(LABELS_DIR, lf))
        classes_in_file = set()
        for bbox in bboxes:
            cls_id = bbox[0]
            bbox_per_class[cls_id] += 1
            classes_in_file.add(cls_id)
        for cls_id in classes_in_file:
            images_per_class[cls_id].append(lf)

    # === 2. Xác định class cần augment ===
    classes_to_augment = []
    for cls_id in range(len(CLASSES)):
        count = bbox_per_class.get(cls_id, 0)
        if count < MIN_BBOX:
            classes_to_augment.append((cls_id, count))

    print(f"\nNgưỡng MIN_BBOX = {MIN_BBOX}, TARGET_BBOX = {TARGET_BBOX}")
    print(f"Số class cần augment: {len(classes_to_augment)}")
    print(f"{'ID':<5} {'Class':<28} {'Hiện tại':<10} {'Cần thêm ~':<10}")
    print("-" * 55)
    for cls_id, count in classes_to_augment:
        need = max(0, TARGET_BBOX - count)
        print(f"{cls_id:<5} {CLASSES[cls_id]:<28} {count:<10} {need:<10}")

    # === 3. Augment ===
    print(f"\n{'='*60}")
    print(">>> BẮT ĐẦU AUGMENTATION <<<")
    print(f"{'='*60}")

    total_created = 0
    stats = defaultdict(int)

    for cls_id, current_count in classes_to_augment:
        target = TARGET_BBOX
        need = target - current_count
        if need <= 0:
            continue

        source_labels = images_per_class.get(cls_id, [])
        if not source_labels:
            print(f"  [SKIP] Class {cls_id} ({CLASSES[cls_id]}): không có ảnh nguồn")
            continue

        print(f"\n  Class [{cls_id}] {CLASSES[cls_id]}: {current_count} -> mục tiêu ~{target} ({need} cần thêm)")

        created = 0
        aug_idx = 0

        # Lặp qua ảnh nguồn, áp dụng augmentations cho đến khi đủ
        while created < need:
            for label_file in source_labels:
                if created >= need:
                    break

                img_path = find_image_for_label(label_file)
                if img_path is None:
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    continue

                bboxes = parse_label_file(os.path.join(LABELS_DIR, label_file))

                # Chọn augmentation
                aug_name, aug_fn = AUGMENTATIONS[aug_idx % len(AUGMENTATIONS)]
                aug_idx += 1

                try:
                    aug_img, aug_bboxes = aug_fn(img, bboxes)
                except Exception:
                    continue

                if not aug_bboxes:
                    continue

                # Tạo tên file mới
                base_name = os.path.splitext(label_file)[0]
                img_ext = os.path.splitext(img_path)[1]
                new_name = f"aug_{cls_id}_{aug_name}_{created}_{base_name}"

                # Lưu ảnh
                new_img_path = os.path.join(IMAGES_DIR, new_name + img_ext)
                cv2.imwrite(new_img_path, aug_img)

                # Lưu label
                new_lbl_path = os.path.join(LABELS_DIR, new_name + '.txt')
                write_label_file(new_lbl_path, aug_bboxes)

                created += 1

            # Đề phòng vòng lặp vô hạn nếu source_labels ít quá
            if aug_idx > need * 3:
                break

        print(f"    -> Tạo {created} ảnh augmented")
        total_created += created
        stats[cls_id] = created

    # === 4. Kết quả ===
    print(f"\n{'='*60}")
    print(f"=== KẾT QUẢ AUGMENTATION ===")
    print(f"{'='*60}")
    print(f"  Tổng ảnh augmented tạo mới: {total_created}")

    # Đếm lại tổng
    new_images = len([f for f in os.listdir(IMAGES_DIR)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    new_labels = len([f for f in os.listdir(LABELS_DIR) if f.endswith('.txt')])
    print(f"  Database_goc tổng: {new_images} images, {new_labels} labels")

    # Đếm lại bbox per class
    print(f"\n{'ID':<5} {'Class':<28} {'Trước':<10} {'Thêm':<10} {'Sau':<10}")
    print("-" * 65)
    for cls_id, old_count in classes_to_augment:
        added = stats.get(cls_id, 0)
        # Đếm lại thực tế
        new_count = 0
        for lf in os.listdir(LABELS_DIR):
            if not lf.endswith('.txt'):
                continue
            for bbox in parse_label_file(os.path.join(LABELS_DIR, lf)):
                if bbox[0] == cls_id:
                    new_count += 1
                    break  # chỉ đếm ảnh, không đếm bbox cho nhanh
        print(f"{cls_id:<5} {CLASSES[cls_id]:<28} {old_count:<10} +{added:<9} ~{old_count + added * 2:<8}")


if __name__ == "__main__":
    main()
