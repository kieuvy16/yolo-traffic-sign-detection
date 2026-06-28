"""
Script bổ sung ảnh từ datasets (Roboflow 38 class) vào database_goc (52 class).
- Remap label index từ datasets sang database_goc
- Chỉ copy ảnh có ít nhất 1 annotation hợp lệ sau remap
- Tránh trùng tên file bằng cách thêm prefix "rb_"
"""

import os
import shutil
from collections import defaultdict

# === CẤU HÌNH ===
DATABASE_GOC_DIR = os.path.join(os.path.dirname(__file__), "database_goc")
DATASETS_DIR = os.path.join(os.path.dirname(__file__), "datasets")

# 52 classes trong database_goc (theo thứ tự trong classes.txt)
DATABASE_GOC_CLASSES = [
    "W.224",        # 0  - Pedestrian Crossing
    "W.205c",       # 1  - Equal-level Intersection
    "P.102",        # 2  - No Entry
    "R.302a",       # 3  - Right Turn Only
    "W.205a",       # 4  - Intersection
    "W.207",        # 5  - Intersection with a non-priority road
    "W.201a",       # 6  - Danger zone on the left
    "P.123a",       # 7  - No Left Turn
    "I.434a",       # 8  - Bus Stop
    "R.303",        # 9  - Roundabout
    "P.130",        # 10 - No Stopping and No Parking
    "I.409",        # 11 - U-Turn Allowed
    "R.415a",       # 12 - Lane Allocation
    "W.245a",       # 13 - Slow Down
    "P.106a*Xe tải",# 14 - No Trucks Allowed
    "W.203c",       # 15 - Narrow Road on the Right
    "P.117*",       # 16 - Height Limit
    "P.124a*",      # 17 - No U-Turn
    "P.107",        # 18 - No Passenger Cars and Trucks
    "P.124d",       # 19 - No U-Turn and No Right Turn
    "P.103a",       # 20 - No Cars Allowed
    "W.203b",       # 21 - Narrow Road on the Left
    "W.221b",       # 22 - Uneven Road
    "P.111",        # 23 - No Two or Three-wheeled Vehicles
    "P.129",        # 24 - Customs Checkpoint
    "S.505a*Xe máy",# 25 - Motorcycles Only
    "W.246a",       # 26 - Obstacle on the Road
    "W.225",        # 27 - Children Present
    "S.505a*Xe tải và công", # 28 - Trucks and Containers
    "P.104",        # 29 - No Motorcycles Allowed
    "S.505a*Xe tải",# 30 - Trucks Only
    "Camera",       # 31 - Road with Surveillance Camera
    "P.123b",       # 32 - No Right Turn
    "W.202b",       # 33 - Double curve first to right
    "B.8a",         # 34 - No Containers Allowed
    "P.137",        # 35 - No Left or Right Turn
    "P.139",        # 36 - No Straight and Right Turn
    "W.205b",       # 37 - Intersection with T-Junction
    "P.127*50",     # 38 - Speed limit (50km/h)
    "P.127*60",     # 39 - Speed limit (60km/h)
    "P.127*80",     # 40 - Speed limit (80km/h)
    "P.127*40",     # 41 - Speed limit (40km/h)
    "R.301e",       # 42 - Left Turn
    "W.239b*",      # 43 - Low Clearance
    "W.233",        # 44 - Other Danger
    "I.407a",       # 45 - One-way street
    "P.131a",       # 46 - No Parking
    "P.124b1",      # 47 - No U-Turn for Cars
    "W.210",        # 48 - Level Crossing with Barriers
    "P.124c",       # 49 - No U-Turn and No Left Turn
    "W.201b",       # 50 - Danger zone on the right
    "W.246c",       # 51 - Warning: Obstacle ahead
]

# 38 classes trong datasets (Roboflow) theo thứ tự trong data.yaml
DATASETS_CLASSES = [
    '0verhead electrical cables',          # 0
    'Bicycle ban',                         # 1
    'Bus Stop',                            # 2
    'Cars ban',                            # 3
    'Compulsary ahead',                    # 4
    'Compulsory keep left',                # 5
    'Compulsory keep right',               # 6
    'Containers ban',                      # 7
    'Dangerous Turn',                      # 8
    'Left Turn',                           # 9
    'Motobike ban',                        # 10
    'Motobike ban1',                       # 11
    'Motorcycles Only',                    # 12
    'No Passenger Cars and Trucks',        # 13
    'No Two or Three-wheeled Vehicles',    # 14
    'No U-Turn and No turn right',         # 15
    'No U-turn',                           # 16
    'No U-turn No turn left',              # 17
    'No car turn left',                    # 18
    'No car turn right',                   # 19
    'No parking',                          # 20
    'No parking stopping',                 # 21
    'No turn left',                        # 22
    'No turn right',                       # 23
    'One way',                             # 24
    'Packing',                             # 25
    'Pedestrian crossing sign',            # 26
    'Pedestrians prohibited',              # 27
    'Priority sign',                       # 28
    'Prohibiting pedestrians',             # 29
    'Slowly',                              # 30
    'Speed -limit 40',                     # 31
    'Speed -limit 50',                     # 32
    'Speed -limit 60',                     # 33
    'Speed -limit 80',                     # 34
    'U-Turn Allowed',                      # 35
    'Watch for children',                  # 36
    'Yield sign',                          # 37
]

# === MAPPING: datasets class index -> database_goc class index ===
# None = không có class tương ứng -> bỏ qua
REMAP = {
    0:  None,       # 0verhead electrical cables -> không có
    1:  None,       # Bicycle ban -> không có chính xác
    2:  8,          # Bus Stop -> I.434a (idx 8)
    3:  20,         # Cars ban -> P.103a (idx 20)
    4:  None,       # Compulsary ahead -> không có
    5:  42,         # Compulsory keep left -> R.301e (idx 42)
    6:  3,          # Compulsory keep right -> R.302a (idx 3)
    7:  34,         # Containers ban -> B.8a (idx 34)
    8:  6,          # Dangerous Turn -> W.201a (idx 6)
    9:  42,         # Left Turn -> R.301e (idx 42)
    10: 29,         # Motobike ban -> P.104 (idx 29)
    11: 29,         # Motobike ban1 -> P.104 (idx 29) (cùng loại)
    12: 25,         # Motorcycles Only -> S.505a*Xe máy (idx 25)
    13: 18,         # No Passenger Cars and Trucks -> P.107 (idx 18)
    14: 23,         # No Two or Three-wheeled Vehicles -> P.111 (idx 23)
    15: 19,         # No U-Turn and No turn right -> P.124d (idx 19)
    16: 17,         # No U-turn -> P.124a* (idx 17)
    17: 49,         # No U-turn No turn left -> P.124c (idx 49)
    18: 7,          # No car turn left -> P.123a (idx 7)
    19: 32,         # No car turn right -> P.123b (idx 32)
    20: 46,         # No parking -> P.131a (idx 46)
    21: 10,         # No parking stopping -> P.130 (idx 10)
    22: 7,          # No turn left -> P.123a (idx 7)
    23: 32,         # No turn right -> P.123b (idx 32)
    24: 45,         # One way -> I.407a (idx 45)
    25: None,       # Packing -> không rõ mapping
    26: 0,          # Pedestrian crossing sign -> W.224 (idx 0)
    27: None,       # Pedestrians prohibited -> không có
    28: None,       # Priority sign -> không có chính xác
    29: None,       # Prohibiting pedestrians -> không có
    30: 13,         # Slowly -> W.245a (idx 13)
    31: 41,         # Speed -limit 40 -> P.127*40 (idx 41)
    32: 38,         # Speed -limit 50 -> P.127*50 (idx 38)
    33: 39,         # Speed -limit 60 -> P.127*60 (idx 39)
    34: 40,         # Speed -limit 80 -> P.127*80 (idx 40)
    35: 11,         # U-Turn Allowed -> I.409 (idx 11)
    36: 27,         # Watch for children -> W.225 (idx 27)
    37: None,       # Yield sign -> không có
}


def remap_label_file(src_label_path):
    """
    Đọc file label YOLO, remap class index.
    Trả về list các dòng annotation đã remap (chỉ giữ dòng hợp lệ).
    """
    remapped_lines = []
    with open(src_label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            old_class_id = int(parts[0])
            new_class_id = REMAP.get(old_class_id, None)
            if new_class_id is not None:
                parts[0] = str(new_class_id)
                remapped_lines.append(' '.join(parts))
    return remapped_lines


def process_dataset_folder(images_dir, labels_dir, dst_images_dir, dst_labels_dir, prefix, stats):
    """
    Xử lý 1 folder (train hoặc valid) của datasets.
    """
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"  [SKIP] Không tìm thấy: {images_dir} hoặc {labels_dir}")
        return

    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    print(f"  Tìm thấy {len(label_files)} file label trong {labels_dir}")

    copied = 0
    skipped = 0

    for label_file in label_files:
        src_label = os.path.join(labels_dir, label_file)
        remapped = remap_label_file(src_label)

        if not remapped:
            skipped += 1
            continue

        # Tìm ảnh tương ứng
        base_name = os.path.splitext(label_file)[0]
        img_file = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            candidate = os.path.join(images_dir, base_name + ext)
            if os.path.exists(candidate):
                img_file = candidate
                break

        if img_file is None:
            skipped += 1
            continue

        # Tạo tên file mới với prefix để tránh trùng
        img_ext = os.path.splitext(img_file)[1]
        new_name = prefix + base_name
        dst_img = os.path.join(dst_images_dir, new_name + img_ext)
        dst_lbl = os.path.join(dst_labels_dir, new_name + '.txt')

        # Copy ảnh
        shutil.copy2(img_file, dst_img)

        # Ghi label đã remap
        with open(dst_lbl, 'w') as f:
            f.write('\n'.join(remapped) + '\n')

        # Thống kê theo class
        for line in remapped:
            cls_id = int(line.split()[0])
            stats[cls_id] += 1

        copied += 1

    print(f"  -> Đã copy: {copied} ảnh, bỏ qua: {skipped} ảnh")
    return copied


def main():
    dst_images_dir = os.path.join(DATABASE_GOC_DIR, "images")
    dst_labels_dir = os.path.join(DATABASE_GOC_DIR, "labels")

    # Đếm ảnh hiện tại
    existing_images = len([f for f in os.listdir(dst_images_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    existing_labels = len([f for f in os.listdir(dst_labels_dir)
                          if f.endswith('.txt')])
    print(f"=== database_goc hiện tại ===")
    print(f"  Images: {existing_images}")
    print(f"  Labels: {existing_labels}")
    print()

    stats = defaultdict(int)
    total_copied = 0

    # Xử lý train set
    print(">>> Xử lý datasets/train ...")
    train_imgs = os.path.join(DATASETS_DIR, "train", "images")
    train_lbls = os.path.join(DATASETS_DIR, "train", "labels")
    n = process_dataset_folder(train_imgs, train_lbls, dst_images_dir, dst_labels_dir, "rb_train_", stats)
    if n:
        total_copied += n

    # Xử lý valid set
    print("\n>>> Xử lý datasets/valid ...")
    valid_imgs = os.path.join(DATASETS_DIR, "valid", "images")
    valid_lbls = os.path.join(DATASETS_DIR, "valid", "labels")
    n = process_dataset_folder(valid_imgs, valid_lbls, dst_images_dir, dst_labels_dir, "rb_valid_", stats)
    if n:
        total_copied += n

    # Đếm lại sau bổ sung
    new_images = len([f for f in os.listdir(dst_images_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    new_labels = len([f for f in os.listdir(dst_labels_dir)
                     if f.endswith('.txt')])

    print(f"\n{'='*60}")
    print(f"=== KẾT QUẢ BỔ SUNG ===")
    print(f"{'='*60}")
    print(f"  Ảnh trước: {existing_images} -> sau: {new_images} (+{new_images - existing_images})")
    print(f"  Labels trước: {existing_labels} -> sau: {new_labels} (+{new_labels - existing_labels})")
    print(f"  Tổng ảnh copy từ datasets: {total_copied}")

    print(f"\n{'='*60}")
    print(f"=== THỐNG KÊ THEO CLASS (annotations bổ sung) ===")
    print(f"{'='*60}")
    print(f"{'Class ID':<10} {'Tên class':<30} {'Số annotations':<15}")
    print(f"{'-'*55}")
    for cls_id in sorted(stats.keys()):
        cls_name = DATABASE_GOC_CLASSES[cls_id] if cls_id < len(DATABASE_GOC_CLASSES) else "???"
        print(f"{cls_id:<10} {cls_name:<30} {stats[cls_id]:<15}")

    # Kiểm tra: không có label nào ngoài range
    all_labels_ok = all(0 <= k <= 51 for k in stats.keys())
    print(f"\n✅ Tất cả label index nằm trong [0, 51]: {all_labels_ok}")


if __name__ == "__main__":
    main()
