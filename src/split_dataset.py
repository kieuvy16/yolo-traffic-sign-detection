"""
Chia dataset database_goc thành train/val/test (70/20/10)
và tạo file data.yaml cho YOLO training.
"""
import os
import sys
import random
import shutil
import yaml

sys.stdout.reconfigure(line_buffering=True)
random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_GOC_DIR = os.path.join(BASE_DIR, "database_goc")
IMAGES_DIR = os.path.join(DATABASE_GOC_DIR, "images")
LABELS_DIR = os.path.join(DATABASE_GOC_DIR, "labels")

# Output directories
SPLIT_DIR = os.path.join(DATABASE_GOC_DIR, "split_dataset")
TRAIN_IMAGES = os.path.join(SPLIT_DIR, "train", "images")
TRAIN_LABELS = os.path.join(SPLIT_DIR, "train", "labels")
VAL_IMAGES = os.path.join(SPLIT_DIR, "valid", "images")
VAL_LABELS = os.path.join(SPLIT_DIR, "valid", "labels")
TEST_IMAGES = os.path.join(SPLIT_DIR, "test", "images")
TEST_LABELS = os.path.join(SPLIT_DIR, "test", "labels")

# Tỷ lệ chia
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# 52 class names
CLASSES = [
    "W.224", "W.205c", "P.102", "R.302a", "W.205a", "W.207", "W.201a",
    "P.123a", "I.434a", "R.303", "P.130", "I.409", "R.415a", "W.245a",
    "P.106a*Xe tai", "W.203c", "P.117*", "P.124a*", "P.107", "P.124d",
    "P.103a", "W.203b", "W.221b", "P.111", "P.129", "S.505a*Xe may",
    "W.246a", "W.225", "S.505a*Xe tai va cong", "P.104", "S.505a*Xe tai",
    "Camera", "P.123b", "W.202b", "B.8a", "P.137", "P.139", "W.205b",
    "P.127*50", "P.127*60", "P.127*80", "P.127*40", "R.301e", "W.239b*",
    "W.233", "I.407a", "P.131a", "P.124b1", "W.210", "P.124c", "W.201b",
    "W.246c",
]

def main():
    # Tạo các thư mục
    for d in [TRAIN_IMAGES, TRAIN_LABELS, VAL_IMAGES, VAL_LABELS, TEST_IMAGES, TEST_LABELS]:
        os.makedirs(d, exist_ok=True)
        # Xóa file cũ nếu có
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))

    # Lấy danh sách ảnh có label tương ứng
    all_images = []
    for f in os.listdir(IMAGES_DIR):
        if not f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        base = os.path.splitext(f)[0]
        label_path = os.path.join(LABELS_DIR, base + '.txt')
        if os.path.exists(label_path):
            all_images.append(f)

    print(f"Total paired images+labels: {len(all_images)}", flush=True)

    # Shuffle
    random.shuffle(all_images)

    # Chia
    n = len(all_images)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_set = all_images[:n_train]
    val_set = all_images[n_train:n_train + n_val]
    test_set = all_images[n_train + n_val:]

    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}", flush=True)

    # Copy files
    def copy_set(file_list, dst_images, dst_labels, name):
        for i, f in enumerate(file_list):
            base = os.path.splitext(f)[0]
            ext = os.path.splitext(f)[1]

            src_img = os.path.join(IMAGES_DIR, f)
            src_lbl = os.path.join(LABELS_DIR, base + '.txt')

            shutil.copy2(src_img, os.path.join(dst_images, f))
            shutil.copy2(src_lbl, os.path.join(dst_labels, base + '.txt'))

            if (i + 1) % 2000 == 0:
                print(f"  {name}: {i+1}/{len(file_list)}", flush=True)

        print(f"  {name}: {len(file_list)} files copied ✅", flush=True)

    print("\nCopying train...", flush=True)
    copy_set(train_set, TRAIN_IMAGES, TRAIN_LABELS, "train")

    print("Copying val...", flush=True)
    copy_set(val_set, VAL_IMAGES, VAL_LABELS, "val")

    print("Copying test...", flush=True)
    copy_set(test_set, TEST_IMAGES, TEST_LABELS, "test")

    # Tạo data.yaml
    data_yaml = {
        'path': SPLIT_DIR.replace('\\', '/'),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(CLASSES),
        'names': CLASSES,
    }

    yaml_path = os.path.join(SPLIT_DIR, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"\n{'='*50}", flush=True)
    print(f"✅ data.yaml created: {yaml_path}", flush=True)
    print(f"{'='*50}", flush=True)

    # Verify
    for split_name, img_dir, lbl_dir in [
        ("train", TRAIN_IMAGES, TRAIN_LABELS),
        ("valid", VAL_IMAGES, VAL_LABELS),
        ("test", TEST_IMAGES, TEST_LABELS),
    ]:
        ni = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))])
        nl = len([f for f in os.listdir(lbl_dir) if f.endswith('.txt')])
        print(f"  {split_name}: {ni} images, {nl} labels {'✅' if ni==nl else '❌'}", flush=True)


if __name__ == "__main__":
    main()
