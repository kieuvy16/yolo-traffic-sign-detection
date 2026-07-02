"""
Phân tích số lượng bounding box theo từng class trong database_goc.
"""

import os
from collections import defaultdict

DATABASE_GOC_DIR = os.path.join(os.path.dirname(__file__), "database_goc")

# 52 classes
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

CLASSES_EN = [
    "Pedestrian Crossing", "Equal-level Intersection", "No Entry",
    "Right Turn Only", "Intersection", "Intersection non-priority",
    "Danger zone left", "No Left Turn", "Bus Stop", "Roundabout",
    "No Stopping/Parking", "U-Turn Allowed", "Lane Allocation", "Slow Down",
    "No Trucks", "Narrow Road Right", "Height Limit", "No U-Turn",
    "No Cars and Trucks", "No U-Turn+Right", "No Cars", "Narrow Road Left",
    "Uneven Road", "No 2-3 wheeled", "Customs Checkpoint", "Motorcycles Only",
    "Obstacle", "Children Present", "Trucks+Containers", "No Motorcycles",
    "Trucks Only", "Surveillance Camera", "No Right Turn",
    "Double curve right", "No Containers", "No Left/Right Turn",
    "No Straight+Right", "T-Junction", "Speed 50km/h", "Speed 60km/h",
    "Speed 80km/h", "Speed 40km/h", "Left Turn", "Low Clearance",
    "Other Danger", "One-way", "No Parking", "No U-Turn Cars",
    "Level Crossing", "No U-Turn+Left", "Danger zone right", "Obstacle right",
]


def main():
    labels_dir = os.path.join(DATABASE_GOC_DIR, "labels")
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

    bbox_count = defaultdict(int)    # class_id -> số bounding box
    image_count = defaultdict(int)   # class_id -> số ảnh chứa class đó
    total_bbox = 0
    total_files = len(label_files)

    for label_file in label_files:
        filepath = os.path.join(labels_dir, label_file)
        classes_in_file = set()
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    bbox_count[cls_id] += 1
                    classes_in_file.add(cls_id)
                    total_bbox += 1
        for cls_id in classes_in_file:
            image_count[cls_id] += 1

    # Header
    print(f"{'='*90}")
    print(f"  PHÂN TÍCH BOUNDING BOX THEO CLASS - database_goc")
    print(f"  Tổng file label: {total_files} | Tổng bounding box: {total_bbox}")
    print(f"{'='*90}")
    print(f"{'ID':<5} {'Mã biển':<25} {'Tên EN':<25} {'BBox':<10} {'Ảnh':<10} {'%BBox':<8}")
    print(f"{'-'*90}")

    for cls_id in range(len(CLASSES)):
        bb = bbox_count.get(cls_id, 0)
        img = image_count.get(cls_id, 0)
        pct = (bb / total_bbox * 100) if total_bbox > 0 else 0
        bar = '█' * int(pct * 2)  # visual bar
        print(f"{cls_id:<5} {CLASSES[cls_id]:<25} {CLASSES_EN[cls_id]:<25} {bb:<10} {img:<10} {pct:>5.1f}% {bar}")

    # Classes không có bbox
    empty = [i for i in range(len(CLASSES)) if bbox_count.get(i, 0) == 0]
    if empty:
        print(f"\n⚠️  {len(empty)} class KHÔNG có bounding box nào:")
        for i in empty:
            print(f"   [{i}] {CLASSES[i]} - {CLASSES_EN[i]}")

    # Top 10 nhiều nhất
    print(f"\n{'='*60}")
    print(f"  TOP 10 CLASS NHIỀU BOX NHẤT")
    print(f"{'='*60}")
    sorted_cls = sorted(bbox_count.items(), key=lambda x: x[1], reverse=True)
    for rank, (cls_id, count) in enumerate(sorted_cls[:10], 1):
        print(f"  {rank:>2}. [{cls_id}] {CLASSES[cls_id]:<25} {count:>6} bbox ({count/total_bbox*100:.1f}%)")

    # Bottom 10 ít nhất (chỉ class có > 0)
    print(f"\n{'='*60}")
    print(f"  TOP 10 CLASS ÍT BOX NHẤT")
    print(f"{'='*60}")
    sorted_asc = sorted([(k, v) for k, v in bbox_count.items() if v > 0], key=lambda x: x[1])
    for rank, (cls_id, count) in enumerate(sorted_asc[:10], 1):
        print(f"  {rank:>2}. [{cls_id}] {CLASSES[cls_id]:<25} {count:>6} bbox ({count/total_bbox*100:.1f}%)")


if __name__ == "__main__":
    main()
