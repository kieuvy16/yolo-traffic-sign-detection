import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from PIL import Image, ImageDraw, ImageFont

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def load_lines(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8-sig") as f:
        return [line.strip() for line in f if line.strip()]


def sanitize_folder_name(name: str) -> str:
    safe = []
    for ch in name:
        if ch.isalnum() or ch in ["-", "_", "."]:
            safe.append(ch)
        else:
            safe.append("_")
    cleaned = "".join(safe).strip("_")
    return cleaned or "unknown"


def find_image_for_label(labels_dir: Path, images_dir: Path, label_file: Path) -> Optional[Path]:
    stem = label_file.stem
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    # Fallback: case-insensitive scan if extension does not match expected list.
    for image_path in images_dir.glob(f"{stem}.*"):
        if image_path.is_file():
            return image_path

    return None


def parse_label_classes(label_file: Path) -> Set[int]:
    class_ids: Set[int] = set()
    with label_file.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if not parts:
                continue
            try:
                class_id = int(float(parts[0]))
            except ValueError:
                raise ValueError(f"Invalid class id in {label_file} at line {line_no}: {parts[0]}")
            class_ids.add(class_id)
    return class_ids


def build_class_mapping(class_codes_file: Path, class_vi_file: Path) -> Dict[int, Tuple[str, str]]:
    class_codes = load_lines(class_codes_file)
    class_vi = load_lines(class_vi_file)

    if len(class_codes) != len(class_vi):
        raise ValueError(
            f"classes.txt count ({len(class_codes)}) and classes_vie.txt count ({len(class_vi)}) do not match"
        )

    mapping: Dict[int, Tuple[str, str]] = {}
    for idx, (code, vi_name) in enumerate(zip(class_codes, class_vi)):
        mapping[idx] = (code, vi_name)
    return mapping


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def color_for_class(class_id: int) -> Tuple[int, int, int]:
    # Deterministic, high-contrast-ish palette from class id.
    return (
        (class_id * 53 + 80) % 256,
        (class_id * 97 + 120) % 256,
        (class_id * 193 + 40) % 256,
    )


def load_font(font_size: int) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    for font_path in candidates:
        try:
            return ImageFont.truetype(font_path, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_labels_on_image(
    source_image: Path,
    label_file: Path,
    output_image: Path,
    class_mapping: Dict[int, Tuple[str, str]],
    target_class_id: Optional[int] = None,
) -> None:
    with Image.open(source_image) as img:
        image = img.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    font_size = max(14, min(width, height) // 30)
    font = load_font(font_size)

    with label_file.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            if len(parts) < 5:
                continue

            try:
                class_id = int(float(parts[0]))
                x_center = float(parts[1])
                y_center = float(parts[2])
                box_w = float(parts[3])
                box_h = float(parts[4])
            except ValueError:
                print(f"[WARN] Invalid label format in {label_file} at line {line_no}")
                continue

            if target_class_id is not None and class_id != target_class_id:
                continue

            x1 = int((x_center - box_w / 2.0) * width)
            y1 = int((y_center - box_h / 2.0) * height)
            x2 = int((x_center + box_w / 2.0) * width)
            y2 = int((y_center + box_h / 2.0) * height)

            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width - 1, x2))
            y2 = max(0, min(height - 1, y2))

            color = color_for_class(class_id)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            class_code, class_name_vi = class_mapping.get(class_id, (f"unknown_{class_id}", "Unknown"))
            label_text = f"{class_code} | {class_name_vi}"

            try:
                text_bbox = draw.textbbox((0, 0), label_text, font=font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
            except Exception:
                text_w = int(len(label_text) * font_size * 0.6)
                text_h = font_size + 4

            text_x = x1
            text_y = y1 - text_h - 6
            if text_y < 0:
                text_y = y1 + 4

            bg_pad = 3
            draw.rectangle(
                [
                    text_x - bg_pad,
                    text_y - bg_pad,
                    text_x + text_w + bg_pad,
                    text_y + text_h + bg_pad,
                ],
                fill=color,
            )
            draw.text((text_x, text_y), label_text, fill=(255, 255, 255), font=font)

    image.save(output_image)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export images grouped by YOLO class for quick Vietnamese label review"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("src/database_goc/split_dataset"),
        help="YOLO dataset root containing train/valid/test folders",
    )
    parser.add_argument(
        "--classes-file",
        type=Path,
        default=Path("src/database_goc/classes.txt"),
        help="Path to classes.txt",
    )
    parser.add_argument(
        "--classes-vie-file",
        type=Path,
        default=Path("src/database_goc/classes_vie.txt"),
        help="Path to classes_vie.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/database_goc/review_by_label"),
        help="Output directory for grouped images",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        choices=["train", "valid", "test"],
        help="Dataset splits to scan",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Maximum images copied per class (0 means no limit)",
    )
    args = parser.parse_args()

    class_mapping = build_class_mapping(args.classes_file, args.classes_vie_file)

    ensure_dir(args.output_dir)
    report_csv = args.output_dir / "review_manifest.csv"

    copied_counts: Dict[int, int] = defaultdict(int)
    missing_images: List[Path] = []
    total_label_files = 0

    with report_csv.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "class_id",
                "class_code",
                "class_name_vi",
                "split",
                "label_file",
                "source_image",
                "copied_image",
            ]
        )

        for split in args.splits:
            split_root = args.dataset_root / split
            labels_dir = split_root / "labels"
            images_dir = split_root / "images"

            if not labels_dir.exists() or not images_dir.exists():
                print(f"[WARN] Skip split '{split}' because labels/images folder is missing")
                continue

            label_files = sorted(labels_dir.glob("*.txt"))
            total_label_files += len(label_files)

            for label_file in label_files:
                try:
                    class_ids = parse_label_classes(label_file)
                except ValueError as e:
                    print(f"[WARN] {e}")
                    continue

                if not class_ids:
                    continue

                image_file = find_image_for_label(labels_dir, images_dir, label_file)
                if image_file is None:
                    missing_images.append(label_file)
                    continue

                for class_id in sorted(class_ids):
                    if class_id not in class_mapping:
                        print(f"[WARN] Unknown class id {class_id} in {label_file}")
                        continue

                    if args.max_per_class > 0 and copied_counts[class_id] >= args.max_per_class:
                        continue

                    class_code, class_name_vi = class_mapping[class_id]
                    class_folder_name = f"{class_id:02d}_{sanitize_folder_name(class_code)}"
                    class_folder = args.output_dir / class_folder_name
                    ensure_dir(class_folder)

                    out_name = f"{split}__{image_file.name}"
                    copied_image = class_folder / out_name

                    if not copied_image.exists():
                        draw_labels_on_image(
                            source_image=image_file,
                            label_file=label_file,
                            output_image=copied_image,
                            class_mapping=class_mapping,
                            target_class_id=class_id,
                        )
                        copied_counts[class_id] += 1

                    writer.writerow(
                        [
                            class_id,
                            class_code,
                            class_name_vi,
                            split,
                            str(label_file),
                            str(image_file),
                            str(copied_image),
                        ]
                    )

    # Write quick per-class summary
    summary_txt = args.output_dir / "summary.txt"
    with summary_txt.open("w", encoding="utf-8") as f:
        f.write("Per-class copied image counts\n")
        f.write("=" * 40 + "\n")
        for class_id in sorted(class_mapping.keys()):
            class_code, class_name_vi = class_mapping[class_id]
            f.write(
                f"{class_id:02d} | {class_code:<20} | {copied_counts[class_id]:>5} images | {class_name_vi}\n"
            )

        f.write("\n")
        f.write(f"Total label files scanned: {total_label_files}\n")
        f.write(f"Missing images for label files: {len(missing_images)}\n")
        if missing_images:
            f.write("Missing list:\n")
            for p in missing_images:
                f.write(f"- {p}\n")

    print("Done.")
    print(f"Output dir: {args.output_dir}")
    print(f"Manifest: {report_csv}")
    print(f"Summary: {summary_txt}")


if __name__ == "__main__":
    main()
