#!/usr/bin/env python3
"""Train YOLO detection from the hand-labeled bounding boxes CSV.

This script does NOT use MedSAM-generated masks to define training labels.
It uses the original images plus the boxes from labels_my-project-name_2026-07-31-10-12-01.csv.

Outputs:
  after_yolo/yolo_dataset_from_labels/
    images/train, images/val
    labels/train, labels/val
    organ.yaml

Then it trains YOLO and saves the resulting best.pt under the run directory.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare YOLO data from CSV boxes and train YOLO.")
    parser.add_argument(
        "--labels",
        default="labels_my-project-name_2026-07-31-10-12-01.csv",
        help="CSV with bbox_x, bbox_y, bbox_width, bbox_height, image_name.",
    )
    parser.add_argument(
        "--images-dir",
        default="training_set/before images",
        help="Directory containing the original training images.",
    )
    parser.add_argument(
        "--dataset-dir",
        default="after_yolo/yolo_dataset_from_labels",
        help="Output YOLO dataset directory.",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=20,
        help="How many label preview images to render for visual QA.",
    )
    parser.add_argument(
        "--strict-single-box",
        action="store_true",
        help="Require exactly one CSV box per image.",
    )
    parser.add_argument(
        "--clear-dataset-dir",
        action="store_true",
        help="Delete and rebuild dataset-dir before conversion.",
    )
    parser.add_argument(
        "--project",
        default="after_yolo/runs",
        help="Ultralytics training project directory.",
    )
    parser.add_argument(
        "--name",
        default="detect_from_labels",
        help="Ultralytics run name.",
    )
    parser.add_argument(
        "--clear-run-dir",
        action="store_true",
        help="Delete existing project/name folder before training.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only build dataset/previews and skip YOLO training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Training batch size.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Split seed.",
    )
    parser.add_argument(
        "--weights",
        default="yolov8n.pt",
        help="Pretrained YOLO weights to fine-tune.",
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="Training device, e.g. mps, cuda:0, or cpu.",
    )
    return parser.parse_args()


def yolo_box_from_row(row: dict[str, str], image_width: int, image_height: int) -> tuple[float, float, float, float]:
    x = float(row["bbox_x"])
    y = float(row["bbox_y"])
    width = float(row["bbox_width"])
    height = float(row["bbox_height"])

    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(image_width), x + width)
    y2 = min(float(image_height), y + height)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid box for {row['image_name']}: {(x1, y1, x2, y2)}")

    cx = ((x1 + x2) / 2.0) / image_width
    cy = ((y1 + y2) / 2.0) / image_height
    bw = (x2 - x1) / image_width
    bh = (y2 - y1) / image_height

    values = {"cx": cx, "cy": cy, "bw": bw, "bh": bh}
    for key, value in values.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{row['image_name']}: {key}={value} is outside [0, 1]")
    if bw <= 0.0 or bh <= 0.0:
        raise ValueError(f"{row['image_name']}: non-positive box size")
    if bw > 0.95 or bh > 0.95:
        print(f"Warning: nearly full-image box for {row['image_name']}: bw={bw:.3f}, bh={bh:.3f}")

    return cx, cy, bw, bh


def copy_normalized_image(source: Path, destination: Path) -> tuple[int, int]:
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.suffix.lower() in {".jpg", ".jpeg"}:
            image.save(destination, quality=95)
        else:
            image.save(destination)
    return width, height


def save_label_preview(image_path: Path, rows: list[dict[str, str]], output_path: Path) -> None:
    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")

    draw = ImageDraw.Draw(image)
    for row in rows:
        x = float(row["bbox_x"])
        y = float(row["bbox_y"])
        box_width = float(row["bbox_width"])
        box_height = float(row["bbox_height"])
        x1 = x
        y1 = y
        x2 = x + box_width
        y2 = y + box_height
        draw.rectangle([x1, y1, x2, y2], outline="red", width=5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    args = parse_args()

    root = Path(__file__).resolve().parent.parent
    labels_path = (root / args.labels).resolve() if not Path(args.labels).is_absolute() else Path(args.labels)
    images_dir = (root / args.images_dir).resolve() if not Path(args.images_dir).is_absolute() else Path(args.images_dir)
    dataset_dir = (root / args.dataset_dir).resolve() if not Path(args.dataset_dir).is_absolute() else Path(args.dataset_dir)
    project_dir = (root / args.project).resolve() if not Path(args.project).is_absolute() else Path(args.project)

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    if args.clear_dataset_dir and dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    images_train = dataset_dir / "images" / "train"
    images_val = dataset_dir / "images" / "val"
    labels_train = dataset_dir / "labels" / "train"
    labels_val = dataset_dir / "labels" / "val"
    preview_dir = dataset_dir / "label_previews"
    for path in [images_train, images_val, labels_train, labels_val]:
        path.mkdir(parents=True, exist_ok=True)

    rows_by_image: dict[str, list[dict[str, str]]] = {}
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows_by_image.setdefault(row["image_name"], []).append(row)

    stems: dict[str, list[str]] = {}
    for image_name in rows_by_image.keys():
        stems.setdefault(Path(image_name).stem, []).append(image_name)
    for stem, names in stems.items():
        if len(names) > 1:
            raise ValueError(f"Images share same label stem {stem!r}: {names}")

    for image_name, rows in rows_by_image.items():
        if len(rows) > 1:
            print(f"Info: {image_name} has {len(rows)} boxes")
        if args.strict_single_box and len(rows) != 1:
            raise ValueError(f"Expected exactly one box for {image_name}, found {len(rows)}")

    image_names = sorted(rows_by_image.keys())
    random.seed(args.seed)
    random.shuffle(image_names)
    val_count = int(len(image_names) * args.val_ratio)
    val_set = set(image_names[:val_count])

    train_count = 0
    val_count_written = 0
    missing_images = 0
    previews_written = 0

    for image_name in image_names:
        image_path = images_dir / image_name
        if not image_path.exists():
            missing_images += 1
            continue

        label_lines = []
        split_is_val = image_name in val_set
        target_images_dir = images_val if split_is_val else images_train
        target_labels_dir = labels_val if split_is_val else labels_train
        target_image_path = target_images_dir / image_name

        width, height = copy_normalized_image(image_path, target_image_path)
        rows = rows_by_image[image_name]

        for row in rows:
            cx, cy, bw, bh = yolo_box_from_row(row, width, height)
            # Single class: organ
            label_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        (target_labels_dir / f"{Path(image_name).stem}.txt").write_text("".join(label_lines), encoding="utf-8")

        if previews_written < args.preview_count:
            save_label_preview(
                image_path=image_path,
                rows=rows,
                output_path=preview_dir / f"{Path(image_name).stem}.jpg",
            )
            previews_written += 1

        if split_is_val:
            val_count_written += 1
        else:
            train_count += 1

    yaml_path = dataset_dir / "organ.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {dataset_dir}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: organ",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Dataset written to: {dataset_dir}")
    print(f"Train images: {train_count}")
    print(f"Val images: {val_count_written}")
    print(f"Missing images: {missing_images}")
    print(f"Preview images: {previews_written} in {preview_dir}")
    print(f"YAML: {yaml_path}")

    if args.prepare_only or args.epochs <= 0:
        reason = "--prepare-only flag" if args.prepare_only else f"epochs={args.epochs}"
        print(f"Skipping training due to {reason}.")
        return

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ultralytics is not installed in this Python environment. Install it before training."
        ) from exc

    model = YOLO(args.weights)
    run_dir = project_dir / args.name
    if args.clear_run_dir and run_dir.exists():
        shutil.rmtree(run_dir)

    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(project_dir),
        name=args.name,
        device=args.device,
        pretrained=True,
    )


if __name__ == "__main__":
    main()