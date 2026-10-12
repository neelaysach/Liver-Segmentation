#!/usr/bin/env python3
"""
Strict YOLO -> MedSAM test pipeline.

Why this version exists:
- Requires an explicit trained YOLO best.pt path. It never falls back to COCO weights.
- Lets Ultralytics load each image from its file path, matching normal YOLO inference.
- Loads the same stored pixels with OpenCV for MedSAM, avoiding PIL/EXIF orientation surprises.
- Verifies YOLO's original image shape against OpenCV's image shape.
- Saves a YOLO-only box preview before MedSAM runs.
- Logs every predicted box and confidence to manifest.csv.
- Optionally pads the YOLO box before passing it to MedSAM.
- Saves binary masks and combined QA overlays.

Expected project layout is flexible because all important paths are command-line arguments.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test an explicitly selected YOLO best.pt checkpoint and optionally run MedSAM."
    )
    parser.add_argument("--images-dir", required=True, help="Directory containing test images.")
    parser.add_argument("--yolo-weights", required=True, help="Exact path to trained best.pt.")
    parser.add_argument("--output-dir", default="pipeline_test_output", help="Output directory.")

    parser.add_argument("--yolo-conf", type=float, default=0.10)
    parser.add_argument("--yolo-iou", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--max-boxes", type=int, default=1)
    parser.add_argument(
        "--box-padding",
        type=float,
        default=0.08,
        help="Fraction of box width/height added on every side before MedSAM.",
    )

    parser.add_argument(
        "--yolo-only",
        action="store_true",
        help="Only test YOLO and save box previews; do not load MedSAM.",
    )
    parser.add_argument(
        "--medsam-repo",
        default="MedSAM",
        help="Path to the MedSAM repository containing segment_anything.",
    )
    parser.add_argument(
        "--medsam-checkpoint",
        default="MedSAM/work_dir/MedSAM/medsam_vit_b.pth",
        help="Path to medsam_vit_b.pth.",
    )
    parser.add_argument("--device", default="auto", help="auto, cuda:0, mps, or cpu.")
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def resolve_existing_path(raw_path: str, description: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def validate_args(args: argparse.Namespace) -> None:
    for name in ("yolo_conf", "yolo_iou", "mask_threshold"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1.")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be positive.")
    if args.max_boxes <= 0:
        raise ValueError("--max-boxes must be positive.")
    if args.box_padding < 0:
        raise ValueError("--box-padding cannot be negative.")


def pad_box(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    fraction: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    pad_x = int(round(box_width * fraction))
    pad_y = int(round(box_height * fraction))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(image_width, x2 + pad_x),
        min(image_height, y2 + pad_y),
    )


def prepare_medsam_image(rgb_image: np.ndarray, device: str) -> torch.Tensor:
    resized = cv2.resize(rgb_image, (1024, 1024), interpolation=cv2.INTER_CUBIC)
    resized = resized.astype(np.float32)
    minimum = float(resized.min())
    maximum = float(resized.max())
    resized = (resized - minimum) / max(maximum - minimum, 1e-8)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


@torch.no_grad()
def medsam_probability(
    medsam_model: torch.nn.Module,
    image_embedding: torch.Tensor,
    box_1024: np.ndarray,
    output_height: int,
    output_width: int,
) -> np.ndarray:
    box_tensor = torch.as_tensor(
        box_1024[None, None, :],
        dtype=torch.float32,
        device=image_embedding.device,
    )
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_tensor,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    probabilities = torch.sigmoid(low_res_logits)
    probabilities = F.interpolate(
        probabilities,
        size=(output_height, output_width),
        mode="bilinear",
        align_corners=False,
    )
    return probabilities[0, 0].detach().cpu().numpy()


def draw_box_preview(
    bgr_image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    scores: list[float],
) -> np.ndarray:
    preview = bgr_image.copy()
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 0, 0), 4)
        cv2.putText(
            preview,
            f"organ {score:.3f}",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return preview


def draw_combined_overlay(
    bgr_image: np.ndarray,
    mask: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    overlay = bgr_image.copy()
    yellow = np.zeros_like(overlay)
    yellow[:, :] = (0, 255, 255)
    foreground = mask.astype(bool)
    overlay[foreground] = cv2.addWeighted(
        overlay[foreground], 0.45, yellow[foreground], 0.55, 0
    )
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 4)
    return overlay


def main() -> None:
    args = parse_args()
    validate_args(args)

    images_dir = resolve_existing_path(args.images_dir, "Images directory")
    yolo_weights = resolve_existing_path(args.yolo_weights, "YOLO weights")
    output_dir = Path(args.output_dir).expanduser().resolve()

    box_preview_dir = output_dir / "yolo_box_previews"
    mask_dir = output_dir / "masks"
    overlay_dir = output_dir / "medsam_overlays"
    box_preview_dir.mkdir(parents=True, exist_ok=True)
    if not args.yolo_only:
        mask_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Ultralytics is not installed. Run: pip install ultralytics"
        ) from exc

    print("=" * 72)
    print("STRICT PIPELINE CONFIGURATION")
    print(f"Images:       {images_dir}")
    print(f"YOLO weights: {yolo_weights}")
    print(f"Output:       {output_dir}")
    print(f"YOLO-only:    {args.yolo_only}")
    print("=" * 72)

    yolo = YOLO(str(yolo_weights))
    print(f"Loaded YOLO task: {getattr(yolo, 'task', 'unknown')}")
    print(f"YOLO class names: {getattr(yolo, 'names', 'unknown')}")

    device = resolve_device(args.device)
    medsam_model = None
    if not args.yolo_only:
        medsam_repo = resolve_existing_path(args.medsam_repo, "MedSAM repository")
        medsam_checkpoint = resolve_existing_path(
            args.medsam_checkpoint, "MedSAM checkpoint"
        )
        if str(medsam_repo) not in sys.path:
            sys.path.insert(0, str(medsam_repo))
        from segment_anything import sam_model_registry

        print(f"MedSAM device:     {device}")
        print(f"MedSAM checkpoint: {medsam_checkpoint}")
        medsam_model = sam_model_registry["vit_b"](
            checkpoint=str(medsam_checkpoint)
        ).to(device)
        medsam_model.eval()

    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if args.max_images is not None:
        image_files = image_files[: args.max_images]
    if not image_files:
        raise ValueError(f"No supported images found in {images_dir}")

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "image_name",
            "image_width",
            "image_height",
            "checkpoint",
            "detections_before_limit",
            "boxes_used",
            "scores",
            "boxes_xyxy",
            "padded_boxes_xyxy",
            "box_preview",
            "mask_path",
            "overlay_path",
            "status",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for index, image_path in enumerate(image_files, start=1):
            row = {
                "image_name": image_path.name,
                "checkpoint": str(yolo_weights),
            }
            try:
                # OpenCV loads stored pixel orientation without silently applying EXIF rotation.
                bgr_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if bgr_image is None:
                    raise ValueError("OpenCV could not decode the image.")
                height, width = bgr_image.shape[:2]
                row["image_width"] = width
                row["image_height"] = height

                result = yolo.predict(
                    source=str(image_path),
                    conf=args.yolo_conf,
                    iou=args.yolo_iou,
                    imgsz=args.imgsz,
                    classes=[args.class_id],
                    verbose=False,
                )[0]

                yolo_shape = tuple(int(v) for v in result.orig_shape)
                if yolo_shape != (height, width):
                    raise ValueError(
                        "Image orientation/shape mismatch: "
                        f"YOLO used {yolo_shape}, OpenCV used {(height, width)}."
                    )

                if result.boxes is None or len(result.boxes) == 0:
                    row.update(
                        {
                            "detections_before_limit": 0,
                            "boxes_used": 0,
                            "scores": "",
                            "boxes_xyxy": "",
                            "padded_boxes_xyxy": "",
                            "box_preview": "",
                            "mask_path": "",
                            "overlay_path": "",
                            "status": "no_detection",
                        }
                    )
                    writer.writerow(row)
                    print(f"[{index}/{len(image_files)}] NO DETECTION: {image_path.name}")
                    continue

                all_xyxy = result.boxes.xyxy.detach().cpu().numpy()
                all_scores = result.boxes.conf.detach().cpu().numpy()
                order = np.argsort(-all_scores)
                selected = order[: args.max_boxes]

                raw_boxes: list[tuple[int, int, int, int]] = []
                padded_boxes: list[tuple[int, int, int, int]] = []
                selected_scores: list[float] = []

                for selected_index in selected:
                    x1f, y1f, x2f, y2f = all_xyxy[selected_index]
                    x1 = int(np.floor(np.clip(x1f, 0, width - 1)))
                    y1 = int(np.floor(np.clip(y1f, 0, height - 1)))
                    x2 = int(np.ceil(np.clip(x2f, x1 + 1, width)))
                    y2 = int(np.ceil(np.clip(y2f, y1 + 1, height)))
                    raw_box = (x1, y1, x2, y2)
                    raw_boxes.append(raw_box)
                    padded_boxes.append(
                        pad_box(raw_box, width, height, args.box_padding)
                    )
                    selected_scores.append(float(all_scores[selected_index]))

                box_preview = draw_box_preview(
                    bgr_image, raw_boxes, selected_scores
                )
                box_preview_path = box_preview_dir / f"{image_path.stem}.png"
                if not cv2.imwrite(str(box_preview_path), box_preview):
                    raise IOError(f"Could not save {box_preview_path}")

                row.update(
                    {
                        "detections_before_limit": len(all_xyxy),
                        "boxes_used": len(raw_boxes),
                        "scores": "|".join(f"{score:.6f}" for score in selected_scores),
                        "boxes_xyxy": "|".join(
                            ",".join(str(v) for v in box) for box in raw_boxes
                        ),
                        "padded_boxes_xyxy": "|".join(
                            ",".join(str(v) for v in box) for box in padded_boxes
                        ),
                        "box_preview": str(box_preview_path.relative_to(output_dir)),
                    }
                )

                if args.yolo_only:
                    row.update(
                        {
                            "mask_path": "",
                            "overlay_path": "",
                            "status": "yolo_ok",
                        }
                    )
                    writer.writerow(row)
                    print(
                        f"[{index}/{len(image_files)}] YOLO OK: {image_path.name} "
                        f"box={raw_boxes[0]} score={selected_scores[0]:.3f}"
                    )
                    continue

                assert medsam_model is not None
                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                image_tensor = prepare_medsam_image(rgb_image, device)
                with torch.no_grad():
                    image_embedding = medsam_model.image_encoder(image_tensor)

                probability_masks: list[np.ndarray] = []
                scale = np.array(
                    [1024.0 / width, 1024.0 / height,
                     1024.0 / width, 1024.0 / height],
                    dtype=np.float32,
                )
                for box in padded_boxes:
                    box_1024 = np.asarray(box, dtype=np.float32) * scale
                    probability_masks.append(
                        medsam_probability(
                            medsam_model,
                            image_embedding,
                            box_1024,
                            height,
                            width,
                        )
                    )

                combined_probability = np.maximum.reduce(probability_masks)
                binary_mask = (combined_probability >= args.mask_threshold).astype(
                    np.uint8
                )

                mask_path = mask_dir / f"{image_path.stem}.png"
                if not cv2.imwrite(str(mask_path), binary_mask * 255):
                    raise IOError(f"Could not save {mask_path}")

                combined_overlay = draw_combined_overlay(
                    bgr_image, binary_mask, padded_boxes
                )
                overlay_path = overlay_dir / f"{image_path.stem}.png"
                if not cv2.imwrite(str(overlay_path), combined_overlay):
                    raise IOError(f"Could not save {overlay_path}")

                row.update(
                    {
                        "mask_path": str(mask_path.relative_to(output_dir)),
                        "overlay_path": str(overlay_path.relative_to(output_dir)),
                        "status": "ok",
                    }
                )
                writer.writerow(row)
                print(
                    f"[{index}/{len(image_files)}] PIPELINE OK: {image_path.name} "
                    f"box={raw_boxes[0]} padded={padded_boxes[0]} "
                    f"score={selected_scores[0]:.3f}"
                )

            except Exception as exc:
                row.update(
                    {
                        "mask_path": "",
                        "overlay_path": "",
                        "status": f"error: {type(exc).__name__}: {exc}",
                    }
                )
                writer.writerow(row)
                print(
                    f"[{index}/{len(image_files)}] ERROR: {image_path.name}: "
                    f"{type(exc).__name__}: {exc}"
                )

    print()
    print(f"Finished. Inspect YOLO boxes first: {box_preview_dir}")
    print(f"Manifest: {manifest_path}")
    if not args.yolo_only:
        print(f"Masks: {mask_dir}")
        print(f"MedSAM overlays: {overlay_dir}")


if __name__ == "__main__":
    main()