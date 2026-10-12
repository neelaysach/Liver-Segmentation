#!/usr/bin/env python3
"""
Run MedSAM masking on every image using YOLO-derived bounding boxes.

Pipeline:
    image
      -> YOLO predicts organ box
      -> optional box padding
      -> MedSAM segments organ
      -> save binary mask PNG
      -> save QA overlay PNG
      -> log results to manifest.csv

This script requires an explicit trained YOLO checkpoint and never falls back
to generic pretrained weights.
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


SUPPORTED_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MedSAM masks using YOLO-predicted bounding boxes."
    )

    parser.add_argument(
        "--images-dir",
        required=True,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--yolo-weights",
        required=True,
        help="Exact path to trained YOLO best.pt.",
    )
    parser.add_argument(
        "--medsam-repo",
        default="MedSAM",
        help="Path to the MedSAM repository.",
    )
    parser.add_argument(
        "--medsam-checkpoint",
        default="MedSAM/work_dir/MedSAM/medsam_vit_b.pth",
        help="Path to medsam_vit_b.pth.",
    )
    parser.add_argument(
        "--output-dir",
        default="yolo_medsam_masks",
        help="Directory for masks, overlays, and manifest.csv.",
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cuda:0, mps, or cpu.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference size.",
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.10,
        help="YOLO confidence threshold.",
    )
    parser.add_argument(
        "--yolo-iou",
        type=float,
        default=0.70,
        help="YOLO NMS IoU threshold.",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="YOLO class ID representing the organ.",
    )
    parser.add_argument(
        "--max-boxes",
        type=int,
        default=1,
        help="Maximum number of highest-confidence boxes used per image.",
    )
    parser.add_argument(
        "--box-padding",
        type=float,
        default=0.05,
        help="Fractional padding added on every side of each YOLO box.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.50,
        help="Threshold used to convert MedSAM probabilities to a binary mask.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap for testing.",
    )

    return parser.parse_args()


def require_path(raw_path: str, description: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested

    if torch.cuda.is_available():
        return "cuda:0"

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


def validate_args(args: argparse.Namespace) -> None:
    for name in ("yolo_conf", "yolo_iou", "mask_threshold"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be between 0 and 1."
            )

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

    width = x2 - x1
    height = y2 - y1

    pad_x = int(round(width * fraction))
    pad_y = int(round(height * fraction))

    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(image_width, x2 + pad_x),
        min(image_height, y2 + pad_y),
    )


def prepare_medsam_tensor(
    rgb_image: np.ndarray,
    device: str,
) -> torch.Tensor:
    resized = cv2.resize(
        rgb_image,
        (1024, 1024),
        interpolation=cv2.INTER_CUBIC,
    ).astype(np.float32)

    minimum = float(resized.min())
    maximum = float(resized.max())

    resized = (resized - minimum) / max(
        maximum - minimum,
        1e-8,
    )

    return (
        torch.from_numpy(resized)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )


@torch.no_grad()
def medsam_probability_for_box(
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

    sparse_embeddings, dense_embeddings = (
        medsam_model.prompt_encoder(
            points=None,
            boxes=box_tensor,
            masks=None,
        )
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

    return probabilities[0, 0].cpu().numpy()


def make_overlay(
    bgr_image: np.ndarray,
    binary_mask: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    overlay = bgr_image.copy()

    mask_pixels = binary_mask.astype(bool)
    yellow_layer = np.zeros_like(overlay)
    yellow_layer[:, :] = (0, 255, 255)

    overlay[mask_pixels] = cv2.addWeighted(
        overlay[mask_pixels],
        0.45,
        yellow_layer[mask_pixels],
        0.55,
        0,
    )

    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            4,
        )

    return overlay


def main() -> None:
    args = parse_args()
    validate_args(args)

    images_dir = require_path(
        args.images_dir,
        "Images directory",
    )
    yolo_weights = require_path(
        args.yolo_weights,
        "YOLO checkpoint",
    )
    medsam_repo = require_path(
        args.medsam_repo,
        "MedSAM repository",
    )
    medsam_checkpoint = require_path(
        args.medsam_checkpoint,
        "MedSAM checkpoint",
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"

    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Ultralytics is not installed. Run: pip install ultralytics"
        ) from exc

    if str(medsam_repo) not in sys.path:
        sys.path.insert(0, str(medsam_repo))

    try:
        from segment_anything import sam_model_registry
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import segment_anything from the MedSAM repository."
        ) from exc

    device = resolve_device(args.device)

    print("=" * 72)
    print("YOLO -> MEDSAM MASK GENERATION")
    print(f"Images:            {images_dir}")
    print(f"YOLO checkpoint:   {yolo_weights}")
    print(f"MedSAM checkpoint: {medsam_checkpoint}")
    print(f"Device:            {device}")
    print(f"Output:            {output_dir}")
    print("=" * 72)

    yolo = YOLO(str(yolo_weights))

    medsam_model = sam_model_registry["vit_b"](
        checkpoint=str(medsam_checkpoint)
    ).to(device)
    medsam_model.eval()

    image_files = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
    )

    if args.max_images is not None:
        image_files = image_files[: args.max_images]

    if not image_files:
        raise ValueError(
            f"No supported image files found in {images_dir}"
        )

    manifest_path = output_dir / "manifest.csv"

    with manifest_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        fieldnames = [
            "image_name",
            "width",
            "height",
            "yolo_score",
            "raw_box_xyxy",
            "padded_box_xyxy",
            "mask_path",
            "overlay_path",
            "status",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for index, image_path in enumerate(
            image_files,
            start=1,
        ):
            row: dict[str, object] = {
                "image_name": image_path.name,
            }

            try:
                bgr_image = cv2.imread(
                    str(image_path),
                    cv2.IMREAD_COLOR,
                )

                if bgr_image is None:
                    raise ValueError(
                        "OpenCV could not decode image."
                    )

                height, width = bgr_image.shape[:2]
                row["width"] = width
                row["height"] = height

                result = yolo.predict(
                    source=str(image_path),
                    conf=args.yolo_conf,
                    iou=args.yolo_iou,
                    imgsz=args.imgsz,
                    classes=[args.class_id],
                    verbose=False,
                )[0]

                yolo_shape = tuple(
                    int(value)
                    for value in result.orig_shape
                )

                if yolo_shape != (height, width):
                    raise ValueError(
                        "Image shape mismatch: "
                        f"YOLO={yolo_shape}, "
                        f"OpenCV={(height, width)}"
                    )

                if (
                    result.boxes is None
                    or len(result.boxes) == 0
                ):
                    row.update(
                        {
                            "yolo_score": "",
                            "raw_box_xyxy": "",
                            "padded_box_xyxy": "",
                            "mask_path": "",
                            "overlay_path": "",
                            "status": "no_detection",
                        }
                    )
                    writer.writerow(row)

                    print(
                        f"[{index}/{len(image_files)}] "
                        f"No detection: {image_path.name}"
                    )
                    continue

                xyxy = (
                    result.boxes.xyxy
                    .detach()
                    .cpu()
                    .numpy()
                )
                scores = (
                    result.boxes.conf
                    .detach()
                    .cpu()
                    .numpy()
                )

                order = np.argsort(-scores)

                # Keep only detections whose confidence exceeds 0.8
                selected_indices = [
                    i for i in order
                    if scores[i] > 0.80
                ][: args.max_boxes]

                # Skip this image if nothing passes the threshold
                if len(selected_indices) == 0:
                    row.update(
                        {
                            "yolo_score": "",
                            "raw_box_xyxy": "",
                            "padded_box_xyxy": "",
                            "mask_path": "",
                            "overlay_path": "",
                            "status": "low_confidence",
                        }
                    )
                    writer.writerow(row)

                    print(
                        f"[{index}/{len(image_files)}] "
                        f"Skipped (score <= 0.80): {image_path.name}"
                    )
                    continue

                raw_boxes: list[
                    tuple[int, int, int, int]
                ] = []
                padded_boxes: list[
                    tuple[int, int, int, int]
                ] = []
                selected_scores: list[float] = []

                for selected_index in selected_indices:
                    x1f, y1f, x2f, y2f = (
                        xyxy[selected_index]
                    )

                    x1 = int(
                        np.floor(
                            np.clip(
                                x1f,
                                0,
                                width - 1,
                            )
                        )
                    )
                    y1 = int(
                        np.floor(
                            np.clip(
                                y1f,
                                0,
                                height - 1,
                            )
                        )
                    )
                    x2 = int(
                        np.ceil(
                            np.clip(
                                x2f,
                                x1 + 1,
                                width,
                            )
                        )
                    )
                    y2 = int(
                        np.ceil(
                            np.clip(
                                y2f,
                                y1 + 1,
                                height,
                            )
                        )
                    )

                    raw_box = (x1, y1, x2, y2)
                    padded_box = pad_box(
                        raw_box,
                        width,
                        height,
                        args.box_padding,
                    )

                    raw_boxes.append(raw_box)
                    padded_boxes.append(padded_box)
                    selected_scores.append(
                        float(scores[selected_index])
                    )

                rgb_image = cv2.cvtColor(
                    bgr_image,
                    cv2.COLOR_BGR2RGB,
                )

                image_tensor = prepare_medsam_tensor(
                    rgb_image,
                    device,
                )

                with torch.no_grad():
                    image_embedding = (
                        medsam_model.image_encoder(
                            image_tensor
                        )
                    )

                scale = np.array(
                    [
                        1024.0 / width,
                        1024.0 / height,
                        1024.0 / width,
                        1024.0 / height,
                    ],
                    dtype=np.float32,
                )

                probability_masks: list[np.ndarray] = []

                for padded_box in padded_boxes:
                    box_1024 = (
                        np.asarray(
                            padded_box,
                            dtype=np.float32,
                        )
                        * scale
                    )

                    probability_masks.append(
                        medsam_probability_for_box(
                            medsam_model,
                            image_embedding,
                            box_1024,
                            height,
                            width,
                        )
                    )

                combined_probability = np.maximum.reduce(
                    probability_masks
                )

                binary_mask = (
                    combined_probability
                    >= args.mask_threshold
                ).astype(np.uint8)

                mask_path = (
                    masks_dir
                    / f"{image_path.stem}.png"
                )

                overlay_path = (
                    overlays_dir
                    / f"{image_path.stem}.png"
                )

                if not cv2.imwrite(
                    str(mask_path),
                    binary_mask * 255,
                ):
                    raise IOError(
                        f"Could not save mask: {mask_path}"
                    )

                overlay = make_overlay(
                    bgr_image,
                    binary_mask,
                    padded_boxes,
                )

                if not cv2.imwrite(
                    str(overlay_path),
                    overlay,
                ):
                    raise IOError(
                        f"Could not save overlay: "
                        f"{overlay_path}"
                    )

                row.update(
                    {
                        "yolo_score": "|".join(
                            f"{score:.6f}"
                            for score in selected_scores
                        ),
                        "raw_box_xyxy": "|".join(
                            ",".join(
                                str(value)
                                for value in box
                            )
                            for box in raw_boxes
                        ),
                        "padded_box_xyxy": "|".join(
                            ",".join(
                                str(value)
                                for value in box
                            )
                            for box in padded_boxes
                        ),
                        "mask_path": str(
                            mask_path.relative_to(
                                output_dir
                            )
                        ),
                        "overlay_path": str(
                            overlay_path.relative_to(
                                output_dir
                            )
                        ),
                        "status": "ok",
                    }
                )

                writer.writerow(row)

                print(
                    f"[{index}/{len(image_files)}] "
                    f"Saved: {image_path.name} "
                    f"score={selected_scores[0]:.3f}"
                )

            except Exception as exc:
                row.update(
                    {
                        "mask_path": "",
                        "overlay_path": "",
                        "status": (
                            f"error: "
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),
                    }
                )

                writer.writerow(row)

                print(
                    f"[{index}/{len(image_files)}] "
                    f"ERROR {image_path.name}: "
                    f"{type(exc).__name__}: {exc}"
                )

    print()
    print(f"Finished.")
    print(f"Masks:    {masks_dir}")
    print(f"Overlays: {overlays_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()