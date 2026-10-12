#!/usr/bin/env python3
"""Extract transparent polygon segments from an image using VIA-style JSON labels.

Example:
  python3 extract_segments.py \
    --image AGEL332_4032_3024_9a98b4428f50b013b69f4288b5f24740.jpeg \
    --labels labels_my-project-name_2026-05-25-10-15-28.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create transparent PNGs for polygon regions (default: region IDs 0,1,2) "
            "for a given image filename."
        )
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Image filename or path present in the labels JSON (usually a .jpeg).",
    )
    parser.add_argument(
        "--labels",
        default="labels_my-project-name_2026-05-25-10-15-28.json",
        help="Path to labels JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default="segments",
        help="Directory to write segment PNG files.",
    )
    parser.add_argument(
        "--region-ids",
        nargs="*",
        default=["0", "1", "2"],
        help="Region IDs to export (default: 0 1 2).",
    )
    parser.add_argument(
        "--full-size",
        action="store_true",
        help="Keep output at full image size. By default, outputs are cropped to polygon bbox.",
    )
    return parser.parse_args()


def _find_entry_for_image(data: Dict, image_arg: str) -> Tuple[str, Dict]:
    image_name = Path(image_arg).name

    if image_name in data:
        return image_name, data[image_name]

    for key, value in data.items():
        if isinstance(value, dict) and value.get("filename") == image_name:
            return key, value

    raise KeyError(f"Image '{image_name}' not found in labels JSON.")


def _to_points(all_x: Iterable[float], all_y: Iterable[float]) -> List[Tuple[int, int]]:
    points = [(int(round(x)), int(round(y))) for x, y in zip(all_x, all_y)]
    if len(points) < 3:
        raise ValueError("Polygon must contain at least 3 points.")
    return points


def _polygon_bbox(points: List[Tuple[int, int]], width: int, height: int) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    left = max(0, min(xs))
    top = max(0, min(ys))
    right = min(width, max(xs) + 1)
    bottom = min(height, max(ys) + 1)
    if left >= right or top >= bottom:
        raise ValueError("Invalid polygon bounds after clipping.")
    return left, top, right, bottom


def sanitize_label(label: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label.strip())
    return cleaned or "region"


def main() -> None:
    args = parse_args()

    labels_path = Path(args.labels)
    with labels_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    _, entry = _find_entry_for_image(data, args.image)
    regions = entry.get("regions", {})

    image_path = Path(args.image)
    if not image_path.exists():
        # If only a filename is given, try next to labels JSON.
        image_path = labels_path.parent / Path(args.image).name
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {args.image}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path).convert("RGBA") as img:
        exported = 0
        for region_id in args.region_ids:
            region = regions.get(str(region_id))
            if not region:
                continue

            shape = region.get("shape_attributes", {})
            if shape.get("name") != "polygon":
                continue

            points = _to_points(shape.get("all_points_x", []), shape.get("all_points_y", []))

            mask = Image.new("L", img.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.polygon(points, fill=255)

            segment = Image.new("RGBA", img.size, (0, 0, 0, 0))
            segment.paste(img, (0, 0), mask)

            label = sanitize_label(region.get("region_attributes", {}).get("label", f"region_{region_id}"))
            stem = Path(entry.get("filename", image_path.name)).stem

            if args.full_size:
                out_img = segment
            else:
                bbox = _polygon_bbox(points, img.width, img.height)
                out_img = segment.crop(bbox)

            out_path = out_dir / f"{stem}_region_{region_id}_{label}.png"
            out_img.save(out_path)
            print(f"Saved: {out_path}")
            exported += 1

    if exported == 0:
        print("No matching polygon regions were exported. Check --region-ids and labels JSON.")


if __name__ == "__main__":
    main()