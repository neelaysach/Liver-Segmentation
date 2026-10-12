#!/usr/bin/env python3

from pathlib import Path
import csv
import cv2
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

INPUT_DIR = Path("batch7")
OUTPUT_DIR = Path("batch7_usable")

# Reject if MORE than this fraction is near-white
MAX_WHITE_FRACTION = 0.30

# Reject if MORE than this fraction is near-black
MAX_BLACK_FRACTION = 0.10

# Reject if average HSV saturation is BELOW this
MIN_MEAN_SATURATION = 0.10

# Reject if grayscale P95 - P5 is BELOW this
# Values are on a 0-255 scale.
MIN_CONTRAST_RANGE = 30

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# IMAGE QUALITY CHECK
# ============================================================

def analyze_image(image_path: Path) -> dict:
    """
    Analyze an image and determine whether it is usable.

    Reject if:
        white pixels > 30%
        black pixels > 10%
        mean saturation < 0.10
        contrast range < 30
    """

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        return {
            "usable": False,
            "reason": "could_not_read",
            "white_fraction": None,
            "black_fraction": None,
            "mean_saturation": None,
            "contrast_range": None,
        }

    # OpenCV loads images as BGR.
    # Convert to RGB just to make channel meaning intuitive.
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --------------------------------------------------------
    # WHITE PIXELS
    # --------------------------------------------------------
    #
    # Pixel counts as white only if ALL three channels > 200.
    #

    white_mask = np.all(rgb > 200, axis=2)
    white_fraction = float(np.mean(white_mask))

    # --------------------------------------------------------
    # BLACK PIXELS
    # --------------------------------------------------------
    #
    # Pixel counts as black only if ALL three channels < 50.
    #

    black_mask = np.all(rgb < 50, axis=2)
    black_fraction = float(np.mean(black_mask))

    # --------------------------------------------------------
    # SATURATION
    # --------------------------------------------------------

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # OpenCV saturation ranges from 0 to 255.
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0

    mean_saturation = float(np.mean(saturation))

    # --------------------------------------------------------
    # CONTRAST
    # --------------------------------------------------------
    #
    # Rather than simply using standard deviation, use the
    # distance between the 5th and 95th grayscale percentiles.
    #
    # This ignores small numbers of extreme pixels and asks:
    #
    # "Do most pixels actually cover a meaningful intensity
    # range?"
    #

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    percentile_5 = float(np.percentile(gray, 5))
    percentile_95 = float(np.percentile(gray, 95))

    contrast_range = percentile_95 - percentile_5

    # --------------------------------------------------------
    # APPLY FILTERS
    # --------------------------------------------------------

    reasons = []

    if white_fraction > MAX_WHITE_FRACTION:
        reasons.append("too_white")

    if black_fraction > MAX_BLACK_FRACTION:
        reasons.append("too_black")

    if mean_saturation < MIN_MEAN_SATURATION:
        reasons.append("low_saturation")

    if contrast_range < MIN_CONTRAST_RANGE:
        reasons.append("low_contrast")

    usable = len(reasons) == 0

    return {
        "usable": usable,
        "reason": "|".join(reasons) if reasons else "usable",
        "white_fraction": white_fraction,
        "black_fraction": black_fraction,
        "mean_saturation": mean_saturation,
        "contrast_range": contrast_range,
    }


# ============================================================
# CREATE UNIQUE SYMLINK NAME
# ============================================================

def make_link_name(image_path: Path) -> str:
    """
    Flatten the batch7 folder structure while avoiding filename
    collisions.

    Example:

        batch2/folder_1/image001.jpg

    becomes:

        folder_1__image001.jpg
    """

    relative = image_path.relative_to(INPUT_DIR)

    parent_parts = relative.parts[:-1]

    if parent_parts:
        prefix = "__".join(parent_parts)
        return f"{prefix}__{relative.name}"

    return relative.name


# ============================================================
# MAIN
# ============================================================

def main():

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {INPUT_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = OUTPUT_DIR / "filter_manifest.csv"

    # --------------------------------------------------------
    # FIND ALL IMAGES
    # --------------------------------------------------------

    image_files = sorted(
        path
        for path in INPUT_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    print("=" * 70)
    print("BATCH 2 IMAGE QUALITY FILTER")
    print("=" * 70)
    print(f"Input:  {INPUT_DIR.resolve()}")
    print(f"Output: {OUTPUT_DIR.resolve()}")
    print(f"Images found: {len(image_files)}")
    print("=" * 70)

    kept = 0
    rejected = 0

    rows = []

    # --------------------------------------------------------
    # PROCESS IMAGES
    # --------------------------------------------------------

    for index, image_path in enumerate(image_files, start=1):

        result = analyze_image(image_path)

        row = {
            "image_name": image_path.name,
            "original_path": str(image_path.resolve()),
            "usable": result["usable"],
            "reason": result["reason"],
            "white_fraction": (
                f"{result['white_fraction']:.4f}"
                if result["white_fraction"] is not None
                else ""
            ),
            "black_fraction": (
                f"{result['black_fraction']:.4f}"
                if result["black_fraction"] is not None
                else ""
            ),
            "mean_saturation": (
                f"{result['mean_saturation']:.4f}"
                if result["mean_saturation"] is not None
                else ""
            ),
            "contrast_range": (
                f"{result['contrast_range']:.2f}"
                if result["contrast_range"] is not None
                else ""
            ),
        }

        rows.append(row)

        # ----------------------------------------------------
        # ACCEPTED IMAGE
        # ----------------------------------------------------

        if result["usable"]:

            link_path = OUTPUT_DIR / image_path.name

            # Remove an existing link with the same name.
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()

            # Absolute symlink is safer if folders get moved around
            # independently.
            link_path.symlink_to(image_path.resolve())

            kept += 1

            print(
                f"[{index}/{len(image_files)}] KEEP   "
                f"{image_path.name} "
                f"(white={result['white_fraction']:.2%}, "
                f"black={result['black_fraction']:.2%}, "
                f"sat={result['mean_saturation']:.3f}, "
                f"contrast={result['contrast_range']:.1f})"
            )

        # ----------------------------------------------------
        # REJECTED IMAGE
        # ----------------------------------------------------

        else:

            rejected += 1

            print(
                f"[{index}/{len(image_files)}] REJECT "
                f"{image_path.name} "
                f"[{result['reason']}]"
            )

    # --------------------------------------------------------
    # SAVE MANIFEST
    # --------------------------------------------------------

    fieldnames = [
        "image_name",
        "original_path",
        "usable",
        "reason",
        "white_fraction",
        "black_fraction",
        "mean_saturation",
        "contrast_range",
    ]

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINISHED")
    print("=" * 70)
    print(f"Total images:    {len(image_files)}")
    print(f"Usable:          {kept}")
    print(f"Rejected:        {rejected}")

    if image_files:
        print(
            f"Percent usable:  "
            f"{100 * kept / len(image_files):.1f}%"
        )

    print()
    print(f"Usable image links: {OUTPUT_DIR.resolve()}")
    print(f"Manifest:           {manifest_path.resolve()}")


if __name__ == "__main__":
    main()