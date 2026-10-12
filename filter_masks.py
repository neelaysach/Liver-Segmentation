from pathlib import Path
import csv

# Paths
output_dir = Path("batch6_outputs")
masks_dir = output_dir / "masks"
overlays_dir = output_dir / "overlays"
manifest_path = output_dir / "manifest.csv"

# Get the names of all images that actually exist in overlays
overlay_names = {
    f.name
    for f in overlays_dir.iterdir()
    if f.is_file()
}

# ------------------------------------------------------------------
# Delete masks without matching overlays
# ------------------------------------------------------------------

deleted_masks = 0

for mask_file in masks_dir.iterdir():
    if not mask_file.is_file():
        continue

    if mask_file.name not in overlay_names:
        print(f"Deleting mask: {mask_file.name}")
        mask_file.unlink()
        deleted_masks += 1

# ------------------------------------------------------------------
# Remove manifest rows whose overlay_path doesn't actually exist
# ------------------------------------------------------------------

with manifest_path.open("r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames

    rows = []
    removed_rows = 0

    for row in reader:

        # Example:
        # "overlays/image123.jpg" -> "image123.jpg"
        overlay_name = Path(row["overlay_path"]).name

        if overlay_name in overlay_names:
            rows.append(row)
        else:
            print(
                f"Removing manifest row: "
                f"{row['overlay_path']}"
            )
            removed_rows += 1

# ------------------------------------------------------------------
# Rewrite manifest with only valid rows
# ------------------------------------------------------------------

with manifest_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(rows)

print("\nFinished.")
print(f"Deleted masks: {deleted_masks}")
print(f"Removed manifest rows: {removed_rows}")