import json
from pathlib import Path
from PIL import Image

CATEGORIES = [{"id": 0, "name": "pole", "supercategory": "none"}]

DATASETS = [
    "Poles2025/roadpoles_v1",
    "Poles2025/Road_poles_iPhone",
]

def convert_split(dataset_dir, split):
    images_dir = dataset_dir / split / "images"
    labels_dir = dataset_dir / split / "labels"

    if not images_dir.exists():
        print(f"  Skipping {split}: {images_dir} not found")
        return

    image_entries = []
    annotation_entries = []
    ann_id = 0

    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.PNG")))

    for img_id, img_path in enumerate(image_files):
        with Image.open(img_path) as img:
            width, height = img.size

        image_entries.append({"id": img_id, "file_name": f"images/{img_path.name}", "width": width, "height": height})

        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        for line in label_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            abs_w = w * width
            abs_h = h * height
            abs_x = (cx * width) - (abs_w / 2)
            abs_y = (cy * height) - (abs_h / 2)
            annotation_entries.append({
                "id": ann_id, "image_id": img_id, "category_id": class_id,
                "bbox": [abs_x, abs_y, abs_w, abs_h],
                "area": abs_w * abs_h, "iscrowd": 0,
            })
            ann_id += 1

    out_path = dataset_dir / split / "_annotations.coco.json"
    with open(out_path, "w") as f:
        json.dump({"images": image_entries, "annotations": annotation_entries, "categories": CATEGORIES}, f, indent=2)

    print(f"  [{split}] {len(image_entries)} images, {len(annotation_entries)} annotations -> {out_path}")

for dataset in DATASETS:
    dataset_dir = Path(dataset)
    print(f"\nConverting: {dataset_dir}")
    for split in ("train", "valid", "test"):
        convert_split(dataset_dir, split)

print("\nDone!")