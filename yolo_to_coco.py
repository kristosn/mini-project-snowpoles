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
    output_file = dataset_dir / split / "_annotations.coco.json"

    if not images_dir.exists():
        print(f"  Skipping {split}: {images_dir} not found")
        return

    image_entries = []
    annotation_entries = []
    annotation_id = 0

    image_paths = sorted([*images_dir.glob("*.jpg"), *images_dir.glob("*.PNG")])

    for image_id, image_path in enumerate(image_paths):
        with Image.open(image_path) as image:
            image_width, image_height = image.size

        image_entries.append({
            "id": image_id,
            "file_name": f"images/{image_path.name}",
            "width": image_width,
            "height": image_height
        })

        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        for line in label_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            center_x, center_y, box_width, box_height = map(float, parts[1:5])

            abs_width = box_width * image_width
            abs_height = box_height * image_height
            abs_x = (center_x * image_width) - (abs_width / 2)
            abs_y = (center_y * image_height) - (abs_height / 2)

            annotation_entries.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [abs_x, abs_y, abs_width, abs_height],
                "area": abs_width * abs_height,
                "iscrowd": 0,
            })
            annotation_id += 1

    with open(output_file, "w") as f:
        json.dump({
            "images": image_entries,
            "annotations": annotation_entries,
            "categories": CATEGORIES
        }, f, indent=2)

    print(f"  [{split}] {len(image_entries)} images, {len(annotation_entries)} annotations -> {output_file}")

for dataset in DATASETS:
    dataset_dir = Path(dataset)
    print(f"\nConverting: {dataset_dir}")
    for split in ("train", "valid", "test"):
        convert_split(dataset_dir, split)

print("\nDone!")