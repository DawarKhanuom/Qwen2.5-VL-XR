import json
import os

annotation_path = "/ibex/scratch/khand0b/qwen/data/train.json"
image_root = "/ibex/scratch/khand0b/qwen/data/train/"

with open(annotation_path, 'r') as f:
    data = json.load(f)

missing = 0
for i, item in enumerate(data):
    image_path = os.path.join(image_root, item["image"])
    if not os.path.exists(image_path):
        print(f"[Missing] {image_path}")
        missing += 1
    elif i < 3:
        print(f"[OK] {image_path} | Q1: {item['conversations'][0]['value']}")

print(f"Done. Total images checked: {len(data)}, missing: {missing}")
