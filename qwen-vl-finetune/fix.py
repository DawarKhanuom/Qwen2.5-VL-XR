import json

json_path = "/ibex/scratch/khand0b/qwen/data/train.json"

# Load original data
with open(json_path, "r") as f:
    data = json.load(f)

# Fix paths
for item in data:
    img_path = item.get("image", "")
    img_path = img_path.replace("\\", "/")  # Normalize slashes
    if img_path.startswith("224x224/"):
        img_path = img_path[len("224x224/"):]  # Remove prefix
    item["image"] = img_path

# Overwrite original file
with open(json_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"✅ Fixed paths and overwritten: {json_path}")
