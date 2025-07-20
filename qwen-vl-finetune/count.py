import json

file_path = "train_dk_normalized.json"

with open(file_path, 'r') as f:
    data = json.load(f)

print(f"✅ Total entries: {len(data)}")
