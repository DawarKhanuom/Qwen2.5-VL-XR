import json
import re

# Path to your JSON file
input_path = "train_dk_shuffled.json" #/ibex/scratch/khand0b/qwen/data/qwenVL_one_q.json"
output_path = "train_dk_normalized.json"

# Optional: Define mappings to unify genre labels
GENRE_MAP = {
    "sci-fi": "science fiction & fantasy",
    "science fiction": "science fiction & fantasy",
    "science fiction and fantasy": "science fiction & fantasy",
    "thriller": "mystery, thriller & suspense",
    "suspense": "mystery, thriller & suspense",
    "mystery": "mystery, thriller & suspense",
    "mystery, thriller and suspense": "mystery, thriller & suspense",
    "calendars.": "calendars",
    # Add more mappings as needed
}

def normalize_label(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation (except &)
    text = re.sub(r"[^\w\s&]", "", text)
    # Strip whitespace
    text = text.strip()
    # Apply mapping
    return GENRE_MAP.get(text, text)

# Load data
with open(input_path, "r") as f:
    data = json.load(f)

# Normalize labels
for entry in data:
    if "conversations" in entry and len(entry["conversations"]) >= 2:
        original = entry["conversations"][1]["value"]
        normalized = normalize_label(original)
        entry["conversations"][1]["value"] = normalized

# Save the normalized data
with open(output_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"✅ Normalized dataset saved to: {output_path}")
