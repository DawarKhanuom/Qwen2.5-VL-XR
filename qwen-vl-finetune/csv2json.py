import pandas as pd
import json

# Path to your CSV file
csv_path = "/ibex/scratch/khand0b/qwen/qwen-vl-finetune/train_dk.csv"
json_output_path = "/ibex/scratch/khand0b/qwen/qwen-vl-finetune/train_dk.json"

print(f"📄 Loading CSV: {csv_path}")
# No headers, so we assign them manually
df = pd.read_csv(csv_path, header=None, names=["image_name", "genre"])

json_data = []

for _, row in df.iterrows():
    image_name = row["image_name"].strip()
    genre = str(row["genre"]).strip().lower()
    
    entry = {
        "image": image_name,
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nWhat is the genre of this book?"
            },
            {
                "from": "gpt",
                "value": genre
            }
        ]
    }
    json_data.append(entry)

with open(json_output_path, "w") as f:
    json.dump(json_data, f, indent=2)

print(f"✅ Converted {len(json_data)} entries to JSON and saved to: {json_output_path}")
