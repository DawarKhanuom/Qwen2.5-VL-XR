import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
import os
import pandas as pd
from tqdm import tqdm
import re

# === Configuration ===
model_path = "./checkpoints3wen50E"
test_images_dir = "/ibex/user/khand0b/qwen/data/test_dk/"
ground_truth_csv = "test100.csv"
output_predictions_csv = "YesNo100-our.csv"

# === Load processor and model ===
print("🔄 Loading model and processor...")
try:
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )
    device = model.device
    print(f"✅ Model and processor loaded to device: {device}")
except Exception as e:
    print(f"❌ Error loading model or processor: {e}")
    exit()

# === Load ground truth CSV ===
if not os.path.exists(ground_truth_csv):
    print(f"❌ Ground truth CSV not found: {ground_truth_csv}")
    exit()
gt_df = pd.read_csv(ground_truth_csv, header=None, names=["image_name", "ground_truth_category"])
print(f"✅ Loaded {len(gt_df)} entries from CSV.")

# === Inference ===
predictions_data = []
correct_predictions = 0
total_images_processed = 0

print("\n🚀 Starting Yes/No prediction...")
for _, row in tqdm(gt_df.iterrows(), total=len(gt_df), desc="Processing"):
    image_name = row["image_name"]
    gt_category = str(row["ground_truth_category"]).strip().lower()
    image_path = os.path.join(test_images_dir, image_name)

    if not os.path.exists(image_path):
        predictions_data.append({
            "image_name": image_name,
            "ground_truth_category": gt_category,
            "predicted_response": "IMAGE_NOT_FOUND",
            "predicted_decision": "ERROR",
            "is_correct": False
        })
        continue

    try:
        image = Image.open(image_path).convert("RGB")

        question = f"Is the type of this book '{gt_category}'? Answer 'Yes' or 'No'."
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]

        text_for_processor = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = processor(
            text=text_for_processor,
            images=image,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)

        raw_pred = tokenizer.decode(output_ids[0], skip_special_tokens=True).lower().strip()

        # === Post-processing to extract meaningful answer ===
        cleaned_pred_lower = raw_pred.lower().strip()

        # Step 1: Try to match using assistant header pattern
        match = re.search(r"(?:user|system|assistant).*?assistant\s*\n\s*(.*)", cleaned_pred_lower, re.DOTALL)
        if match:
            extracted_answer = match.group(1).strip()
        else:
            # Step 2: Fallback if no assistant pattern found
            extracted_answer = cleaned_pred_lower.replace(question.lower(), "").strip()
            # Remove redundant text
            extracted_answer = re.sub(
                r"^(yes|no|system|you are a helpful assistant\.?).*?(\byes\b|\bno\b)",
                r"\2",
                extracted_answer,
                flags=re.IGNORECASE | re.DOTALL
            ).strip()
            extracted_answer = re.sub(
                r"^(yes|no|system|you are a helpful assistant\.?)\s*",
                "",
                extracted_answer,
                flags=re.IGNORECASE
            ).strip()

        # === Final decision: Yes / No / Unknown ===
        final_prediction_status = "UNKNOWN"
        if re.search(r"\byes\b", extracted_answer):
            final_prediction_status = "Yes"
        elif re.search(r"\bno\b", extracted_answer):
            final_prediction_status = "No"

        is_correct = (final_prediction_status == "Yes")

        # === Logging ===
        print(f"{image_name:<20} GT: {gt_category:<30} | Model Response: '{extracted_answer[:40]:<40}' | Predicted Decision: {final_prediction_status:<10} | Correct: {'✅' if is_correct else '❌'}")

        predictions_data.append({
            "image_name": image_name,
            "ground_truth_category": gt_category,
            "predicted_response": extracted_answer,
            "predicted_decision": final_prediction_status,
            "is_correct": is_correct
        })
        total_images_processed += 1
        if is_correct:
            correct_predictions += 1

    except Exception as e:
        print(f"❌ Error processing {image_name}: {e}")
        predictions_data.append({
            "image_name": image_name,
            "ground_truth_category": gt_category,
            "predicted_response": f"ERROR: {e}",
            "predicted_decision": "ERROR",
            "is_correct": False
        })

# === Save results ===
pd.DataFrame(predictions_data).to_csv(output_predictions_csv, index=False)
print(f"\n✅ Predictions saved to {output_predictions_csv}")

# === Accuracy summary ===
if total_images_processed > 0:
    accuracy = (correct_predictions / total_images_processed) * 100
    print(f"\n📊 Evaluation Summary:")
    print(f"Processed: {total_images_processed}")
    print(f"Correct: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
else:
    print("\n⚠️ No images were processed.")

print("🎉 Done.")
