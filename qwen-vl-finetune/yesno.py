import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
import os
import pandas as pd
from tqdm import tqdm # For progress bar
import re # For regular expressions in post-processing

# === Configuration ===
# Path to your fine-tuned model checkpoints 

model_path = "/ibex/scratch/khand0b/qwen/Qwen2.5-VL-3B-Instruct"

#model_path = "./checkpoints3wen50F" # <<< VERIFY THIS PATH
#model_path = "./checkpoints3wen50k" # <<< VERIFY THIS PATH
# Path to the directory containing the working tokenizer and processor
tokenizer_processor_path = "./qwen2.5-prepped-tokenizer" # <<< VERIFY THIS PATH

# Directory where your test images are located
test_images_dir = "/ibex/user/khand0b/qwen/data/test_dk/" # <<< VERIFY THIS PATH
# Path to your ground truth CSV file
# ground_truth_csv = "correct.csv" # <<< VERIFY THIS PATH
ground_truth_csv = "test_dk.csv" # <<< VERIFY THIS PATH

# Output file for predictions
output_predictions_csv = "YesNo100-our.csv"

# === Load tokenizer, processor, and model ===
print("🔄 Loading tokenizer and processor...")
try:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_processor_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(tokenizer_processor_path, trust_remote_code=True)
    assert "<image>" in tokenizer.get_vocab(), "❌ ERROR: <image> token not found in tokenizer! Ensure you are loading the correct Qwen-VL tokenizer."
    print("✅ Tokenizer and processor loaded.")
except Exception as e:
    print(f"❌ Error loading tokenizer or processor from '{tokenizer_processor_path}': {e}")
    exit()

print("🔄 Loading model...")
try:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16, # Use torch.bfloat16 if your training used bf16
        device_map="auto", # Use "cuda" if you have only one GPU and want to force it
        trust_remote_code=True,
        local_files_only=True # Crucial for loading from a local directory
    )
    device = model.device # Capture the actual device the model loaded onto
    print(f"✅ Model loaded successfully to device: {device}")
except Exception as e:
    print(f"❌ Error loading model from '{model_path}': {e}")
    exit()

# === Load ground truth data ===
print(f"📄 Loading ground truth from {ground_truth_csv}...")
if not os.path.exists(ground_truth_csv):
    print(f"❌ Error: Ground truth CSV not found at '{ground_truth_csv}'.")
    exit()
try:
    gt_df = pd.read_csv(ground_truth_csv, header=None, names=['image_name', 'ground_truth_category'])
    print(f"✅ Loaded {len(gt_df)} entries from CSV.")
except Exception as e:
    print(f"❌ Error reading CSV file: {e}")
    exit()

# === Prepare for predictions ===
predictions_data = []
correct_predictions = 0
total_images_processed = 0 # This correctly excludes IMAGE_NOT_FOUND cases

print("\n🚀 Starting inference on test images with 'Yes/No' questions...")
for index, row in tqdm(gt_df.iterrows(), total=len(gt_df), desc="Processing Images"):
    image_name = row['image_name']
    ground_truth_category = str(row['ground_truth_category']).strip().lower()

    full_image_path = os.path.join(test_images_dir, image_name)

    if not os.path.exists(full_image_path):
        predictions_data.append({
            'image_name': image_name,
            'ground_truth_category': ground_truth_category,
            'predicted_response': "IMAGE_NOT_FOUND", # Renamed column for clarity in this mode
            'is_correct': False
        })
        continue

    try:
        image = Image.open(full_image_path).convert("RGB")

        # === Question Formulation for NLI-style ===
        # The core idea: Ask if the image *is* the ground truth category.
        question_for_model = f"Is the type of this book '{ground_truth_category}'? Answer 'Yes' or 'No'."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question_for_model}
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
            # Keep max_new_tokens very low as we expect 'Yes' or 'No' (plus minimal preamble)
            output_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)
            raw_pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # === Simplified Post-processing for 'Yes/No' ===
        cleaned_pred_lower = raw_pred.lower().strip()

        # Step 1: Remove common preambles (including the question if echoed)
        # This regex looks for the pattern up to 'assistant\n' and captures what follows.
        # It's made more flexible to catch variations.
        match = re.search(r"(?:user|system|assistant).*?assistant\s*\n\s*(.*)", cleaned_pred_lower, re.DOTALL)
        if match:
            extracted_answer = match.group(1).strip()
        else:
            # If the specific chat template pattern isn't found, try to clean more generally
            # Remove direct question echo if it happens
            extracted_answer = cleaned_pred_lower.replace(question_for_model.lower(), "").strip()
            # Remove simple preamble remnants
            extracted_answer = re.sub(r"^(yes|no|system|you are a helpful assistant\.?).*?(\s*yes|\s*no)", r"\2", extracted_answer, flags=re.IGNORECASE | re.DOTALL).strip()
            extracted_answer = re.sub(r"^(yes|no|system|you are a helpful assistant\.?)\s*", "", extracted_answer, flags=re.IGNORECASE).strip()


        # Now, analyze the 'extracted_answer' to determine 'Yes'/'No'
        final_prediction_status = "UNKNOWN"
        # Check for 'yes' or 'no' as a prominent part of the response
        if re.search(r'\byes\b', extracted_answer):
            final_prediction_status = "Yes"
        elif re.search(r'\bno\b', extracted_answer):
            final_prediction_status = "No"
        else:
            # If no clear 'yes' or 'no' word, categorize as UNKNOWN
            final_prediction_status = "UNKNOWN"
            # Optional: for debugging, you might want to see the raw extracted answer
            # print(f"DEBUG: Could not parse Yes/No from: '{extracted_answer}' for {image_name}")

        # === Determine if prediction is 'correct' ===
        # In this NLI-style task, 'is_correct' means the model positively confirmed the GT category.
        is_correct = (final_prediction_status == "Yes")

        # Log to console
        print(f"{image_name:<20} GT: {ground_truth_category:<30} | Model Response: '{extracted_answer[:40]:<40}' | Predicted Decision: {final_prediction_status:<10} | Correct: {'✅' if is_correct else '❌'}")


        predictions_data.append({
            'image_name': image_name,
            'ground_truth_category': ground_truth_category,
            'predicted_response': extracted_answer, # Store the full extracted response for debugging
            'predicted_decision': final_prediction_status, # Store the 'Yes'/'No'/'UNKNOWN' decision
            'is_correct': is_correct
        })
        total_images_processed += 1 # Only increments for successfully processed images
        if is_correct:
            correct_predictions += 1

    except Exception as e:
        print(f"\n❌ Error processing image {image_name}: {e}")
        predictions_data.append({
            'image_name': image_name,
            'ground_truth_category': ground_truth_category,
            'predicted_response': f"ERROR: {str(e)}",
            'predicted_decision': "ERROR",
            'is_correct': False
        })
        continue

# === Save and Report Results ===
output_df = pd.DataFrame(predictions_data)
output_df.to_csv(output_predictions_csv, index=False)
print(f"\n✅ Predictions saved to {output_predictions_csv}")

if total_images_processed > 0:
    accuracy = (correct_predictions / total_images_processed) * 100
    print(f"\n--- Evaluation Summary ---")
    print(f"Total entries in input CSV: {len(gt_df)}")
    print(f"Images processed for inference (excluding 'IMAGE_NOT_FOUND'): {total_images_processed}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("------------------------")
else:
    print("\nNo images were successfully processed for inference.")

print("\nScript finished.")
