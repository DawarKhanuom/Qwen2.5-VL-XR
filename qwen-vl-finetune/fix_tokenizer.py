from transformers import AutoTokenizer

# === Paths ===
tokenizer_path = "./qwen2.5-prepped-tokenizer"  # ✅ Replace with the actual path if different

# === Load tokenizer ===
print("🔄 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

# === Define and add special tokens ===
print("🔧 Adding special tokens...")
special_tokens = {
    "additional_special_tokens": ["<image>"],
    "pad_token": "<|endoftext|>",  # For Qwen2.5, this is typically the same as eos_token
}

added = tokenizer.add_special_tokens(special_tokens)
print(f"✅ Added {added} new tokens.")

# === Save updated tokenizer ===
tokenizer.save_pretrained(tokenizer_path)
print(f"💾 Tokenizer with <image> saved to: {tokenizer_path}")
