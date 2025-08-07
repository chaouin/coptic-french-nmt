from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import MarianTokenizer, MarianMTModel
from uroman import Uroman

# === Configuration ===
INPUT_CSV = "examples/test.csv"              # Input CSV file
OUTPUT_CSV = "examples/test_translated_with_helsinki.csv"  # Output CSV file
USE_UROMAN = True                     # Set to False if Coptic is already romanized
MODEL_PATH = "chaouin/coptic-french-translation-helsinki"  # Local or HF model path

# === Load input file ===
print(f"\n📥 Loading input file: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

# === Romanize if needed ===
if USE_UROMAN:
    if "coptic_text" not in df.columns:
        raise ValueError("Input CSV must contain a 'coptic_text' column when USE_UROMAN is True.")
    print("🔤 Applying uroman romanization to Coptic text...")

    uroman = Uroman()

    def preserve_brackets(text):
        return str(text).replace("[]", "<MISSING>")

    df["coptic_text_romanized"] = [
        uroman.romanize_string(preserve_brackets(text)).replace("<MISSING>", "[]")
        for text in df["coptic_text"].tolist()
    ]

elif "coptic_text_romanized" not in df.columns:
    raise ValueError("Input CSV must contain a 'coptic_text_romanized' column if USE_UROMAN is False.")

texts = df["coptic_text_romanized"].tolist()

# === Load model from local or Hugging Face ===
print(f"\n🚀 Loading MarianMT model from: {MODEL_PATH}")
tokenizer = MarianTokenizer.from_pretrained(MODEL_PATH, local_files_only=False)
model = MarianMTModel.from_pretrained(MODEL_PATH, local_files_only=False).to("cpu")

# === Translation generation ===
def generate_batch_translations(model, tokenizer, texts):
    translations = []
    for text in tqdm(texts, desc="Generating translations"):
        input_text = ">>fra<< " + text  # ensure target language is French
        inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=128,
                num_beams=6,
                repetition_penalty=1.5,
                length_penalty=2.5
            )
        translations.append(tokenizer.decode(output[0], skip_special_tokens=True))
    return translations

df["generated_translation"] = generate_batch_translations(model, tokenizer, texts)

# === Save to output file ===
output_path = Path(OUTPUT_CSV)
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)
print(f"\n✅ Translations saved to: {OUTPUT_CSV}")