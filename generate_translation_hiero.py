from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from uroman import Uroman

# === Configuration ===
INPUT_CSV = "examples/test.csv"              # Input CSV file
OUTPUT_CSV = "examples/test_translated_hiero.csv"  # Output CSV file
USE_UROMAN = True                     # Set to False if Coptic is already romanized

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

# === Load model from Hugging Face ===
model_name = "chaouin/coptic-french-translation-hiero"
print(f"\n🚀 Loading model from Hugging Face: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# === Translation generation ===
def generate_batch_translations(model, tokenizer, texts):
    translations = []
    for text in tqdm(texts, desc="Generating translations"):
        inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
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