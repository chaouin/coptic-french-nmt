import pandas as pd
from tqdm import tqdm
from transformers import MarianTokenizer, MarianMTModel

# === Global parameters ===
n_samples = None

input_files = {
    "evaluation_data": "../data/evaluation_data_cop_en_romanized.csv",
}

# === Load model from Hugging Face ===
print("🚀 Loading model from Hugging Face: megalaa/coptic-english-translator")
tokenizer = MarianTokenizer.from_pretrained("megalaa/coptic-english-translator")
model = MarianMTModel.from_pretrained("megalaa/coptic-english-translator")

# === Translation function ===
def generate_translation(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# === Main loop ===
for file_label, input_path in input_files.items():
    print(f"\n📥 Processing file: {file_label}")
    df = pd.read_csv(input_path).reset_index(drop=True)
    if n_samples:
        df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)

    # Generate translations
    translations = []
    for row in tqdm(df.itertuples(), total=len(df), desc="Translating"):
        translation = generate_translation(row.coptic_text_romanized)
        translations.append(translation)
    df["english_generated_translation"] = translations

    # === Save results ===
    output_path = f"generated_translations_megalaa_baseline.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ File saved: {output_path}")
