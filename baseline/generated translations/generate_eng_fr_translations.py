import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# === Global parameters ===
n_samples = None

input_files = {
    "evaluation_data": "../data/evaluation_data_en_fr.csv"
}

# === Load T5 model from Hugging Face ===
print("🚀 Loading model: google-t5/t5-small")
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small").eval()

# === Translation function ===
def generate_translation(text):
    # For T5, the task must be explicitly specified
    input_text = f"translate English to French: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, num_beams=5, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# === Main loop ===
for file_label, input_path in input_files.items():
    print(f"\n📥 Processing file: {file_label}")
    df = pd.read_csv(input_path).reset_index(drop=True)
    if n_samples:
        df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)

    # Generate translations from English to French
    translations = []
    for row in tqdm(df.itertuples(), total=len(df), desc="Translating EN→FR"):
        translation = generate_translation(row.english_text)
        translations.append(translation)
    df["generated_translation_fr"] = translations

    # === Save results ===
    output_path = f"generated_translations_t5_baseline.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ File saved: {output_path}")
