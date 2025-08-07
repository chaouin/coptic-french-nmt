from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# === Global parameters ===
BATCH_SIZE = 8
MAX_LENGTH = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Input file
input_path = Path("../../../evaluation data/evaluation_data.csv")
df = pd.read_csv(input_path).reset_index(drop=True)

# Optional: for debugging or testing
n_samples = None
if n_samples:
    df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)

# === Models to evaluate
model_paths = {
    "hiero_all": "../../finetune/hiero-finetuned-coptic-fr-clean-data",
    "hiero_crampon": "../../finetune/hiero-finetuned-coptic-fr-crampon-data",
    "hiero_darby": "../../finetune/hiero-finetuned-coptic-fr-darby-data",
    "hiero_segond": "../../finetune/hiero-finetuned-coptic-fr-segond-data"
}

def generate_batch_translations(model, tokenizer, texts):
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True,
        max_length=MAX_LENGTH
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=MAX_LENGTH,
            num_beams=4
        )

    return [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]

# === Loop over each model
for label, path in model_paths.items():
    print(f"\n🚀 Processing model: {label}")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True).to(DEVICE)
    model.eval()

    translations = []
    for i in tqdm(range(0, len(df), BATCH_SIZE), desc=f"{label}"):
        batch = df.iloc[i:i+BATCH_SIZE]["coptic_text_romanized"].tolist()
        batch_trans = generate_batch_translations(model, tokenizer, batch)
        translations.extend(batch_trans)

    df[f"generated_translation_{label}"] = translations

# === Save output
output_file = "generated_translations_exp_3_hiero.csv"
df.to_csv(output_file, index=False)
print(f"\n✅ File saved: {output_file}")
