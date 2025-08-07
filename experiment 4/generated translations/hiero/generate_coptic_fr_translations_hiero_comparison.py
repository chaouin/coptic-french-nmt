import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# === Global Parameters ===
BATCH_SIZE = 8
MAX_LENGTH = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Input files ===
input_files = {
    "clean": "../../../evaluation data/evaluation_data.csv",
    "noisy_10": "../../../evaluation data/evaluation_data_noisy_10.csv",
    "noisy_30": "../../../evaluation data/evaluation_data_noisy_30.csv",
    "noisy_50": "../../../evaluation data/evaluation_data_noisy_50.csv",
    "noisy_100": "../../../evaluation data/evaluation_data_noisy_100.csv"
}

# === Models to evaluate ===
model_paths = {
    "hiero_clean": "../../finetune/hiero-finetuned-coptic-fr-clean-data",
    "hiero_noisy_10": "../../finetune/hiero-finetuned-coptic-fr-noisy-10-data",
    "hiero_noisy_30": "../../finetune/hiero-finetuned-coptic-fr-noisy-30-data",
    "hiero_noisy_50": "../../finetune/hiero-finetuned-coptic-fr-noisy-50-data",
    "hiero_noisy_100": "../../finetune/hiero-finetuned-coptic-fr-noisy-100-data"
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

# === Process each input file ===
for file_label, input_path in input_files.items():
    print(f"\n📥 Processing input file: {file_label}")
    df = pd.read_csv(input_path).reset_index(drop=True)

    # Load and apply each model once per input file
    for model_label, model_path in model_paths.items():
        print(f"\n🚀 Model: {model_label}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True).to(DEVICE)
        model.eval()

        translations = []
        for i in tqdm(range(0, len(df), BATCH_SIZE), desc=f"{file_label}_{model_label}"):
            batch = df.iloc[i:i + BATCH_SIZE]["coptic_text_romanized"].tolist()
            batch_trans = generate_batch_translations(model, tokenizer, batch)
            translations.extend(batch_trans)

        df[f"generated_translation_exp4_hiero_{model_label}"] = translations

    # === Save output file for this input
    output_path = f"generated_translations_exp_4_hiero_{file_label}.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ File saved: {output_path}")
