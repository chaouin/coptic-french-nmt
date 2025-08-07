import pandas as pd
from tqdm import tqdm
from transformers import MarianTokenizer, MarianMTModel, AutoTokenizer, AutoModelForSeq2SeqLM

# === Global Parameters ===
n_samples = None

input_files = {
    "evaluation_data": "../../evaluation data/evaluation_data.csv",
}

model_paths = {
    "opus": "../../models/opus-finetuned-coptic-fr-clean-data",
    "t5": "../../models/t5-finetuned-coptic-fr-clean-data",
    "hiero": "../../models/hiero-finetuned-coptic-fr-clean-data"
}

# === Load all models once into memory ===
print("📦 Loading models into memory...")
marian_models = {
    "opus": {
        "tokenizer": MarianTokenizer.from_pretrained(model_paths["opus"], local_files_only=True),
        "model": MarianMTModel.from_pretrained(model_paths["opus"], local_files_only=True)
    }
}

auto_models = {
    "t5": {
        "tokenizer": AutoTokenizer.from_pretrained(model_paths["t5"], local_files_only=True),
        "model": AutoModelForSeq2SeqLM.from_pretrained(model_paths["t5"], local_files_only=True)
    },
    "hiero": {
        "tokenizer": AutoTokenizer.from_pretrained(model_paths["hiero"], local_files_only=True),
        "model": AutoModelForSeq2SeqLM.from_pretrained(model_paths["hiero"], local_files_only=True)
    }
}

# === Unified translation generation function ===
def generate_translation(model, tokenizer, text, tgt_lang=None):
    if tgt_lang:
        tokenizer.tgt_lang = tgt_lang
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
    output = model.generate(
        **inputs,
        max_length=128,
        num_beams=6,
        repetition_penalty=1.5,
        length_penalty=2.5
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# === Main loop ===
for file_label, input_path in input_files.items():
    print(f"\n📥 Processing file: {file_label}")
    df = pd.read_csv(input_path).reset_index(drop=True)
    if n_samples:
        df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)

    # --- Opus finetune Coptic -> FR ---
    print("🚀 Opus finetune Coptic -> French")
    translations = []
    model = marian_models["opus"]["model"]
    tokenizer = marian_models["opus"]["tokenizer"]
    for row in tqdm(df.itertuples(), total=len(df), desc="opus_finetune"):
        coptic_input = ">>fra<< " + str(row.coptic_text_romanized)
        translation = generate_translation(model, tokenizer, coptic_input)
        translations.append(translation)
    df["generated_translation_opus"] = translations

    # --- T5 finetune Coptic -> FR ---
    print("🚀 T5 finetune Coptic -> French")
    translations = []
    model = auto_models["t5"]["model"]
    tokenizer = auto_models["t5"]["tokenizer"]
    for row in tqdm(df.itertuples(), total=len(df), desc="t5_finetune"):
        coptic_input = "translate Coptic to French: " + str(row.coptic_text_romanized)
        translation = generate_translation(model, tokenizer, coptic_input)
        translations.append(translation)
    df["generated_translation_t5"] = translations

    # --- Hiero finetune Coptic -> FR ---
    print("🚀 Hiero finetune Coptic -> French")
    translations = []
    model = auto_models["hiero"]["model"]
    tokenizer = auto_models["hiero"]["tokenizer"]
    for row in tqdm(df.itertuples(), total=len(df), desc="hiero_finetune"):
        translation = generate_translation(model, tokenizer, str(row.coptic_text_romanized), tgt_lang="fr")
        translations.append(translation)
    df["generated_translation_hiero"] = translations

    # === Final save ===
    output_path = f"generated_translations_{file_label}_all_finetuned_models.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ File saved: {output_path}")
