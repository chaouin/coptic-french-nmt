import pandas as pd
from transformers import MarianTokenizer, MarianMTModel
from tqdm import tqdm

# === Paramètres globaux ===
n_samples = None

input_files = {
    "eval": "../../evaluation data/evaluation_data.csv",
}

model_paths = {
    "megalaa-finetune-fr-clean": "../../models/megalaa-finetuned-coptic-fr-clean-data",
    "megalaa-coptic-en": "megalaa/coptic-english-translator",
    "opus-en-fr": "Helsinki-NLP/opus-mt-en-fr",
    "opus-cop-fr": "Helsinki-NLP/opus-mt-tc-bible-big-mul-mul"
}

def generate_simple_translation_local(model_path, text):
    tokenizer = MarianTokenizer.from_pretrained(model_path, local_files_only=True)
    model = MarianMTModel.from_pretrained(model_path, local_files_only=True)
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
    output = model.generate(
        **inputs,
        max_length=128,
        num_beams=6,
        repetition_penalty=1.5,
        length_penalty=2.5
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def generate_two_step_translation(coptic_text, model_cop_en_path, model_en_fr_path):
    tokenizer_cop_en = MarianTokenizer.from_pretrained(model_cop_en_path)
    model_cop_en = MarianMTModel.from_pretrained(model_cop_en_path)
    inputs_cop_en = tokenizer_cop_en(coptic_text, return_tensors="pt", max_length=128, truncation=True)
    output_cop_en = model_cop_en.generate(**inputs_cop_en, max_length=128, num_beams=6)
    english_text = tokenizer_cop_en.decode(output_cop_en[0], skip_special_tokens=True)

    tokenizer_en_fr = MarianTokenizer.from_pretrained(model_en_fr_path)
    model_en_fr = MarianMTModel.from_pretrained(model_en_fr_path)
    inputs_en_fr = tokenizer_en_fr(english_text, return_tensors="pt", max_length=128, truncation=True)
    output_en_fr = model_en_fr.generate(**inputs_en_fr, max_length=128, num_beams=6)
    return tokenizer_en_fr.decode(output_en_fr[0], skip_special_tokens=True)

# === Boucle principale ===
for file_label, input_path in input_files.items():
    print(f"\n📥 Traitement du fichier : {file_label}")
    df = pd.read_csv(input_path).reset_index(drop=True)
    if n_samples:
        df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)

    # --- Megalaa finetune direct copte -> FR ---
    print("🚀 Megalaa finetune direct copte -> FR")
    translations = []
    for row in tqdm(df.itertuples(), total=len(df), desc="megalaa_finetune"):
        translation = generate_simple_translation_local(model_paths["megalaa-finetune-fr-clean"], row.coptic_text_romanized)
        translations.append(translation)
    df["generated_translation_megalaa_finetune"] = translations

    # --- Pipeline copte -> EN -> FR ---
    print("🚀 Pipeline copte -> EN -> FR")
    translations = []
    for row in tqdm(df.itertuples(), total=len(df), desc="pipeline_cop_en_fr"):
        translation = generate_two_step_translation(row.coptic_text_romanized,
                                                   model_paths["megalaa-coptic-en"],
                                                   model_paths["opus-en-fr"])
        translations.append(translation)
    df["generated_translation_pipeline"] = translations

    # --- Megalaa forcé en sortie FR ---
    print("🚀 Megalaa copte->EN forcé vers FR")
    translations = []
    tokenizer_force = MarianTokenizer.from_pretrained(model_paths["megalaa-coptic-en"])
    model_force = MarianMTModel.from_pretrained(model_paths["megalaa-coptic-en"])
    for row in tqdm(df.itertuples(), total=len(df), desc="megalaa_force_fr"):
        forced_input = ">>fra<< " + row.coptic_text_romanized
        inputs = tokenizer_force(forced_input, return_tensors="pt", max_length=128, truncation=True)
        output = model_force.generate(**inputs, max_length=128, num_beams=6)
        translation = tokenizer_force.decode(output[0], skip_special_tokens=True)
        translations.append(translation)
    df["generated_translation_force_fr"] = translations

    # --- Helsinki direct copte -> FR ---
    print("🚀 Helsinki direct copte -> FR")
    translations = []
    tokenizer_hel = MarianTokenizer.from_pretrained(model_paths["opus-cop-fr"])
    model_hel = MarianMTModel.from_pretrained(model_paths["opus-cop-fr"])
    for row in tqdm(df.itertuples(), total=len(df), desc="helsinki_cop_fr"):
        forced_input = ">>fra<< " + row.coptic_text_romanized
        inputs = tokenizer_hel(forced_input, return_tensors="pt", max_length=128, truncation=True)
        output = model_hel.generate(**inputs, max_length=128, num_beams=6)
        translation = tokenizer_hel.decode(output[0], skip_special_tokens=True)
        translations.append(translation)
    df["generated_translation_opus_cop_fr"] = translations

    # === Sauvegarde finale ===
    output_path = f"generated_translations_{file_label}_all_methods.csv"
    df.to_csv(output_path, index=False)
    print(f"✅i Fichier sauvegardé : {output_path}")