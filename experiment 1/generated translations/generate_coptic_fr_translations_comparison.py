import pandas as pd
from tqdm import tqdm
from transformers import MarianTokenizer, MarianMTModel, AutoTokenizer, AutoModelForSeq2SeqLM

# === Global parameters ===
n_samples = None

input_files = {
    "eval": "../../evaluation data/evaluation_data.csv",
}

model_paths = {
    "megalaa-finetune-fr-clean": "../../models/megalaa-finetuned-coptic-fr-clean-data",
    "megalaa-coptic-en": "megalaa/coptic-english-translator",
    "t5-en-fr": "google-t5/t5-small",
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

def generate_en_fr_with_t5(english_text, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    input_text = f"translate English to French: {english_text}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
    outputs = model.generate(**inputs, max_length=128, num_beams=6)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_two_step_translation(coptic_text, model_cop_en_path, model_en_fr_path):
    tokenizer_cop_en = MarianTokenizer.from_pretrained(model_cop_en_path)
    model_cop_en = MarianMTModel.from_pretrained(model_cop_en_path)
    inputs_cop_en = tokenizer_cop_en(coptic_text, return_tensors="pt", max_length=128, truncation=True)
    output_cop_en = model_cop_en.generate(**inputs_cop_en, max_length=128, num_beams=6)
    english_text = tokenizer_cop_en.decode(output_cop_en[0], skip_special_tokens=True)

    return generate_en_fr_with_t5(english_text, model_en_fr_path)

# === Main loop ===
for file_label, input_path in input_files.items():
    print(f"\n📥 Processing file: {file_label}")
    df = pd.read_csv(input_path).reset_index(drop=True)
    if n_samples:
        df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)

    # --- Pipeline: Coptic -> EN -> FR (T5) ---
    print("🚀 Pipeline Coptic -> EN -> FR (T5)")
    translations = []
    for row in tqdm(df.itertuples(), total=len(df), desc="pipeline_cop_en_fr"):
        translation = generate_two_step_translation(
            row.coptic_text_romanized,
            model_paths["megalaa-coptic-en"],
            model_paths["t5-en-fr"]
        )
        translations.append(translation)
    df["generated_translation_pipeline"] = translations

    # --- Megalaa fine-tuned direct Coptic -> FR ---
    print("🚀 Megalaa fine-tuned direct Coptic -> FR")
    translations = []
    for row in tqdm(df.itertuples(), total=len(df), desc="megalaa_finetune"):
        translation = generate_simple_translation_local(model_paths["megalaa-finetune-fr-clean"], row.coptic_text_romanized)
        translations.append(translation)
    df["generated_translation_megalaa_finetune"] = translations

    # --- Megalaa forced output to FR ---
    print("🚀 Megalaa Coptic->EN forced to FR")
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

    # --- Helsinki direct Coptic -> FR ---
    print("🚀 Helsinki direct Coptic -> FR")
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

    # === Final save ===
    output_path = f"generated_translations_{file_label}_only_pipeline.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ File saved: {output_path}")
