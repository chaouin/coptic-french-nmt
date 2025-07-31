import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import torch

# === Paramètres globaux ===
n_samples = None

input_files = {
    "evaluation_data": "../data/evaluation_data_en_fr.csv"
}

# === Charger le modèle T5 depuis Hugging Face ===
print("🚀 Chargement du modèle google-t5/t5-small")
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small").eval()

# === Fonction de génération ===
def generate_translation(text):
    # pour T5 il faut explicitement donner la tâche
    input_text = f"translate English to French: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, num_beams=5, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# === Boucle principale ===
for file_label, input_path in input_files.items():
    print(f"\n📥 Traitement du fichier : {file_label}")
    df = pd.read_csv(input_path).reset_index(drop=True)
    if n_samples:
        df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)

    # Générer les traductions de english_translation → français
    translations = []
    for row in tqdm(df.itertuples(), total=len(df), desc="Traduction EN→FR"):
        translation = generate_translation(row.english_text)
        translations.append(translation)
    df["generated_translation_fr"] = translations

    # === Sauvegarde finale ===
    output_path = f"generated_translations_t5_baseline.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Fichier sauvegardé : {output_path}")
