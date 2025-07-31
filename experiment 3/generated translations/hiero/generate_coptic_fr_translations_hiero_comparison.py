import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm
from pathlib import Path

# === Paramètres globaux ===
BATCH_SIZE = 8
MAX_LENGTH = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Fichier d’entrée
input_path = Path("../../../evaluation data/evaluation_data.csv")
df = pd.read_csv(input_path).reset_index(drop=True)

# Optionnel : pour debug ou tests
n_samples = None
if n_samples:
    df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)

# === Modèles à tester
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

# === Boucle sur chaque modèle
for label, path in model_paths.items():
    print(f"\n🚀 Traitement modèle : {label}")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True).to(DEVICE)
    model.eval()

    translations = []
    for i in tqdm(range(0, len(df), BATCH_SIZE), desc=f"{label}"):
        batch = df.iloc[i:i+BATCH_SIZE]["coptic_text_romanized"].tolist()
        batch_trans = generate_batch_translations(model, tokenizer, batch)
        translations.extend(batch_trans)

    df[f"generated_translation_{label}"] = translations

# === Sauvegarde
output_file = "generated_translations_exp_3_hiero.csv"
df.to_csv(output_file, index=False)
print(f"\n✅ Fichier sauvegardé : {output_file}")
