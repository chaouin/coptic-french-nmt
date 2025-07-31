import pandas as pd
from transformers import MarianTokenizer, MarianMTModel
from tqdm import tqdm
import torch
from concurrent.futures import ProcessPoolExecutor
import os

# === Paramètres globaux ===
n_samples = None
input_file = "../../evaluation data/evaluation_data.csv"
df = pd.read_csv(input_file).reset_index(drop=True)

if n_samples:
    df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)

model_paths = {
    "opus_all": "../../models/opus-finetuned-coptic-fr-clean-data",
    "opus_crampon": "../../models/opus-finetuned-coptic-fr-crampon-data",
    "opus_darby": "../../models/opus-finetuned-coptic-fr-darby-data",
    "opus_segond": "../../models/opus-finetuned-coptic-fr-segond-data"
}

PREVIOUS_EXP_PATH = "../../experiment 2/generated translations/generated_translations_evaluation_data_all_finetuned_models.csv"
PREVIOUS_COL = "generated_translation_opus"
NEW_COL_NAME = "generated_translation_opus_all"

print("📥 Chargement des traductions existantes pour opus_all...")
df_prev = pd.read_csv(PREVIOUS_EXP_PATH)
if PREVIOUS_COL not in df_prev.columns:
    raise ValueError(f"Colonne '{PREVIOUS_COL}' introuvable dans {PREVIOUS_EXP_PATH}")
df[NEW_COL_NAME] = df_prev[PREVIOUS_COL]
print("✅ Traductions opus_all récupérées avec succès sous le nom :", NEW_COL_NAME)


def generate_for_model(model_key, model_path, df_input_path):
    import pandas as pd
    from transformers import MarianTokenizer, MarianMTModel
    import torch
    from tqdm import tqdm

    df_input = pd.read_csv(df_input_path)
    print(f"🚀 [{os.getpid()}] Génération pour : {model_key}")

    tokenizer = MarianTokenizer.from_pretrained(model_path, local_files_only=True)
    model = MarianMTModel.from_pretrained(model_path, local_files_only=True).to("cpu")

    translations = []
    for row in tqdm(df_input.itertuples(), total=len(df_input), desc=model_key):
        text = ">>fra<< " + row.coptic_text_romanized
        inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        with torch.no_grad():
            output = model.generate(**inputs, max_length=128, num_beams=6,
                                    repetition_penalty=1.5, length_penalty=2.5)
        translation = tokenizer.decode(output[0], skip_special_tokens=True)
        translations.append(translation)

    df_input[f"generated_translation_{model_key}"] = translations
    output_path = f"generated_translations_exp3_{model_key}.csv"
    df_input.to_csv(output_path, index=False)
    print(f"✅ Fichier sauvegardé : {output_path}")


# Écriture du fichier temporaire partagé pour tous les processus
TMP_INPUT_PATH = "df_shared_input.csv"
df.to_csv(TMP_INPUT_PATH, index=False)

# === Lancement parallèle
if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = []
        for model_key, model_path in model_paths.items():
            if model_key == "opus_all":
                continue
            futures.append(executor.submit(generate_for_model, model_key, model_path, TMP_INPUT_PATH))

        for future in futures:
            future.result()

    print("\n✅ Toutes les générations sont terminées.")
