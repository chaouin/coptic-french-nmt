import pandas as pd
from transformers import MarianTokenizer, MarianMTModel
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import torch
import os

# --- Paramètres Globaux ---
n_samples = None  # Laisse à None pour traiter tous les échantillons, ou définis un entier (ex: 100)

# --- Fichiers d'Entrée des Données d'Évaluation ---
# Liste de tous les fichiers de données d'évaluation. Chacun donnera lieu à un fichier de sortie distinct.
INPUT_EVAL_FILES = [
    "../../evaluation data/evaluation_data.csv",
    "../../evaluation data/evaluation_data_noisy_10.csv",
    "../../evaluation data/evaluation_data_noisy_30.csv",
    "../../evaluation data/evaluation_data_noisy_50.csv",
    "../../evaluation data/evaluation_data_noisy_100.csv"
]

# --- Définition des Modèles à Utiliser pour la Génération de Traductions ---
model_paths = {
    "opus_clean": "../../models/opus-finetuned-coptic-fr-clean-data",
    "opus_noisy_10": "../../models/opus-finetuned-coptic-fr-noisy-10-data",
    "opus_noisy_30": "../../models/opus-finetuned-coptic-fr-noisy-30-data",
    "opus_noisy_50": "../../models/opus-finetuned-coptic-fr-noisy-50-data",
    "opus_noisy_100": "../../models/opus-finetuned-coptic-fr-noisy-100-data",
}


# --- Fonction pour générer des traductions pour un seul modèle et un DataFrame ---
# Cette fonction sera exécutée dans des processus parallèles.
def generate_translations_for_model(model_key, model_path, df_input, temp_output_path):
    # Ré-importer à l'intérieur de la fonction pour la robustesse du multiprocessing
    import pandas as pd
    from transformers import MarianTokenizer, MarianMTModel
    import torch
    from tqdm import tqdm

    print(f"🚀 [{os.getpid()}] Démarrage de la traduction pour le modèle : {model_key}")

    # Charger le tokenizer et le modèle
    tokenizer = MarianTokenizer.from_pretrained(model_path, local_files_only=True)
    # Assure que le modèle utilise le CPU si aucun GPU n'est disponible ou souhaité
    model = MarianMTModel.from_pretrained(model_path, local_files_only=True).to("cpu")

    translations = []
    # Parcourt la colonne 'coptic_text_romanized' pour la traduction
    for text_to_translate in tqdm(df_input["coptic_text_romanized"], total=len(df_input),
                                  desc=f"Traduction avec {model_key}"):
        text = ">>fra<< " + str(text_to_translate)  # Assure que le texte est une chaîne
        inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
        inputs = {k: v.to("cpu") for k, v in inputs.items()}  # Déplace les entrées vers le CPU
        with torch.no_grad():
            output = model.generate(**inputs, max_length=128, num_beams=6,
                                    repetition_penalty=1.5, length_penalty=2.5)
        translation = tokenizer.decode(output[0], skip_special_tokens=True)
        translations.append(translation)

    # Crée un DataFrame temporaire avec juste verse_id et la nouvelle colonne de traduction
    df_temp_result = pd.DataFrame({
        "verse_id": df_input["verse_id"],
        f"generated_translation_{model_key}": translations
    })

    # Sauvegarde les résultats temporaires pour ce modèle sur ce jeu de données
    df_temp_result.to_csv(temp_output_path, index=False)
    print(f"✅ [{os.getpid()}] Traductions pour {model_key} sauvegardées dans {temp_output_path}")
    return temp_output_path  # Retourne le chemin pour confirmation


# --- Boucle Principale de Traitement pour Chaque Fichier de Données d'Évaluation ---
if __name__ == "__main__":

    # Crée un répertoire pour les sorties s'il n'existe pas
    output_dir = "generated_translations_per_dataset_all_models"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 Les fichiers de sortie seront sauvegardés dans : {output_dir}")

    for input_eval_file_path in INPUT_EVAL_FILES:
        print(f"\n--- Traitement du jeu de données d'évaluation : {input_eval_file_path} ---")

        # Extrait le nom de base pour le fichier de sortie
        base_name = os.path.basename(input_eval_file_path).replace(".csv", "")
        output_file_name = os.path.join(output_dir, f"{base_name}_all_models_generated_translations.csv")

        # Charge le jeu de données d'évaluation actuel
        if not os.path.exists(input_eval_file_path):
            print(
                f"⚠️ Erreur : Fichier d'entrée d'évaluation introuvable : {input_eval_file_path}. Ce jeu de données sera ignoré.")
            continue

        current_df = pd.read_csv(input_eval_file_path)
        if n_samples:
            current_df = current_df.sample(n=n_samples, random_state=42).reset_index(drop=True)

        # Crée un répertoire temporaire pour les sorties des processus parallèles pour CE jeu de données
        temp_dir_for_dataset = os.path.join(output_dir, f"temp_{base_name}_translations")
        os.makedirs(temp_dir_for_dataset, exist_ok=True)

        # Prépare le DataFrame à partager avec les processus parallèles
        # Partage uniquement les colonnes nécessaires (verse_id et coptic_text_romanized)
        df_for_parallel = current_df[["verse_id", "coptic_text_romanized"]].copy()

        temp_output_paths_for_dataset = []
        with ProcessPoolExecutor(max_workers=3) as executor:  # Ajuste max_workers selon tes ressources CPU
            futures = []
            for model_key, model_path in model_paths.items():
                # Définit un chemin de sortie temporaire unique pour chaque modèle pour CE jeu de données spécifique
                temp_out_path_model = os.path.join(temp_dir_for_dataset, f"tmp_{base_name}_{model_key}.csv")
                futures.append(executor.submit(
                    generate_translations_for_model,
                    model_key, model_path, df_for_parallel, temp_out_path_model
                ))
                temp_output_paths_for_dataset.append(temp_out_path_model)

            # Attendre que toutes les tâches parallèles pour CE jeu de données soient terminées
            for future in tqdm(futures, desc=f"Attente des traductions parallèles pour {base_name}"):
                future.result()  # Cela lèvera les exceptions si elles se sont produites dans le sous-processus

        # --- Fusionne les résultats pour le jeu de données actuel ---
        # Le current_df original contient déjà verse_id, coptic_text_romanized, et les références françaises originales.

        # Itère sur les fichiers de sortie temporaires pour CE jeu de données et les fusionne
        for model_key in model_paths.keys():
            tmp_path = os.path.join(temp_dir_for_dataset, f"tmp_{base_name}_{model_key}.csv")
            if os.path.exists(tmp_path):
                df_temp_model_output = pd.read_csv(tmp_path)
                # Assure que la fusion est effectuée sur 'verse_id' et conserve toutes les colonnes de current_df
                current_df = pd.merge(current_df, df_temp_model_output, on="verse_id", how="left")
            else:
                print(
                    f"⚠️ Avertissement : Fichier temporaire {tmp_path} introuvable pour la fusion de {model_key} pour {base_name}.")

        # Nettoie les fichiers temporaires pour ce jeu de données
        for temp_file in temp_output_paths_for_dataset:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        if os.path.exists(temp_dir_for_dataset):
            os.rmdir(temp_dir_for_dataset)

        # --- Sauvegarde le fichier final fusionné pour CE jeu de données spécifique ---
        current_df.to_csv(output_file_name, index=False)
        print(f"✅ Traductions finales pour {base_name} sauvegardées dans : {output_file_name}")

    print(
        "\n✨ Tous les jeux de données d'évaluation ont été traités et les résultats sauvegardés dans des fichiers individuels.")