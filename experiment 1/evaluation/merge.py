import pandas as pd

# === Paramètres pour BLEURT ===
bleurt_all_path = "evaluation scores/generated_translations_evaluation_data_all_methods_bleurt.csv"
bleurt_pipeline_path = "evaluation scores/generated_translations_evaluation_data_all_methods_pipeline_bleurt.csv"
bleurt_output_path = "evaluation scores/generated_translations_evaluation_data_all_methods_bleurt.csv"

bleurt_columns_to_replace = [
    "bleurt_crampon_generated_translation_pipeline",
    "bleurt_segond_generated_translation_pipeline",
    "bleurt_darby_generated_translation_pipeline"
]

# === Paramètres pour OTHER SCORES ===
other_all_path = "evaluation scores/generated_translations_evaluation_data_all_methods_other_scores.csv"
other_pipeline_path = "evaluation scores/generated_translations_evaluation_data_all_methods_pipeline_other_scores.csv"
other_output_path = "evaluation scores/generated_translations_evaluation_data_all_methods_other_scores.csv"

other_columns_to_replace = [
    "meteor_crampon_generated_translation_pipeline",
    "bertscore_crampon_generated_translation_pipeline",
    "comet_crampon_generated_translation_pipeline",
    "meteor_segond_generated_translation_pipeline",
    "bertscore_segond_generated_translation_pipeline",
    "comet_segond_generated_translation_pipeline",
    "meteor_darby_generated_translation_pipeline",
    "bertscore_darby_generated_translation_pipeline",
    "comet_darby_generated_translation_pipeline"
]

# === Fonction générique de remplacement ===
def replace_columns(base_file, update_file, columns, output_file):
    print(f"\n📥 Chargement du fichier complet : {base_file}")
    df_all = pd.read_csv(base_file)

    print(f"📥 Chargement du fichier pipeline : {update_file}")
    df_update = pd.read_csv(update_file)

    if len(df_all) != len(df_update):
        raise ValueError("❌ Les fichiers n'ont pas le même nombre de lignes.")

    print("🔁 Remplacement des colonnes suivantes :")
    for col in columns:
        if col not in df_update.columns:
            raise ValueError(f"❌ Colonne manquante dans le fichier pipeline : {col}")
        print(f"   - {col}")
        df_all[col] = df_update[col]

    df_all.to_csv(output_file, index=False)
    print(f"✅ Sauvegarde terminée : {output_file}")

# === Exécution ===
replace_columns(bleurt_all_path, bleurt_pipeline_path, bleurt_columns_to_replace, bleurt_output_path)
replace_columns(other_all_path, other_pipeline_path, other_columns_to_replace, other_output_path)
