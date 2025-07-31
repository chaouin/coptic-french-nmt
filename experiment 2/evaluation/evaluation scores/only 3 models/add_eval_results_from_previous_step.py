import pandas as pd

# === Fichiers
csv_source = "../../../experiment 1/evaluation scores/generated_translations_evaluation_data_all_methods_bleurt.csv"
csv_target = "generated_translations_evaluation_data_3_finetuned_models_bleurt.csv"
output_csv = "generated_translations_evaluation_data_all_finetuned_models_bleurt.csv"

# === Paires (colonne source → nouveau nom)
colonnes_a_ajouter = {
    "bleurt_crampon_generated_translation_megalaa_finetune": "bleurt_crampon_generated_translation_megalaa",
    "bleurt_segond_generated_translation_megalaa_finetune": "bleurt_segond_generated_translation_megalaa",
    "bleurt_darby_generated_translation_megalaa_finetune": "bleurt_darby_generated_translation_megalaa",
}

# === Chargement
df_source = pd.read_csv(csv_source)
df_target = pd.read_csv(csv_target)

# === Copie des colonnes avec renommage
for col_source, col_nouveau in colonnes_a_ajouter.items():
    if col_source not in df_source.columns:
        raise ValueError(f"❌ Colonne absente dans source : {col_source}")
    df_target[col_nouveau] = df_source[col_source]
    print(f"✅ Colonne copiée : {col_source} → {col_nouveau}")

# === Sauvegarde
df_target.to_csv(output_csv, index=False)
print(f"\n💾 Fichier final sauvegardé : {output_csv}")
