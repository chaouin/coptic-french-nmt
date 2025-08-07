import pandas as pd

# === Fichiers ===
original_file = "generated_translations_evaluation_data_all_methods_old_pip.csv"
pipeline_file = "generated_translations_eval_only_pipeline.csv"
output_file = "generated_translations_evaluation_data_all_methods.csv"

# === Chargement des fichiers ===
print("📥 Chargement du fichier original...")
df_all = pd.read_csv(original_file)

print("📥 Chargement du fichier avec pipeline mis à jour...")
df_pipeline = pd.read_csv(pipeline_file)

# === Remplacement de la colonne ===
if "generated_translation_pipeline" not in df_pipeline.columns:
    raise ValueError("La colonne 'generated_translation_pipeline' est absente du fichier pipeline.")
if len(df_all) != len(df_pipeline):
    raise ValueError("Les deux fichiers n'ont pas le même nombre de lignes.")

print("🔁 Remplacement de la colonne 'generated_translation_pipeline'...")
df_all["generated_translation_pipeline"] = df_pipeline["generated_translation_pipeline"]

# === Sauvegarde du fichier ===
df_all.to_csv(output_file, index=False)
print(f"✅ Colonne remplacée. Fichier sauvegardé sous : {output_file}")
