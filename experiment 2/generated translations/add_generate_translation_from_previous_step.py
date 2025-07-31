import pandas as pd

# === Fichiers
csv_source = "../../experiment 1/generated translations/generated_translations_evaluation_data_all_methods.csv"
csv_target = "generated_translations_evaluation_data_3_finetuned_models.csv"

# === Paramètres
colonne_a_copier = "generated_translation_megalaa_finetune"
nouveau_nom = "generated_translation_megalaa"

# === Chargement des fichiers
df_source = pd.read_csv(csv_source)
df_target = pd.read_csv(csv_target)

# === Vérification existence
if colonne_a_copier not in df_source.columns:
    raise ValueError(f"La colonne '{colonne_a_copier}' est absente du fichier source.")

# === Ajout de la colonne renommée
df_target[nouveau_nom] = df_source[colonne_a_copier]

# === Sauvegarde du nouveau fichier
df_target.to_csv("generated_translations_evaluation_data_all_finetuned_models.csv", index=False)
print(f"✅ Colonne copiée sous le nom '{nouveau_nom}' et ajoutée au fichier cible.")
