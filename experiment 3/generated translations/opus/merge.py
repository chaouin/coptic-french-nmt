import pandas as pd
from pathlib import Path

# === Fichiers à fusionner
base_dir = Path(".")  # adapte selon l'emplacement réel
files = {
    "crampon": base_dir / "generated_translations_exp3_opus_crampon.csv",
    "darby": base_dir / "generated_translations_exp3_opus_darby.csv",
    "segond": base_dir / "generated_translations_exp3_opus_segond.csv",
}

# === Chargement de tous les fichiers
dfs = {}
for key, path in files.items():
    df = pd.read_csv(path)
    dfs[key] = df

# === Fusion progressive via verse_id
# On commence avec le fichier crampon
df_merged = dfs["crampon"]

# On ajoute ensuite darby et segond sans écraser les colonnes déjà présentes
for key in ["darby", "segond"]:
    cols_to_add = [col for col in dfs[key].columns if col.startswith("generated_translation_opus")]
    df_merged = df_merged.merge(
        dfs[key][["verse_id"] + cols_to_add],
        on="verse_id",
        how="left"
    )

# === Sauvegarde finale
output_file = "generated_translations_evaluation_data_all_models.csv"
df_merged.to_csv(output_file, index=False)
print(f"✅ Fichier fusionné sauvegardé sous : {output_file}")
