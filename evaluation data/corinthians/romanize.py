import pandas as pd
import uroman as ur

# === Paramètres ===
INPUT_FILE = "1corinthians_coptic_3versions.csv"
OUTPUT_FILE = "1corinthians_coptic_3versions_romanized.csv"

# === Charger le CSV
df = pd.read_csv(INPUT_FILE)

# === Initialiser Uroman
uroman = ur.Uroman()

# === Romaniser et remplacer la colonne
df["coptic_text_romanized"] = df["coptic_text"].apply(lambda x: uroman.romanize_string(str(x)))
df.drop(columns=["coptic_text"], inplace=True)

# === Réorganiser les colonnes comme demandé
df = df[["verse_id", "coptic_text_romanized", "french_segond", "french_darby", "french_crampon"]]

# === Sauvegarder
df.to_csv(OUTPUT_FILE, index=False)

print("✅ Romanisation terminée.")
print(f"Fichier sauvegardé sous : {OUTPUT_FILE}")
