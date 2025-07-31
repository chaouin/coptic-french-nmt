import pandas as pd
import uroman as ur
from pathlib import Path

# === Fichiers d'entrée ===
input_files = [
    Path("../../evaluation data/evaluation_data_cop_en.csv"),
]

# === Initialiser uroman
uroman = ur.Uroman()

def preserve_brackets(text):
    return str(text).replace('[]', '<MISSING>')

for input_file in input_files:
    # Charger le CSV
    df = pd.read_csv(input_file)

    # Appliquer la romanisation
    df['coptic_text_romanized'] = df['coptic_text'].apply(
        lambda x: uroman.romanize_string(preserve_brackets(x)).replace('<MISSING>', '[]')
    )

    # Ne garder que les colonnes demandées
    df_final = df[['verse_id', 'coptic_text_romanized', 'english_translation']]

    # Construire le nom du fichier de sortie dans le dossier du script
    output_file = input_file.with_name(f"{input_file.stem}_romanized.csv")

    # Sauvegarder
    df_final.to_csv(output_file, index=False)

    print(f"✅ Romanisation terminée pour {input_file}")
    print(f"Fichier sauvegardé sous : {output_file}")
