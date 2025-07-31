import argparse
import pandas as pd
import uroman as ur
import os

# === MAIN ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Romaniser la colonne 'coptic_text' d'un corpus Copte–Français.")
    parser.add_argument("--input", required=True, help="Fichier CSV du corpus à romaniser")
    parser.add_argument("--version", required=True, help="Nom de la version cible (ex: darby, crampon, segond)")
    parser.add_argument("--suffix", default="romanized", help="Suffixe pour le fichier de sortie (ex: romanized)")

    args = parser.parse_args()

    # === Charger le fichier CSV
    df = pd.read_csv(args.input)

    # === Initialiser uroman
    uroman = ur.Uroman()

    def preserve_brackets(text):
        return str(text).replace('[]', '<MISSING>')

    # === Appliquer la romanisation
    df['coptic_text_romanized'] = df['coptic_text'].apply(
        lambda x: uroman.romanize_string(preserve_brackets(x)).replace('<MISSING>', '[]')
    )

    # === Ne garder que les colonnes demandées
    df_final = df[['verse_id', 'coptic_text_romanized', 'french_translation']]

    # === Déterminer le chemin de sauvegarde
    output_folder = os.path.join(os.path.dirname(args.input), '..', args.version)
    output_folder = os.path.normpath(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Construire le nom du fichier de sortie en ajoutant le suffixe avant ".csv"
    input_basename = os.path.basename(args.input)
    input_basename_no_ext = os.path.splitext(input_basename)[0]

    output_file = os.path.join(output_folder, f"{input_basename_no_ext}_{args.suffix}.csv")

    # === Sauvegarder
    df_final.to_csv(output_file, index=False)

    print("=== Romanisation terminée ===")
    print(f"Fichier sauvegardé sous : {output_file}")
    print("Colonnes conservées : verse_id, coptic_text_romanized, french_translation")

# ===== EXEMPLE LANCER LE SCRIPT ======
# python3 romanization.py --input crampon/coptic_parallel_corpus_fr_crampon_clean.csv --version crampon

