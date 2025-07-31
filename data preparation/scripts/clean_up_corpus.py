import argparse
import pandas as pd
import re
import os

# === MAIN ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nettoyer un corpus Copte–Français en supprimant les versets invalides.")
    parser.add_argument("--input", required=True, help="Fichier CSV du corpus à nettoyer")
    parser.add_argument("--version", required=True, help="Nom de la version cible (ex: darby, crampon, segond)")

    args = parser.parse_args()

    # === Charger le fichier CSV
    df = pd.read_csv(args.input)

    # === Nettoyer les versets où le texte copte est invalide
    mask_coptic_invalid = df['coptic_text'].str.strip().eq('[...]')
    mask_french_missing = df['french_translation'].isnull() | df['french_translation'].str.strip().eq('')
    mask_removed = mask_coptic_invalid | mask_french_missing

    df_clean = df[~mask_removed].copy()

    # === Nettoyer les annotations bibliques en début de phrase (regex)
    df_clean['french_translation'] = df_clean['french_translation'].str.replace(
        r'^\(\s*[\d.:]+\s*\)\s*', '', regex=True
    )

    # === Déterminer le chemin de sauvegarde
    output_folder = os.path.join(os.path.dirname(args.input), '..', args.version)
    output_folder = os.path.normpath(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    output_clean = os.path.join(output_folder, f"coptic_parallel_corpus_fr_{args.version}_clean.csv")
    output_removed = os.path.join(output_folder, f"removed_verses_{args.version}.csv")

    # === Sauvegarder
    df_clean.to_csv(output_clean, index=False)
    df_removed = df[mask_removed]
    df_removed.to_csv(output_removed, index=False)

    # === Rapport
    print("=== Nettoyage terminé ===")
    print(f"Nombre de versets supprimés (copte manquant)  : {mask_coptic_invalid.sum()}")
    print(f"Nombre de versets supprimés (français manquant): {mask_french_missing.sum()}")
    print(f"Nombre de versets restants                    : {len(df_clean)}")
    print(f"Fichier propre : {output_clean}")
    print(f"Versets supprimés sauvegardés dans : {output_removed}")


# ===== EXEMPLE LANCER LE SCRIPT ======
# python3 clean_up_corpus.py --input crampon/coptic_parallel_corpus_fr_crampon.csv --version crampon