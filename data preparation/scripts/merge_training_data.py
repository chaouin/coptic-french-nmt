import argparse
import pandas as pd
import glob
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fusionner les fichiers de corpus en un fichier d'entraînement global.")
    parser.add_argument("--root", default="..", help="Dossier parellel corpus")
    parser.add_argument("--pattern", default="*_romanized.csv",
                        help="Pattern des fichiers à fusionner (défaut: *_romanized.csv)")
    parser.add_argument("--output", default="train_all_data.csv", help="Nom du fichier de sortie (défaut: train_all_data.csv)")
    parser.add_argument("--only_clean", action="store_true",
                        help="Ne fusionner que les fichiers non noisy (défaut: False)")
    parser.add_argument("--only_noisy", action="store_true",
                        help="Ne fusionner que les fichiers noisy (défaut: False)")
    parser.add_argument("--version", help="Filtrer les fichiers par version de la Bible (ex: segond)")

    args = parser.parse_args()

    if args.only_clean and args.only_noisy:
        raise ValueError("Les options --only_clean et --only_noisy sont mutuellement exclusives.")

    # Chercher les fichiers
    all_files = [
        f for f in glob.glob(os.path.join(args.root, "**", args.pattern), recursive=True)
        if "training data" not in f.replace("\\", "/")
           and (args.version.lower() in f.lower() if args.version else True)
    ]

    print(f"Nombre de fichiers trouvés : {len(all_files)}")
    for f in all_files:
        print(f"- {f}")

    columns_to_keep = ['verse_id', 'coptic_text_romanized', 'french_translation']
    dfs = []
    compteur_clean = 0
    compteur_noisy = 0

    for filename in all_files:
        is_noisy = "noisy" in filename.lower()

        if args.only_clean and is_noisy:
            print(f"⏭️ Fichier ignoré (noisy et --only_clean activé) : {filename}")
            continue
        if args.only_noisy and not is_noisy:
            print(f"⏭️ Fichier ignoré (clean et --only_noisy activé) : {filename}")
            continue

        df = pd.read_csv(filename)
        if all(col in df.columns for col in columns_to_keep):
            df = df[columns_to_keep].copy()

            if is_noisy:
                compteur_noisy += len(df)
            else:
                compteur_clean += len(df)

            dfs.append(df)
        else:
            print(f"⚠️ Colonnes manquantes dans {filename}, fichier ignoré.")

    # Fusionner
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        print("=== Résumé de la fusion ===")
        print(f"Nombre total de versets après fusion : {len(combined_df)}")
        print(f"Nombre de versets propres (clean)     : {compteur_clean}")
        print(f"Nombre de versets bruités (noisy)     : {compteur_noisy}")
        print(f"Ratio clean / total : {compteur_clean / len(combined_df) * 100:.2f} %")

        combined_df.to_csv(args.output, index=False)
        print(f"Fichier final sauvegardé sous : {args.output}")
    else:
        print("Aucun fichier valide trouvé. Rien à fusionner.")

# === Exemple d’appel normal (avec clean + noisy) ===
# python3 merge_training_data.py --root ".." --output ../training\ data/train_all_data.csv

# === Exemple d’appel --only_clean (contient uniquement les versets propres sans bruits) ===
# python3 merge_training_data.py --root ".." --output ../training\ data/train_clean_data.csv --only_clean

# === Exemple d'appel --only_noisy (contient uniquement les versets bruitées) ===
# python3 merge_training_data.py --root ".." --output ../training\ data/train_noisy_data.csv --only_noisy

# === Exemple d'appel --version (contient uniquement une version précise de la bible) ===
# python3 merge_training_data.py --root ".." --version segond --output ../training\ data/train_segond_all_data.csv