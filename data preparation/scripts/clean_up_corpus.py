import argparse
import os

import pandas as pd

# === MAIN ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean a Coptic–French corpus by removing invalid verses.")
    parser.add_argument("--input", required=True, help="CSV file of the corpus to clean")
    parser.add_argument("--version", required=True, help="Target version name (e.g., darby, crampon, segond)")

    args = parser.parse_args()

    # === Load the CSV file
    df = pd.read_csv(args.input)

    # === Remove verses with invalid Coptic text
    mask_coptic_invalid = df['coptic_text'].str.strip().eq('[...]')
    mask_french_missing = df['french_translation'].isnull() | df['french_translation'].str.strip().eq('')
    mask_removed = mask_coptic_invalid | mask_french_missing

    df_clean = df[~mask_removed].copy()

    # === Clean biblical annotations at the beginning of the French text using regex
    df_clean['french_translation'] = df_clean['french_translation'].str.replace(
        r'^\(\s*[\d.:]+\s*\)\s*', '', regex=True
    )

    # === Determine save paths
    output_folder = os.path.join(os.path.dirname(args.input), '..', args.version)
    output_folder = os.path.normpath(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    output_clean = os.path.join(output_folder, f"coptic_parallel_corpus_fr_{args.version}_clean.csv")
    output_removed = os.path.join(output_folder, f"removed_verses_{args.version}.csv")

    # === Save cleaned and removed data
    df_clean.to_csv(output_clean, index=False)
    df_removed = df[mask_removed]
    df_removed.to_csv(output_removed, index=False)

    # === Report
    print("=== Cleaning complete ===")
    print(f"Number of verses removed (missing Coptic)   : {mask_coptic_invalid.sum()}")
    print(f"Number of verses removed (missing French)   : {mask_french_missing.sum()}")
    print(f"Number of verses remaining                  : {len(df_clean)}")
    print(f"Cleaned file saved to: {output_clean}")
    print(f"Removed verses saved to: {output_removed}")


# ===== EXAMPLE RUN COMMAND =====
# python3 clean_up_corpus.py --input crampon/coptic_parallel_corpus_fr_crampon.csv --version crampon
