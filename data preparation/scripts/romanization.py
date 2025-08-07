import argparse
import os

import pandas as pd
import uroman as ur

# === MAIN ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Romanize the 'coptic_text' column in a Coptic–French corpus.")
    parser.add_argument("--input", required=True, help="CSV file of the corpus to romanize")
    parser.add_argument("--version", required=True, help="Target version name (e.g., darby, crampon, segond)")
    parser.add_argument("--suffix", default="romanized", help="Suffix for the output file name (e.g., romanized)")

    args = parser.parse_args()

    # === Load CSV file
    df = pd.read_csv(args.input)

    # === Initialize uroman
    uroman = ur.Uroman()

    def preserve_brackets(text):
        return str(text).replace('[]', '<MISSING>')

    # === Apply romanization
    df['coptic_text_romanized'] = df['coptic_text'].apply(
        lambda x: uroman.romanize_string(preserve_brackets(x)).replace('<MISSING>', '[]')
    )

    # === Keep only required columns
    df_final = df[['verse_id', 'coptic_text_romanized', 'french_translation']]

    # === Determine output path
    output_folder = os.path.join(os.path.dirname(args.input), '..', args.version)
    output_folder = os.path.normpath(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Construct output file name with suffix
    input_basename = os.path.basename(args.input)
    input_basename_no_ext = os.path.splitext(input_basename)[0]
    output_file = os.path.join(output_folder, f"{input_basename_no_ext}_{args.suffix}.csv")

    # === Save file
    df_final.to_csv(output_file, index=False)

    print("=== Romanization complete ===")
    print(f"File saved to: {output_file}")
    print("Columns retained: verse_id, coptic_text_romanized, french_translation")

# ===== EXAMPLE USAGE =====
# python3 romanization.py --input crampon/coptic_parallel_corpus_fr_crampon_clean.csv --version crampon
