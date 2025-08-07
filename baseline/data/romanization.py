from pathlib import Path

import pandas as pd
import uroman as ur

# === Input files ===
input_files = [
    Path("../../evaluation data/evaluation_data_cop_en.csv"),
]

# === Initialize uroman
uroman = ur.Uroman()

def preserve_brackets(text):
    return str(text).replace('[]', '<MISSING>')

for input_file in input_files:
    # Upload CSV
    df = pd.read_csv(input_file)

    # Apply romanization
    df['coptic_text_romanized'] = df['coptic_text'].apply(
        lambda x: uroman.romanize_string(preserve_brackets(x)).replace('<MISSING>', '[]')
    )

    # Keeping right columns
    df_final = df[['verse_id', 'coptic_text_romanized', 'english_translation']]

    # Construct the output file's name
    output_file = input_file.with_name(f"{input_file.stem}_romanized.csv")

    # Save
    df_final.to_csv(output_file, index=False)

    print(f"✅ Romanization finished for {input_file}")
    print(f" File saved under : {output_file}")
