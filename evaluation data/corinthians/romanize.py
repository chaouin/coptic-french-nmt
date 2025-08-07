import pandas as pd
import uroman as ur

# === Parameters ===
INPUT_FILE = "1corinthians_coptic_3versions.csv"
OUTPUT_FILE = "1corinthians_coptic_3versions_romanized.csv"

# === Load the CSV
df = pd.read_csv(INPUT_FILE)

# === Initialize Uroman
uroman = ur.Uroman()

# === Romanize and replace the column
df["coptic_text_romanized"] = df["coptic_text"].apply(lambda x: uroman.romanize_string(str(x)))
df.drop(columns=["coptic_text"], inplace=True)

# === Reorder columns as requested
df = df[["verse_id", "coptic_text_romanized", "french_segond", "french_darby", "french_crampon"]]

# === Save to file
df.to_csv(OUTPUT_FILE, index=False)

print("✅ Romanization complete.")
print(f"File saved to: {OUTPUT_FILE}")
