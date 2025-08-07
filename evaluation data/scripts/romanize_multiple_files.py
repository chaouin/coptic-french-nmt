import pandas as pd
import uroman as ur
import os  # Import the os module for path manipulation

# === Parameters ===
# Define lists for input and output files
INPUT_FILES = [
    "galatians/galatians_coptic_3versions.csv",
    "hebrews/hebrews_coptic_3versions.csv",
    "mark/mark_coptic_3versions.csv"
]
# You can generate output filenames based on input, or define them explicitly
OUTPUT_FILES = [
    "galatians/galatians_coptic_3versions_romanized.csv",
    "hebrews/hebrews_coptic_3versions_romanized.csv",
    "mark/mark_coptic_3versions_romanized.csv"
]

# Ensure input and output file lists have the same number of elements
if len(INPUT_FILES) != len(OUTPUT_FILES):
    raise ValueError("The number of input files must match the number of output files.")

# === Initialize Uroman ===
uroman = ur.Uroman()

# === Process Each File ===
for i in range(len(INPUT_FILES)):
    input_file = INPUT_FILES[i]
    output_file = OUTPUT_FILES[i]

    print(f"🔄 Processing {input_file}...")

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"⚠️ Error: Input file not found: {input_file}. Skipping.")
        continue

    # Load the CSV
    df = pd.read_csv(input_file)

    # Romanize and replace the column
    # Ensure 'coptic_text' column exists before applying romanization
    if "coptic_text" in df.columns:
        df["coptic_text_romanized"] = df["coptic_text"].apply(lambda x: uroman.romanize_string(str(x)))
        df.drop(columns=["coptic_text"], inplace=True)
    else:
        print(f"⚠️ Warning: 'coptic_text' column not found in {input_file}. Skipping romanization for this file.")
        # If 'coptic_text' isn't there, we'll try to keep existing columns for output
        # You might want to adjust the output columns based on this scenario
        df["coptic_text_romanized"] = ""  # Add an empty column if not found to prevent error later

    # Reorder columns as requested
    # It's safer to only include columns that actually exist in the DataFrame
    desired_columns = ["verse_id", "coptic_text_romanized", "french_segond", "french_darby", "french_crampon"]
    existing_columns = [col for col in desired_columns if col in df.columns]
    df = df[existing_columns]

    # Save
    df.to_csv(output_file, index=False)

    print(f"✅ Romanization complete for {input_file}.")
    print(f"File saved to: {output_file}\n")

print("✨ All specified files processed.")
