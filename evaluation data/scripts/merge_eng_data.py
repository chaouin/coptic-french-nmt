import pandas as pd
import os

# --- Parameters ---
FILES_TO_MERGE = [
    "galatians/galatians_coptic_en.csv",
    "hebrews/hebrews_coptic_en.csv",
    "mark/mark_coptic_en.csv",
    "corinthians/1corinthians_coptic_en.csv"
]

# Name for the final merged output file
MERGED_OUTPUT_FILE = "evaluation_data_cop_en.csv"

# --- Merge Process ---
all_dfs = []  # This list will hold each DataFrame as we load it

print("🔄 Starting merge process...")

for file_path in FILES_TO_MERGE:
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: File not found: {file_path}. Skipping this file.")
        continue

    try:
        df = pd.read_csv(file_path)
        all_dfs.append(df)
        print(f"✅ Loaded {file_path}")
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}. Skipping this file.")

if not all_dfs:
    print("❌ No files were successfully loaded. Nothing to merge.")
else:
    # Concatenate all DataFrames in the list
    # ignore_index=True resets the index of the new DataFrame
    merged_df = pd.concat(all_dfs, ignore_index=True)

    # --- Save Merged File ---
    try:
        merged_df.to_csv(MERGED_OUTPUT_FILE, index=False)
        print(f"\n✨ All files successfully merged into: {MERGED_OUTPUT_FILE}")
        print(f"Total rows in merged file: {len(merged_df)}")
    except Exception as e:
        print(f"❌ Error saving merged file to {MERGED_OUTPUT_FILE}: {e}")

print("--- Merge process complete ---")