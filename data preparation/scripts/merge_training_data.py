import argparse
import glob
import os

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge corpus files into a single global training file."
    )
    parser.add_argument("--root", default="..", help="Root folder of the parallel corpus")
    parser.add_argument("--pattern", default="*_romanized.csv",
                        help="Pattern of files to merge (default: *_romanized.csv)")
    parser.add_argument("--output", default="train_all_data.csv",
                        help="Name of the output file (default: train_all_data.csv)")
    parser.add_argument("--only_clean", action="store_true",
                        help="Merge only clean (non-noisy) files (default: False)")
    parser.add_argument("--only_noisy", action="store_true",
                        help="Merge only noisy files (default: False)")
    parser.add_argument("--version", help="Filter files by Bible version (e.g., segond)")

    args = parser.parse_args()

    if args.only_clean and args.only_noisy:
        raise ValueError("Options --only_clean and --only_noisy are mutually exclusive.")

    # Search for files
    all_files = [
        f for f in glob.glob(os.path.join(args.root, "**", args.pattern), recursive=True)
        if "training data" not in f.replace("\\", "/")
           and (args.version.lower() in f.lower() if args.version else True)
    ]

    print(f"Number of files found: {len(all_files)}")
    for f in all_files:
        print(f"- {f}")

    columns_to_keep = ['verse_id', 'coptic_text_romanized', 'french_translation']
    dfs = []
    clean_count = 0
    noisy_count = 0

    for filename in all_files:
        is_noisy = "noisy" in filename.lower()

        if args.only_clean and is_noisy:
            print(f"⏭️ File skipped (noisy and --only_clean enabled): {filename}")
            continue
        if args.only_noisy and not is_noisy:
            print(f"⏭️ File skipped (clean and --only_noisy enabled): {filename}")
            continue

        df = pd.read_csv(filename)
        if all(col in df.columns for col in columns_to_keep):
            df = df[columns_to_keep].copy()

            if is_noisy:
                noisy_count += len(df)
            else:
                clean_count += len(df)

            dfs.append(df)
        else:
            print(f"⚠️ Missing required columns in {filename}, file skipped.")

    # Merge
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        print("=== Merge Summary ===")
        print(f"Total verses after merge        : {len(combined_df)}")
        print(f"Number of clean verses          : {clean_count}")
        print(f"Number of noisy verses          : {noisy_count}")
        print(f"Clean ratio                     : {clean_count / len(combined_df) * 100:.2f} %")

        combined_df.to_csv(args.output, index=False)
        print(f"✅ Final file saved to: {args.output}")
    else:
        print("No valid files found. Nothing to merge.")

# === Example: normal run (clean + noisy) ===
# python3 merge_training_data.py --root ".." --output ../training\ data/train_all_data.csv

# === Example: --only_clean (clean verses only) ===
# python3 merge_training_data.py --root ".." --output ../training\ data/train_clean_data.csv --only_clean

# === Example: --only_noisy (noisy verses only) ===
# python3 merge_training_data.py --root ".." --output ../training\ data/train_noisy_data.csv --only_noisy

# === Example: --version (only specific Bible version) ===
# python3 merge_training_data.py --root ".." --version segond --output ../training\ data/train_segond_all_data.csv
