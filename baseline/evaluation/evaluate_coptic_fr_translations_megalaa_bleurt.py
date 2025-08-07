from pathlib import Path

import pandas as pd
from bleurt import score

# === Parameters ===
bleurt_checkpoint = "../../BLEURT-20"  # BLEURT model folder
OUTPUT_DIR = Path(__file__).parent / "evaluation scores"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_DIR = Path("../generated translations")

# === Input files ===
input_files = {
    "megalaa": BASE_DIR / "generated_translations_megalaa_baseline.csv",
}

# === Generated translation columns ===
generated_columns = [
    "english_generated_translation",
]

print("✅ Loading BLEURT model...")
scorer = score.BleurtScorer(bleurt_checkpoint)

def evaluate_bleurt_multi_outputs(label, input_csv_path):
    print(f"\n📥 BLEURT evaluation for: {label}")
    df = pd.read_csv(input_csv_path)

    bleurt_scores = {}

    for gen_col in generated_columns:
        if gen_col not in df.columns:
            print(f"⚠️ Missing generated column: {gen_col}, skipping.")
            continue

        candidates = df[gen_col].fillna("").tolist()
        for ref_name in ["english_translation"]:
            ref_col = ref_name
            if ref_col not in df.columns:
                print(f"⚠️ Missing reference column: {ref_col}, skipping.")
                continue

            references = df[ref_col].fillna("").tolist()
            print(f"🔍 Evaluating {len(references)} pairs: {gen_col} vs {ref_name}...")
            scores = scorer.score(references=references, candidates=candidates)
            df[f"bleurt_{ref_name}"] = scores
            bleurt_scores[(ref_name, gen_col)] = scores

    # === Save results
    output_csv = OUTPUT_DIR / (input_csv_path.stem + "_bleurt.csv")
    df.to_csv(output_csv, index=False)

    # === Summary of average scores
    print(f"\n🎯 Average BLEURT scores for {label}:")
    for (ref, gen_col), scores in bleurt_scores.items():
        avg = sum(scores) / len(scores)
        print(f"   → {ref.capitalize()} vs {gen_col}: {avg:.4f}")

    print(f"💾 File saved: {output_csv}")

# === Run evaluation for each input file
for label, path in input_files.items():
    evaluate_bleurt_multi_outputs(label, path)
