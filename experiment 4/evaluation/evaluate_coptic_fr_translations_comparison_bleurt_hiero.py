from pathlib import Path
import pandas as pd
from bleurt import score

# === Parameters ===
bleurt_checkpoint = "../../BLEURT-20"  # BLEURT model directory
OUTPUT_DIR = Path(__file__).parent / "evaluation scores"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_DIR = Path("../../experiment 4/generated translations/hiero")

input_files = {
    "clean": BASE_DIR / "generated_translations_exp_4_hiero_clean.csv",
    "noisy_10": BASE_DIR / "generated_translations_exp_4_hiero_noisy_10.csv",
    "noisy_30": BASE_DIR / "generated_translations_exp_4_hiero_noisy_30.csv",
    "noisy_50": BASE_DIR / "generated_translations_exp_4_hiero_noisy_50.csv",
    "noisy_100": BASE_DIR / "generated_translations_exp_4_hiero_noisy_100.csv",
}

# === Generated translation columns
generated_columns = [
    "generated_translation_hiero_clean",
    "generated_translation_hiero_noisy_10",
    "generated_translation_hiero_noisy_30",
    "generated_translation_hiero_noisy_50",
    "generated_translation_hiero_noisy_100"
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
        for ref_name in ["crampon", "segond", "darby"]:
            ref_col = f"french_{ref_name}"
            if ref_col not in df.columns:
                print(f"⚠️ Missing reference column: {ref_col}, skipping.")
                continue

            references = df[ref_col].fillna("").tolist()
            print(f"🔍 Evaluating {len(references)} pairs for {gen_col} vs {ref_name}...")
            scores = scorer.score(references=references, candidates=candidates)
            df[f"bleurt_{ref_name}_{gen_col}"] = scores
            bleurt_scores[(ref_name, gen_col)] = scores

    # === Save file
    output_csv = OUTPUT_DIR / (input_csv_path.stem + "_bleurt.csv")
    df.to_csv(output_csv, index=False)

    # === Print average scores summary
    print(f"\n🎯 Average BLEURT scores for {label}:")
    for (ref, gen_col), scores in bleurt_scores.items():
        avg = sum(scores) / len(scores)
        print(f"   → {ref.capitalize()} vs {gen_col}: {avg:.4f}")

    print(f"💾 File saved: {output_csv}")

# === Process all input files
for label, path in input_files.items():
    evaluate_bleurt_multi_outputs(label, path)
