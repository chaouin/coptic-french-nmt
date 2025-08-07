import os
import warnings
from pathlib import Path

import evaluate
import nltk
import pandas as pd
from comet import download_model, load_from_checkpoint

# === Silence unnecessary warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# === NLTK downloads required
nltk.download("punkt")
nltk.download("wordnet")

# === Load metrics
print("✅ Loading METEOR and BERTScore...")
meteor_metric = evaluate.load("meteor")
bertscore_metric = evaluate.load("bertscore")

print("✅ Loading COMET...")
model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(model_path)

# === Directories
BASE_DIR = Path("../../experiment 4/generated translations/hiero")
OUTPUT_DIR = Path(__file__).parent / "evaluation scores/hiero"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Input files
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

# === Reference columns
ref_names = ["crampon", "segond", "darby"]

# === Evaluation function
def evaluate_all_metrics(label, csv_path):
    print(f"\n📥 Evaluating: {label}")
    df = pd.read_csv(csv_path)

    for gen_col in generated_columns:
        if gen_col not in df.columns:
            print(f"⚠️ Missing generated column: {gen_col}, skipping.")
            continue

        print(f"\n🔎 Evaluating: {gen_col}")
        candidates = df[gen_col].fillna("").tolist()

        for ref in ref_names:
            col_name = f"french_{ref}"
            if col_name not in df.columns:
                print(f"⚠️ Missing reference column: {col_name}, skipping.")
                continue

            references = df[col_name].fillna("").tolist()
            print(f"🔍 {len(references)} pairs for reference: {ref}...")

            # METEOR
            meteor_result = meteor_metric.compute(predictions=candidates, references=references)
            df[f"meteor_{ref}_{gen_col}"] = [meteor_result["meteor"]] * len(df)

            # BERTScore
            bertscore_result = bertscore_metric.compute(predictions=candidates, references=references, lang="fr")
            df[f"bertscore_{ref}_{gen_col}"] = bertscore_result["f1"]

            # COMET
            comet_data = [{"src": s, "mt": c, "ref": r} for s, c, r in zip(references, candidates, references)]
            comet_scores = comet_model.predict(comet_data, batch_size=8, num_workers=1, gpus=0)
            df[f"comet_{ref}_{gen_col}"] = comet_scores["scores"]

            # Summary
            avg_bertscore = sum(bertscore_result["f1"]) / len(bertscore_result["f1"])
            avg_comet = sum(comet_scores["scores"]) / len(comet_scores["scores"])
            print(f"📊 {ref} — METEOR: {meteor_result['meteor']:.4f} | "
                  f"BERTScore F1: {avg_bertscore:.4f} | COMET: {avg_comet:.4f}")

    # === Save file
    output_csv = OUTPUT_DIR / (csv_path.stem + "_other_scores.csv")
    df.to_csv(output_csv, index=False)
    print(f"✅ File saved: {output_csv}")

# === Run evaluations
for label, path in input_files.items():
    evaluate_all_metrics(label, path)
