import os
import pandas as pd
from comet import download_model, load_from_checkpoint
import evaluate
import nltk
from pathlib import Path
import warnings

# === Silence warnings inutiles
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# === Téléchargement nécessaire pour NLTK
nltk.download("punkt")
nltk.download("wordnet")

# === Chargement des métriques en ligne
print("✅ Chargement METEOR et BERTScore...")
meteor_metric = evaluate.load("meteor")
bertscore_metric = evaluate.load("bertscore")

print("✅ Chargement COMET...")
comet_model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(comet_model_path)

# === Répertoires d’évaluation
BASE_DIR = Path("../../experiment 3/generated translations/hiero")
OUTPUT_DIR = Path(__file__).parent / "evaluation scores/hiero"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Fichier d’entrée
input_files = {
    "evaluation_data": BASE_DIR / "generated_translations_exp_3_hiero.csv",
}

generated_columns = [
    "generated_translation_hiero_all",
    "generated_translation_hiero_crampon",
    "generated_translation_hiero_darby",
    "generated_translation_hiero_segond"
]

ref_names = ["french_segond", "french_darby", "french_crampon"]

# === Évaluation combinée
def evaluate_all_metrics(label, csv_path):
    print(f"\n📥 Évaluation pour : {label}")
    df = pd.read_csv(csv_path)

    for gen_col in generated_columns:
        if gen_col not in df.columns:
            print(f"⚠️ Colonne générée absente : {gen_col}, sautée.")
            continue

        print(f"\n🔎 Évaluation de : {gen_col}")
        candidates = df[gen_col].fillna("").tolist()

        for ref in ref_names:
            col_name = ref
            if col_name not in df.columns:
                print(f"⚠️ Colonne référence absente : {col_name}, sautée.")
                continue

            references = df[col_name].fillna("").tolist()
            print(f"🔍 {len(references)} paires avec référence {ref}...")

            meteor_result = meteor_metric.compute(predictions=candidates, references=references)
            df[f"meteor_{ref}_{gen_col}"] = [meteor_result["meteor"]] * len(df)

            bertscore_result = bertscore_metric.compute(predictions=candidates, references=references, lang="fr")
            df[f"bertscore_{ref}_{gen_col}"] = bertscore_result["f1"]

            comet_data = [{"src": s, "mt": c, "ref": r} for s, c, r in zip(references, candidates, references)]
            comet_scores = comet_model.predict(comet_data, batch_size=8, num_workers=1, gpus=0)
            df[f"comet_{ref}_{gen_col}"] = comet_scores["scores"]

            avg_bertscore = sum(bertscore_result["f1"]) / len(bertscore_result["f1"])
            avg_comet = sum(comet_scores["scores"]) / len(comet_scores["scores"])
            print(f"📊 {ref} — METEOR : {meteor_result['meteor']:.4f} | "
                  f"BERTScore F1 : {avg_bertscore:.4f} | COMET : {avg_comet:.4f}")

    output_csv = OUTPUT_DIR / (csv_path.stem + "_other_scores.csv")
    df.to_csv(output_csv, index=False)
    print(f"✅ Fichier enrichi sauvegardé : {output_csv}")

# === Exécution
for label, path in input_files.items():
    evaluate_all_metrics(label, path)
