import pandas as pd
from pathlib import Path

# === Paramètres
RESULT_DIR = Path("evaluation scores")
csv_files = list(RESULT_DIR.glob("*_other_scores.csv")) + list(RESULT_DIR.glob("*_bleurt.csv"))

# === Stockage
summary_rows = []

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    file_label = csv_file.stem  # ex: generated_translations_megalaa_baseline_other_scores

    # Déterminer le modèle à partir du nom du fichier
    if "hiero" in file_label:
        model = "hiero_baseline"
    elif "megalaa" in file_label:
        model = "megalaa_baseline"
    elif "t5" in file_label:
        model = "t5_baseline"
    else:
        model = "unknown_baseline"

    # Calculer la moyenne pour chaque métrique
    for col in df.columns:
        if any(col.startswith(prefix) for prefix in ["bleurt_", "meteor_", "bertscore_", "comet_"]):
            metric = col.split("_")[0]
            avg_score = df[col].mean()
            summary_rows.append({"model": model, "metric": metric, "avg_score": avg_score})

# === Création du DataFrame
summary_df = pd.DataFrame(summary_rows)

# === Pivot pour avoir un tableau clair
pivot_df = summary_df.pivot_table(index="model", columns="metric", values="avg_score").round(4)

# === Affichage
print("\n📊 Tableau des baselines :")
print(pivot_df)

# === Sauvegarde
pivot_df.to_csv(RESULT_DIR / "summary_baseline_models.csv")
print(f"\n💾 Fichier sauvegardé : {RESULT_DIR / 'summary_baseline_models.csv'}")
