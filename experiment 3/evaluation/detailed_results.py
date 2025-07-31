import pandas as pd
from pathlib import Path

# === Paramètres
RESULT_DIR = Path("evaluation scores/hiero")  # à adapter si besoin
csv_files = list(RESULT_DIR.glob("*_other_scores.csv")) + list(RESULT_DIR.glob("*_bleurt.csv"))

# === Initialisation
detailed_rows = []

# === Parcours des fichiers
for csv_file in csv_files:
    df = pd.read_csv(csv_file)

    for testset in ["segond", "darby", "crampon"]:
        for col in df.columns:
            if any(col.startswith(prefix) for prefix in ["bleurt_", "meteor_", "bertscore_", "comet_"]) and f"french_{testset}_" in col:
                metric = col.split("_")[0]
                model = "_".join(col.split("_")[4:]) if len(col.split("_")) > 4 else "unknown_model"

                avg_score = df[col].mean()
                detailed_rows.append({
                    "metric": metric,
                    "testset": testset,
                    "model": model,
                    "avg_score": avg_score
                })

# === Format final
df_detailed = pd.DataFrame(detailed_rows)
pivot_df = df_detailed.pivot_table(index=["testset", "model"], columns="metric", values="avg_score").round(4)

# === Sauvegarde
output_path = RESULT_DIR / "detailed_model_performance_per_testset.csv"
pivot_df.to_csv(output_path)
print(f"✅ Fichier sauvegardé : {output_path}")
