import pandas as pd
from pathlib import Path

# === Paramètres
RESULT_DIR = Path("evaluation scores")
csv_files = list(RESULT_DIR.glob("*_other_scores.csv")) + list(RESULT_DIR.glob("*_bleurt.csv"))

# === Stockage global
model_scores = {}

# === Parcours des fichiers
for csv_file in csv_files:
    df = pd.read_csv(csv_file)

    # Pour chaque colonne de métrique
    for col in df.columns:
        # Chercher si c'est une métrique et extraire le nom du modèle
        if any(col.startswith(prefix) for prefix in ["bleurt_", "meteor_", "bertscore_", "comet_"]):
            # Exemple: comet_crampon_generated_translation_pipeline
            parts = col.split("_")
            # L'indice 3 contient généralement le nom du modèle
            model = "_".join(parts[3:]) if len(parts) > 3 else "unknown_model"
            metric = parts[0]

            # Initialiser si pas encore vu
            if (metric, model) not in model_scores:
                model_scores[(metric, model)] = []

            # Ajouter les valeurs
            model_scores[(metric, model)].append(df[col].mean())

# === Calculer les moyennes globales par modèle / metric
summary_rows = []
for (metric, model), values in model_scores.items():
    global_avg = sum(values) / len(values)
    summary_rows.append({"metric": metric, "model": model, "avg_score": global_avg})

# === DataFrame final
summary_df = pd.DataFrame(summary_rows)

# === Pivot pour avoir modèle en ligne et métrique en colonne
pivot_df = summary_df.pivot(index="model", columns="metric", values="avg_score")
pivot_df = pivot_df.round(4)

# === Highlight meilleur modèle par métrique
best_models = pivot_df.idxmax()
for metric in pivot_df.columns:
    best_model = best_models[metric]
    pivot_df.loc[best_model, metric] = f"{pivot_df.loc[best_model, metric]} 🏆"

# === Affichage
print("\n📊 Tableau des moyennes globales par modèle et par métrique :")
print(pivot_df)

# === Sauvegarde
output_csv = RESULT_DIR / "summary_best_model_per_metric.csv"
pivot_df.to_csv(output_csv)
print(f"\n💾 Fichier sauvegardé : {output_csv}")
