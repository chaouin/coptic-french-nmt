from pathlib import Path
import pandas as pd

# === Parameters
RESULT_DIR = Path("evaluation scores/hiero")
csv_files = list(RESULT_DIR.glob("*_other_scores.csv")) + list(RESULT_DIR.glob("*_bleurt.csv"))

# === Storage
summary_rows = []
detailed_rows = []

# === Loop through evaluation files
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    file_label = csv_file.stem

    # === Extract dataset name from filename
    if file_label.startswith("generated_translations_exp_4_hiero_noisy_100"):
        dataset = "noisy_100"
    elif file_label.startswith("generated_translations_exp_4_hiero_noisy_50"):
        dataset = "noisy_50"
    elif file_label.startswith("generated_translations_exp_4_hiero_noisy_30"):
        dataset = "noisy_30"
    elif file_label.startswith("generated_translations_exp_4_hiero_noisy_10"):
        dataset = "noisy_10"
    elif file_label.startswith("generated_translations_exp_4_hiero_clean"):
        dataset = "clean"
    else:
        dataset = "unknown"

    # === For each metric column
    for col in df.columns:
        if any(col.startswith(prefix) for prefix in ["bleurt_", "meteor_", "bertscore_", "comet_"]):
            metric = col.split("_")[0]
            model = "_".join(col.split("_")[3:]) if len(col.split("_")) > 3 else "unknown_model"

            avg_score = df[col].mean()
            summary_rows.append({"metric": metric, "model": model, "avg_score": avg_score})
            detailed_rows.append({
                "metric": metric,
                "dataset": dataset,
                "model": model,
                "avg_score": avg_score
            })

# === Global summary
summary_df = pd.DataFrame(summary_rows)
global_pivot = summary_df.groupby(["model", "metric"]).mean().reset_index()
pivot_df = global_pivot.pivot(index="model", columns="metric", values="avg_score").round(4)

# === Highlight best model per metric
best_models = pivot_df.idxmax()
for metric in pivot_df.columns:
    best_model = best_models[metric]
    pivot_df.loc[best_model, metric] = f"{pivot_df.loc[best_model, metric]} 🏆"

# === Display global summary table
print("\n📊 Global average scores by model and metric:")
print(pivot_df)

# === Save global summary
pivot_df.to_csv(RESULT_DIR / "summary_best_model_per_metric.csv")

# === Detailed performance table per dataset (not aggregated)
detailed_df = pd.DataFrame(detailed_rows)
detailed_pivot = detailed_df.pivot_table(index=["dataset", "model"], columns="metric", values="avg_score").round(4)
detailed_pivot.to_csv(RESULT_DIR / "detailed_model_performance_per_dataset.csv")

# === Identify best/worst datasets per model/metric
best_worst_rows = []
for (model, metric), group in detailed_df.groupby(["model", "metric"]):
    idxmax = group["avg_score"].idxmax()
    idxmin = group["avg_score"].idxmin()
    best_dataset = group.loc[idxmax, "dataset"]
    worst_dataset = group.loc[idxmin, "dataset"]
    best_score = group.loc[idxmax, "avg_score"]
    worst_score = group.loc[idxmin, "avg_score"]

    best_worst_rows.append({
        "model": model,
        "metric": metric,
        "best_dataset": best_dataset,
        "best_score": best_score,
        "worst_dataset": worst_dataset,
        "worst_score": worst_score
    })

best_worst_df = pd.DataFrame(best_worst_rows)
best_worst_df = best_worst_df.sort_values(by=["model", "metric"])
best_worst_df.to_csv(RESULT_DIR / "best_worst_dataset_per_model_metric.csv", index=False)

print(f"\n💾 Files saved:\n - summary_best_model_per_metric.csv"
      f"\n - detailed_model_performance_per_dataset.csv"
      f"\n - best_worst_dataset_per_model_metric.csv")
