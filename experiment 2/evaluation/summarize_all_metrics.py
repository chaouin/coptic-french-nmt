from pathlib import Path

import pandas as pd

# === Parameters
RESULT_DIR = Path("evaluation scores")
csv_files = list(RESULT_DIR.glob("*_other_scores.csv")) + list(RESULT_DIR.glob("*_bleurt.csv"))

# === Global score storage
model_scores = {}

# === Parse each file
for csv_file in csv_files:
    df = pd.read_csv(csv_file)

    # For each metric column
    for col in df.columns:
        # Check if the column is a metric and extract model name
        if any(col.startswith(prefix) for prefix in ["bleurt_", "meteor_", "bertscore_", "comet_"]):
            # Example: comet_crampon_generated_translation_pipeline
            parts = col.split("_")
            # Index 3 typically contains the model name
            model = "_".join(parts[3:]) if len(parts) > 3 else "unknown_model"
            metric = parts[0]

            # Initialize if not seen before
            if (metric, model) not in model_scores:
                model_scores[(metric, model)] = []

            # Add mean value
            model_scores[(metric, model)].append(df[col].mean())

# === Compute global averages per model/metric
summary_rows = []
for (metric, model), values in model_scores.items():
    global_avg = sum(values) / len(values)
    summary_rows.append({"metric": metric, "model": model, "avg_score": global_avg})

# === Create final DataFrame
summary_df = pd.DataFrame(summary_rows)

# === Pivot to have models as rows and metrics as columns
pivot_df = summary_df.pivot(index="model", columns="metric", values="avg_score")
pivot_df = pivot_df.round(4)

# === Highlight best model per metric
best_models = pivot_df.idxmax()
for metric in pivot_df.columns:
    best_model = best_models[metric]
    pivot_df.loc[best_model, metric] = f"{pivot_df.loc[best_model, metric]} 🏆"

# === Display
print("\n📊 Summary table of average scores per model and per metric:")
print(pivot_df)

# === Save to CSV
output_csv = RESULT_DIR / "summary_best_model_per_metric.csv"
pivot_df.to_csv(output_csv)
print(f"\n💾 File saved: {output_csv}")
