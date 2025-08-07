from pathlib import Path

import pandas as pd

# === Parameters
RESULT_DIR = Path("evaluation scores")
csv_files = list(RESULT_DIR.glob("*_other_scores.csv")) + list(RESULT_DIR.glob("*_bleurt.csv"))

# === Storage for summary
summary_rows = []

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    file_label = csv_file.stem  # e.g., generated_translations_megalaa_baseline_other_scores

    # Determine model name from the file name
    if "hiero" in file_label:
        model = "hiero_baseline"
    elif "megalaa" in file_label:
        model = "megalaa_baseline"
    elif "t5" in file_label:
        model = "t5_baseline"
    else:
        model = "unknown_baseline"

    # Compute average for each metric
    for col in df.columns:
        if any(col.startswith(prefix) for prefix in ["bleurt_", "meteor_", "bertscore_", "comet_"]):
            metric = col.split("_")[0]
            avg_score = df[col].mean()
            summary_rows.append({"model": model, "metric": metric, "avg_score": avg_score})

# === Create summary DataFrame
summary_df = pd.DataFrame(summary_rows)

# === Pivot for a clearer table
pivot_df = summary_df.pivot_table(index="model", columns="metric", values="avg_score").round(4)

# === Display
print("\n📊 Baseline summary table:")
print(pivot_df)

# === Save to file
pivot_df.to_csv(RESULT_DIR / "summary_baseline_models.csv")
print(f"\n💾 File saved: {RESULT_DIR / 'summary_baseline_models.csv'}")
