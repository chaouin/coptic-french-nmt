import pandas as pd

# === Files
csv_source = "../../experiment 1/generated translations/generated_translations_evaluation_data_all_methods.csv"
csv_target = "generated_translations_evaluation_data_3_finetuned_models.csv"

# === Parameters
column_to_copy = "generated_translation_megalaa_finetune"
new_column_name = "generated_translation_megalaa"

# === Load CSV files
df_source = pd.read_csv(csv_source)
df_target = pd.read_csv(csv_target)

# === Check if column exists
if column_to_copy not in df_source.columns:
    raise ValueError(f"The column '{column_to_copy}' is missing from the source file.")

# === Add renamed column to target
df_target[new_column_name] = df_source[column_to_copy]

# === Save the updated target file
df_target.to_csv("generated_translations_evaluation_data_all_finetuned_models.csv", index=False)
print(f"✅ Column copied as '{new_column_name}' and added to the target file.")
