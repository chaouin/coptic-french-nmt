import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from tqdm import tqdm

# --- Global Parameters ---
n_samples = None  # Leave as None to process all samples, or set an integer (e.g., 100)

# --- Input Evaluation Files ---
# List of all evaluation data files. Each will result in a distinct output file.
INPUT_EVAL_FILES = [
    "../../evaluation data/evaluation_data.csv",
    "../../evaluation data/evaluation_data_noisy_10.csv",
    "../../evaluation data/evaluation_data_noisy_30.csv",
    "../../evaluation data/evaluation_data_noisy_50.csv",
    "../../evaluation data/evaluation_data_noisy_100.csv"
]

# --- Definition of Models to Use for Translation Generation ---
model_paths = {
    "opus_clean": "../../models/opus-finetuned-coptic-fr-clean-data",
    "opus_noisy_10": "../../models/opus-finetuned-coptic-fr-noisy-10-data",
    "opus_noisy_30": "../../models/opus-finetuned-coptic-fr-noisy-30-data",
    "opus_noisy_50": "../../models/opus-finetuned-coptic-fr-noisy-50-data",
    "opus_noisy_100": "../../models/opus-finetuned-coptic-fr-noisy-100-data",
}

# --- Function to generate translations for a single model and DataFrame ---
# This function will be executed in parallel processes.
def generate_translations_for_model(model_key, model_path, df_input, temp_output_path):
    import pandas as pd
    from transformers import MarianTokenizer, MarianMTModel
    import torch
    from tqdm import tqdm

    print(f"🚀 [{os.getpid()}] Starting translation for model: {model_key}")

    tokenizer = MarianTokenizer.from_pretrained(model_path, local_files_only=True)
    model = MarianMTModel.from_pretrained(model_path, local_files_only=True).to("cpu")

    translations = []
    for text_to_translate in tqdm(df_input["coptic_text_romanized"], total=len(df_input),
                                  desc=f"Translating with {model_key}"):
        text = ">>fra<< " + str(text_to_translate)
        inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        with torch.no_grad():
            output = model.generate(**inputs, max_length=128, num_beams=6,
                                    repetition_penalty=1.5, length_penalty=2.5)
        translation = tokenizer.decode(output[0], skip_special_tokens=True)
        translations.append(translation)

    df_temp_result = pd.DataFrame({
        "verse_id": df_input["verse_id"],
        f"generated_translation_{model_key}": translations
    })

    df_temp_result.to_csv(temp_output_path, index=False)
    print(f"✅ [{os.getpid()}] Translations for {model_key} saved to {temp_output_path}")
    return temp_output_path

# --- Main Processing Loop for Each Evaluation Dataset File ---
if __name__ == "__main__":

    output_dir = "generated_translations_per_dataset_all_models"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 Output files will be saved in: {output_dir}")

    for input_eval_file_path in INPUT_EVAL_FILES:
        print(f"\n--- Processing evaluation dataset: {input_eval_file_path} ---")

        base_name = os.path.basename(input_eval_file_path).replace(".csv", "")
        output_file_name = os.path.join(output_dir, f"{base_name}_all_models_generated_translations.csv")

        if not os.path.exists(input_eval_file_path):
            print(f"⚠️ Error: Input evaluation file not found: {input_eval_file_path}. Skipping this dataset.")
            continue

        current_df = pd.read_csv(input_eval_file_path)
        if n_samples:
            current_df = current_df.sample(n=n_samples, random_state=42).reset_index(drop=True)

        temp_dir_for_dataset = os.path.join(output_dir, f"temp_{base_name}_translations")
        os.makedirs(temp_dir_for_dataset, exist_ok=True)

        df_for_parallel = current_df[["verse_id", "coptic_text_romanized"]].copy()

        temp_output_paths_for_dataset = []
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = []
            for model_key, model_path in model_paths.items():
                temp_out_path_model = os.path.join(temp_dir_for_dataset, f"tmp_{base_name}_{model_key}.csv")
                futures.append(executor.submit(
                    generate_translations_for_model,
                    model_key, model_path, df_for_parallel, temp_out_path_model
                ))
                temp_output_paths_for_dataset.append(temp_out_path_model)

            for future in tqdm(futures, desc=f"Waiting for parallel translations for {base_name}"):
                future.result()

        for model_key in model_paths.keys():
            tmp_path = os.path.join(temp_dir_for_dataset, f"tmp_{base_name}_{model_key}.csv")
            if os.path.exists(tmp_path):
                df_temp_model_output = pd.read_csv(tmp_path)
                current_df = pd.merge(current_df, df_temp_model_output, on="verse_id", how="left")
            else:
                print(f"⚠️ Warning: Temporary file {tmp_path} not found for merging {model_key} for {base_name}.")

        for temp_file in temp_output_paths_for_dataset:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        if os.path.exists(temp_dir_for_dataset):
            os.rmdir(temp_dir_for_dataset)

        current_df.to_csv(output_file_name, index=False)
        print(f"✅ Final translations for {base_name} saved to: {output_file_name}")

    print("\n✨ All evaluation datasets processed and results saved to individual files.")
