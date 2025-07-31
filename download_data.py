import os
from huggingface_hub import hf_hub_download

base_paths = [
    "experiment 1/",
    "experiment 2/",
    "experiment 3/",
    "experiment 4/",
    "baseline/",
    "data preparation",
    "evaluation data",
]

# Helper: download all CSV files
def recursive_download_csvs(repo_id, base_dirs):
    for base_dir in base_dirs:
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith(".csv"):
                    full_path = os.path.join(root, file)
                    try:
                        print(f"⬇️ Downloading: {full_path}")
                        hf_hub_download(
                            repo_id=repo_id,
                            filename=full_path,
                            repo_type="dataset",
                            local_dir=".",
                            local_dir_use_symlinks=False
                        )
                    except Exception as e:
                        print(f"⚠️ Failed to download {full_path}: {e}")

# Run it
recursive_download_csvs("chaouin/coptic-french-translation", base_paths)
