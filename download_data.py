from huggingface_hub import hf_hub_download

REPO_ID = "chaouin/coptic-french-translation-data"
PATH_LIST_FILE = "all_data_paths.txt"

# Load and clean paths from the txt file
with open(PATH_LIST_FILE, "r", encoding="utf-8") as f:
    paths = [
        line.strip().lstrip("./")
        for line in f
        if line.strip().endswith(".csv") or line.strip().endswith(".json")
    ]

# Download each file
for path in paths:
    try:
        print(f"⬇️ Downloading: {path}")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=path,
            repo_type="dataset",
            local_dir="."
        )
    except Exception as e:
        print(f"⚠️ Failed to download {path}: {e}")
