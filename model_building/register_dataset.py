
from pathlib import Path

from huggingface_hub import HfApi, upload_file


DATASET_REPO = "Greeshmadudu/tourism-dataset"
DATA_FILE = Path("data/tourism.csv")


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    api = HfApi()
    api.create_repo(repo_id=DATASET_REPO, repo_type="dataset", exist_ok=True)
    upload_file(
        path_or_fileobj=str(DATA_FILE),
        path_in_repo="tourism.csv",
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )
    print("Dataset uploaded successfully.")


if __name__ == "__main__":
    main()
