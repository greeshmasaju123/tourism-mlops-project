
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, upload_file
from sklearn.model_selection import train_test_split


DATASET_REPO = "Greeshmadudu/tourism-dataset"
LOCAL_DATA = Path("data/tourism.csv")
OUTPUT_DIR = Path("artifacts")


def load_data():
    if LOCAL_DATA.exists():
        return pd.read_csv(LOCAL_DATA)

    downloaded = hf_hub_download(
        repo_id=DATASET_REPO,
        filename="tourism.csv",
        repo_type="dataset",
    )
    return pd.read_csv(downloaded)


def clean_data(df):
    df = df.copy()
    df = df.drop(columns=[col for col in ["Unnamed: 0", "", "CustomerID"] if col in df.columns])
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})
    return df


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = clean_data(load_data())

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["ProdTaken"],
    )

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

    for filename in ["train.csv", "test.csv"]:
        upload_file(
            path_or_fileobj=str(OUTPUT_DIR / filename),
            path_in_repo=filename,
            repo_id=DATASET_REPO,
            repo_type="dataset",
        )
    print("Data preparation completed.")


if __name__ == "__main__":
    main()
