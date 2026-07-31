from pathlib import Path

from huggingface_hub import hf_hub_download


REPOSITORY_ID = "AmazonScience/XRAG"
DATASET_FILENAME = "xrag.dev.jsonl"


def main() -> None:
    """Download the XRAG development dataset into the data folder."""

    project_folder = Path(__file__).resolve().parents[1]
    data_folder = project_folder / "data"
    data_folder.mkdir(parents=True, exist_ok=True)

    print("Downloading the XRAG development dataset...")

    downloaded_file = hf_hub_download(
        repo_id=REPOSITORY_ID,
        repo_type="dataset",
        filename=DATASET_FILENAME,
        local_dir=data_folder,
    )

    print("\nDownload complete!")
    print(f"Dataset saved at: {downloaded_file}")


if __name__ == "__main__":
    main()