import json
from pathlib import Path


def main() -> None:
    project_folder = Path(__file__).resolve().parents[1]
    dataset_path = project_folder / "data" / "xrag.dev.jsonl"

    with dataset_path.open("r", encoding="utf-8") as file:
        first_line = file.readline()

    first_example = json.loads(first_line)

    print("XRAG DATASET INSPECTION")
    print("=" * 50)

    print("\nAvailable fields:")
    for field_name, value in first_example.items():
        print(f"{field_name}: {type(value).__name__}")

    print("\nQuestion:")
    print(first_example["question"])

    print("\nGold answer:")
    print(first_example["answer"])

    print("\nLanguage:")
    print(first_example["language"])

    print("\nTask:")
    print(first_example["task"])

    print("\nQuestion type:")
    print(first_example["q_type"])

    print("\nCross-document type:")
    print(first_example["crossdoc_type"])

    print("\nNumber of articles:")
    print(len(first_example["articles"]))

    print("\nSupport labels:")
    print(first_example["is_support"])


if __name__ == "__main__":
    main()