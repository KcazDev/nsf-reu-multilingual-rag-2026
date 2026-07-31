import json
from collections import Counter
from pathlib import Path


TARGET_LANGUAGES = {"ar", "es"}
TARGET_TASKS = {"mono_retrieval", "multi_retrieval"}


def main() -> None:
    project_folder = Path(__file__).resolve().parents[1]
    dataset_path = project_folder / "data" / "xrag.dev.jsonl"

    group_counts = Counter()

    with dataset_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            example = json.loads(line)

            language = example["language"]
            task = example["task"]
            crossdoc_type = example["crossdoc_type"]

            if (
                language in TARGET_LANGUAGES
                and task in TARGET_TASKS
            ):
                group_counts[
                    (language, task, crossdoc_type)
                ] += 1

    print("XRAG PILOT GROUP COUNTS")
    print("=" * 60)

    for language in sorted(TARGET_LANGUAGES):
        for task in sorted(TARGET_TASKS):
            print(f"\nLanguage: {language} | Task: {task}")

            for crossdoc_type in [0, 1, 2]:
                count = group_counts[
                    (language, task, crossdoc_type)
                ]

                print(
                    f"  Cross-document type {crossdoc_type}: "
                    f"{count} examples"
                )


if __name__ == "__main__":
    main()