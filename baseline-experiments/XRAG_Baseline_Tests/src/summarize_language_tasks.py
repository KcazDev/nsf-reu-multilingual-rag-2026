import json
from collections import Counter
from pathlib import Path


def main() -> None:
    project_folder = Path(__file__).resolve().parents[1]
    dataset_path = project_folder / "data" / "xrag.dev.jsonl"

    group_counts = Counter()

    with dataset_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            example = json.loads(line)

            group = (
                example["language"],
                example["task"],
            )

            group_counts[group] += 1

    print("XRAG LANGUAGE AND TASK GROUPS")
    print("=" * 50)

    for (language, task), count in sorted(group_counts.items()):
        print(
            f"Language: {language:>2} | "
            f"Task: {task:<16} | "
            f"Examples: {count}"
        )


if __name__ == "__main__":
    main()