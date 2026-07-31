import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


RANDOM_SEED = 42

SELECTION_PLAN = {
    ("ar", "mono_retrieval", 0): 3,
    ("ar", "mono_retrieval", 1): 2,

    ("ar", "multi_retrieval", 0): 2,
    ("ar", "multi_retrieval", 1): 1,
    ("ar", "multi_retrieval", 2): 2,

    ("es", "mono_retrieval", 0): 3,
    ("es", "mono_retrieval", 1): 2,

    ("es", "multi_retrieval", 0): 2,
    ("es", "multi_retrieval", 1): 1,
    ("es", "multi_retrieval", 2): 2,
}


def read_jsonl(dataset_path: Path) -> list[dict[str, Any]]:
    examples = []

    with dataset_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                examples.append(json.loads(line))

    return examples


def main() -> None:
    random.seed(RANDOM_SEED)

    project_folder = Path(__file__).resolve().parents[1]
    input_path = project_folder / "data" / "xrag.dev.jsonl"
    output_path = project_folder / "data" / "xrag_pilot_sample.jsonl"

    examples = read_jsonl(input_path)

    grouped_examples = defaultdict(list)

    for example in examples:
        group_key = (
            example["language"],
            example["task"],
            example["crossdoc_type"],
        )

        grouped_examples[group_key].append(example)

    selected_examples = []

    for group_key, requested_count in SELECTION_PLAN.items():
        available_examples = grouped_examples[group_key]

        if len(available_examples) < requested_count:
            raise ValueError(
                f"Not enough examples in group {group_key}. "
                f"Requested {requested_count}, "
                f"but found {len(available_examples)}."
            )

        chosen_examples = random.sample(
            available_examples,
            requested_count,
        )

        selected_examples.extend(chosen_examples)

    random.shuffle(selected_examples)

    with output_path.open("w", encoding="utf-8") as file:
        for example in selected_examples:
            file.write(
                json.dumps(example, ensure_ascii=False) + "\n"
            )

    print("XRAG PILOT SAMPLE CREATED")
    print("=" * 60)

    print(f"\nRandom seed: {RANDOM_SEED}")
    print(f"Total selected examples: {len(selected_examples)}")
    print(f"Saved to: {output_path}")

    group_summary = Counter(
        (
            example["language"],
            example["task"],
            example["crossdoc_type"],
        )
        for example in selected_examples
    )

    print("\nSelected examples by group:")

    for group_key, count in sorted(group_summary.items()):
        language, task, crossdoc_type = group_key

        print(
            f"Language: {language} | "
            f"Task: {task} | "
            f"Cross-document type: {crossdoc_type} | "
            f"Count: {count}"
        )

    question_type_summary = Counter(
        example["q_type"]
        for example in selected_examples
    )

    print("\nQuestion types in pilot sample:")

    for question_type, count in sorted(question_type_summary.items()):
        print(f"{question_type}: {count}")

    print("\nSelected XRAG IDs:")

    for example in selected_examples:
        print(
            f"ID: {example['id']} | "
            f"Language: {example['language']} | "
            f"Task: {example['task']} | "
            f"Question type: {example['q_type']} | "
            f"Cross-document type: {example['crossdoc_type']}"
        )


if __name__ == "__main__":
    main()