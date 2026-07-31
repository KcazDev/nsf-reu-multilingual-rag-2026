import json
from collections import Counter
from pathlib import Path


def main() -> None:
    project_folder = Path(__file__).resolve().parents[1]
    dataset_path = project_folder / "data" / "xrag.dev.jsonl"

    language_counts = Counter()
    task_counts = Counter()
    question_type_counts = Counter()
    crossdoc_type_counts = Counter()
    article_count_counts = Counter()

    total_examples = 0

    with dataset_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                example = json.loads(line)
            except json.JSONDecodeError as error:
                print(f"Could not read line {line_number}: {error}")
                continue

            total_examples += 1

            language_counts[example["language"]] += 1
            task_counts[example["task"]] += 1
            question_type_counts[example["q_type"]] += 1
            crossdoc_type_counts[example["crossdoc_type"]] += 1
            article_count_counts[len(example["articles"])] += 1

    print("XRAG DEVELOPMENT DATASET SUMMARY")
    print("=" * 50)

    print(f"\nTotal examples: {total_examples}")

    print("\nExamples by language:")
    for language, count in sorted(language_counts.items()):
        print(f"  {language}: {count}")

    print("\nExamples by retrieval task:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count}")

    print("\nExamples by question type:")
    for question_type, count in sorted(question_type_counts.items()):
        print(f"  {question_type}: {count}")

    print("\nExamples by cross-document type:")
    for crossdoc_type, count in sorted(crossdoc_type_counts.items()):
        print(f"  {crossdoc_type}: {count}")

    print("\nNumber of articles per example:")
    for article_count, count in sorted(article_count_counts.items()):
        print(f"  {article_count} articles: {count} examples")


if __name__ == "__main__":
    main()