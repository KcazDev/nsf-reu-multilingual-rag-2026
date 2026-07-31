import json
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_EXAMPLES = 20
EXPECTED_CONDITIONS = {
    "no_documents",
    "support_only",
    "all_documents",
}


def main() -> None:
    project_folder = Path(__file__).resolve().parents[1]

    results_path = (
        project_folder
        / "outputs"
        / "sonnet_pilot_three_conditions_results.jsonl"
    )

    records = []

    with results_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                print(
                    f"Could not read line {line_number}: {error}"
                )

    condition_counts = Counter(
        record["condition"]
        for record in records
    )

    language_counts = Counter(
        record["language"]
        for record in records
    )

    results_by_example = defaultdict(set)

    for record in records:
        results_by_example[
            record["example_number"]
        ].add(record["condition"])

    incomplete_examples = []

    for example_number in range(
        1,
        EXPECTED_EXAMPLES + 1,
    ):
        completed_conditions = results_by_example[
            example_number
        ]

        missing_conditions = (
            EXPECTED_CONDITIONS
            - completed_conditions
        )

        if missing_conditions:
            incomplete_examples.append(
                (
                    example_number,
                    sorted(missing_conditions),
                )
            )

    total_input_tokens = sum(
        record["input_tokens"]
        for record in records
    )

    total_output_tokens = sum(
        record["output_tokens"]
        for record in records
    )

    stop_reason_counts = Counter(
        record["stop_reason"]
        for record in records
    )

    print("XRAG PILOT RESULTS VERIFICATION")
    print("=" * 70)

    print(f"\nTotal result records: {len(records)}")
    print(
        f"Expected result records: "
        f"{EXPECTED_EXAMPLES * 3}"
    )

    print("\nResults by condition:")

    for condition in sorted(EXPECTED_CONDITIONS):
        print(
            f"  {condition}: "
            f"{condition_counts[condition]}"
        )

    print("\nResults by language:")

    for language, count in sorted(
        language_counts.items()
    ):
        print(f"  {language}: {count}")

    print("\nStop reasons:")

    for stop_reason, count in sorted(
        stop_reason_counts.items()
    ):
        print(f"  {stop_reason}: {count}")

    print("\nToken usage:")
    print(f"  Total input tokens: {total_input_tokens}")
    print(f"  Total output tokens: {total_output_tokens}")
    print(
        f"  Total combined tokens: "
        f"{total_input_tokens + total_output_tokens}"
    )

    if incomplete_examples:
        print("\nINCOMPLETE EXAMPLES FOUND:")

        for example_number, missing in incomplete_examples:
            print(
                f"  Example {example_number} is missing: "
                f"{', '.join(missing)}"
            )
    else:
        print(
            "\nAll 20 examples contain all "
            "three conditions."
        )

    if len(records) == EXPECTED_EXAMPLES * 3:
        print("\nVERIFICATION STATUS: COMPLETE")
    else:
        print("\nVERIFICATION STATUS: CHECK REQUIRED")


if __name__ == "__main__":
    main()