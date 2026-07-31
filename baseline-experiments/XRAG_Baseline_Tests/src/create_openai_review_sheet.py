import csv
import json
from pathlib import Path


CONDITION_ORDER = {
    "no_documents": 1,
    "support_only": 2,
    "all_documents": 3,
}


def main() -> None:
    project_folder = Path(__file__).resolve().parents[1]

    input_path = (
        project_folder
        / "outputs"
        / "openai_pilot_three_conditions_results.jsonl"
    )

    output_path = (
        project_folder
        / "outputs"
        / "openai_pilot_results_review.csv"
    )

    rubric_path = (
        project_folder
        / "notes"
        / "openai_manual_scoring_rubric.txt"
    )

    records = []

    with input_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    records.sort(
        key=lambda record: (
            record["example_number"],
            CONDITION_ORDER[record["condition"]],
        )
    )

    fieldnames = [
        "example_number",
        "xrag_id",
        "language",
        "task",
        "question_type",
        "crossdoc_type",
        "condition",
        "question",
        "gold_answer",
        "model_answer",
        "correctness_score",
        "refusal",
        "correct_language",
        "unsupported_information",
        "review_notes",
        "input_tokens",
        "output_tokens",
    ]

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in records:
            writer.writerow(
                {
                    "example_number": record["example_number"],
                    "xrag_id": record["xrag_id"],
                    "language": record["language"],
                    "task": record["task"],
                    "question_type": record["question_type"],
                    "crossdoc_type": record["crossdoc_type"],
                    "condition": record["condition"],
                    "question": record["question"],
                    "gold_answer": record["gold_answer"],
                    "model_answer": record["model_answer"],
                    "correctness_score": "",
                    "refusal": "",
                    "correct_language": "",
                    "unsupported_information": "",
                    "review_notes": "",
                    "input_tokens": record["input_tokens"],
                    "output_tokens": record["output_tokens"],
                }
            )

    rubric_text = """XRAG MANUAL SCORING RUBRIC
============================================================

CORRECTNESS SCORE

2 = Fully correct
The response contains all essential information from the gold answer.
Minor wording differences are acceptable.

1 = Partially correct
The response contains some correct information, but misses an important
detail, adds a meaningful error, or is too vague.

0 = Incorrect
The response gives the wrong answer, refuses to answer, does not address
the question, or provides no useful answer.

REFUSAL

yes = The model refuses or avoids answering the factual question.
no = The model attempts to answer the question.

CORRECT LANGUAGE

yes = The response is written in the same language as the question.
no = The response is primarily in another language or mixes languages
in a way that affects answer quality.

UNSUPPORTED INFORMATION

yes = The response adds claims not supported by the gold answer or
relevant retrieved documents.
no = The response stays within the supported facts.

REVIEW NOTES

Briefly explain why the response received its score. Mention missing
facts, incorrect facts, refusals, language problems, or distractor use.
"""

    rubric_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with rubric_path.open(
        "w",
        encoding="utf-8",
    ) as rubric_file:
        rubric_file.write(rubric_text)

    print("XRAG REVIEW FILES CREATED")
    print("=" * 70)
    print(f"Results included: {len(records)}")
    print("Review spreadsheet:")
    print(output_path)
    print("\nScoring rubric:")
    print(rubric_path)


if __name__ == "__main__":
    main()