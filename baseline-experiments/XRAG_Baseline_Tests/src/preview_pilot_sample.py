import json
from pathlib import Path
from typing import Any


ARTICLE_PREVIEW_LENGTH = 700


def load_jsonl(file_path: Path) -> list[dict[str, Any]]:
    examples = []

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                examples.append(json.loads(line))

    return examples


def clean_text(text: str) -> str:
    """Replace extra whitespace so article previews are easier to read."""
    return " ".join(text.split())


def main() -> None:
    project_folder = Path(__file__).resolve().parents[1]

    input_path = (
        project_folder
        / "data"
        / "xrag_pilot_sample.jsonl"
    )

    output_path = (
        project_folder
        / "notes"
        / "pilot_sample_preview.txt"
    )

    examples = load_jsonl(input_path)

    with output_path.open("w", encoding="utf-8") as output_file:
        output_file.write("XRAG PILOT SAMPLE PREVIEW\n")
        output_file.write("=" * 90 + "\n")
        output_file.write(
            f"Total examples: {len(examples)}\n\n"
        )

        for example_number, example in enumerate(
            examples,
            start=1,
        ):
            output_file.write("\n" + "#" * 90 + "\n")
            output_file.write(
                f"EXAMPLE {example_number} OF {len(examples)}\n"
            )
            output_file.write("#" * 90 + "\n")

            output_file.write(
                f"XRAG ID: {example['id']}\n"
            )
            output_file.write(
                f"Language: {example['language']}\n"
            )
            output_file.write(
                f"Task: {example['task']}\n"
            )
            output_file.write(
                f"Question type: {example['q_type']}\n"
            )
            output_file.write(
                f"Answer type: {example['a_type']}\n"
            )
            output_file.write(
                f"Cross-document type: "
                f"{example['crossdoc_type']}\n"
            )
            output_file.write(
                f"Article order: {example['article_order']}\n"
            )
            output_file.write(
                f"Support labels: {example['is_support']}\n"
            )

            output_file.write("\nQUESTION:\n")
            output_file.write(
                example["question"] + "\n"
            )

            output_file.write("\nGOLD ANSWER:\n")
            output_file.write(
                example["answer"] + "\n"
            )

            output_file.write("\nENGLISH QUESTION:\n")
            output_file.write(
                example["en_question"] + "\n"
            )

            output_file.write("\nENGLISH GOLD ANSWER:\n")
            output_file.write(
                example["en_answer"] + "\n"
            )

            output_file.write("\nARTICLES:\n")

            for article_index, article_text in enumerate(
                example["articles"]
            ):
                cleaned_article = clean_text(article_text)

                preview = cleaned_article[
                    :ARTICLE_PREVIEW_LENGTH
                ]

                if len(cleaned_article) > ARTICLE_PREVIEW_LENGTH:
                    preview += "..."

                support_label = example["is_support"][
                    article_index
                ]

                article_date = example["dates"][
                    article_index
                ]

                article_url = example["urls"][
                    article_index
                ]

                output_file.write(
                    "\n"
                    + "-" * 80
                    + "\n"
                )

                output_file.write(
                    f"Stored article index: {article_index}\n"
                )
                output_file.write(
                    f"Support label: {support_label}\n"
                )
                output_file.write(
                    f"Publication date: {article_date}\n"
                )
                output_file.write(
                    f"URL: {article_url}\n"
                )
                output_file.write(
                    f"Preview:\n{preview}\n"
                )

    print("PILOT SAMPLE PREVIEW CREATED")
    print("=" * 60)
    print(f"Examples included: {len(examples)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()