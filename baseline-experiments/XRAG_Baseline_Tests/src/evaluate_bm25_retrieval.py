import csv
import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PILOT_FILE = (
    PROJECT_ROOT
    / "data"
    / "xrag_pilot_sample.jsonl"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "bm25_retrieval_results.csv"
)


def load_jsonl(file_path: Path) -> list[dict]:
    """Load JSON objects from a JSONL file."""
    examples = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

    return examples


def get_article_text(article: Any) -> str:
    """Return readable text from a string or article dictionary."""
    if isinstance(article, str):
        return article

    if isinstance(article, dict):
        possible_fields = [
            "title",
            "text",
            "content",
            "article",
            "body",
        ]

        text_parts = []

        for field in possible_fields:
            value = article.get(field)

            if isinstance(value, str) and value.strip():
                text_parts.append(value.strip())

        return " ".join(text_parts)

    return str(article)


def tokenize(text: str) -> list[str]:
    """
    Create simple lowercase tokens.

    This tokenizer supports Unicode characters, including
    Spanish and Arabic text.
    """
    return re.findall(
        r"\w+",
        text.lower(),
        flags=re.UNICODE,
    )


def rank_articles(
    question: str,
    articles: list[Any],
) -> list[tuple[int, float]]:
    """
    Rank articles using BM25.

    Returns article indexes using Python's zero-based indexing.
    """
    article_texts = [
        get_article_text(article)
        for article in articles
    ]

    tokenized_articles = [
        tokenize(article_text)
        for article_text in article_texts
    ]

    tokenized_question = tokenize(question)

    bm25 = BM25Okapi(tokenized_articles)
    scores = bm25.get_scores(tokenized_question)

    ranked_articles = sorted(
        enumerate(scores),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        (article_index, float(score))
        for article_index, score in ranked_articles
    ]


def get_support_articles(
    support_labels: list[Any],
) -> list[int]:
    """
    Return supporting article numbers using human-readable
    one-based numbering.
    """
    support_articles = []

    for index, label in enumerate(support_labels):
        try:
            is_support = float(label) > 0
        except (TypeError, ValueError):
            is_support = False

        if is_support:
            support_articles.append(index + 1)

    return support_articles


def calculate_metrics(
    ranked_article_numbers: list[int],
    true_support_articles: list[int],
    k: int,
) -> tuple[float, float, bool]:
    """Calculate Precision@K, Recall@K, and complete support coverage."""
    top_k = ranked_article_numbers[:k]

    relevant_found = [
        article_number
        for article_number in top_k
        if article_number in true_support_articles
    ]

    precision = len(relevant_found) / k

    recall = (
        len(relevant_found) / len(true_support_articles)
        if true_support_articles
        else 0.0
    )

    all_support_found = set(true_support_articles).issubset(
        set(top_k)
    )

    return precision, recall, all_support_found


def calculate_reciprocal_rank(
    ranked_article_numbers: list[int],
    true_support_articles: list[int],
) -> float:
    """
    Calculate the reciprocal rank of the first supporting article.
    """
    for rank_position, article_number in enumerate(
        ranked_article_numbers,
        start=1,
    ):
        if article_number in true_support_articles:
            return 1.0 / rank_position

    return 0.0


def main() -> None:
    if not PILOT_FILE.exists():
        raise FileNotFoundError(
            f"Pilot dataset was not found:\n{PILOT_FILE}"
        )

    examples = load_jsonl(PILOT_FILE)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for example_number, example in enumerate(
        examples,
        start=1,
    ):
        question = example.get("question", "")
        articles = example.get("articles", [])
        support_labels = example.get("is_support", [])

        ranked_articles = rank_articles(
            question=question,
            articles=articles,
        )

        ranked_article_numbers = [
            article_index + 1
            for article_index, score in ranked_articles
        ]

        ranked_scores = [
            round(score, 4)
            for article_index, score in ranked_articles
        ]

        true_support_articles = get_support_articles(
            support_labels
        )

        precision_at_1, recall_at_1, all_support_at_1 = (
            calculate_metrics(
                ranked_article_numbers,
                true_support_articles,
                k=1,
            )
        )

        precision_at_2, recall_at_2, all_support_at_2 = (
            calculate_metrics(
                ranked_article_numbers,
                true_support_articles,
                k=2,
            )
        )

        precision_at_3, recall_at_3, all_support_at_3 = (
            calculate_metrics(
                ranked_article_numbers,
                true_support_articles,
                k=3,
            )
        )

        reciprocal_rank = calculate_reciprocal_rank(
            ranked_article_numbers,
            true_support_articles,
        )

        results.append(
            {
                "example_number": example_number,
                "xrag_id": example.get("id"),
                "language": example.get("language"),
                "task": example.get("task"),
                "question_type": example.get("q_type"),
                "crossdoc_type": example.get("crossdoc_type"),
                "question": question,
                "true_support_articles": ",".join(
                    map(str, true_support_articles)
                ),
                "bm25_full_ranking": ",".join(
                    map(str, ranked_article_numbers)
                ),
                "bm25_scores": ",".join(
                    map(str, ranked_scores)
                ),
                "top_1_article": ranked_article_numbers[0],
                "top_2_articles": ",".join(
                    map(str, ranked_article_numbers[:2])
                ),
                "top_3_articles": ",".join(
                    map(str, ranked_article_numbers[:3])
                ),
                "precision_at_1": precision_at_1,
                "recall_at_1": recall_at_1,
                "precision_at_2": precision_at_2,
                "recall_at_2": recall_at_2,
                "precision_at_3": precision_at_3,
                "recall_at_3": recall_at_3,
                "all_support_at_1": all_support_at_1,
                "all_support_at_2": all_support_at_2,
                "all_support_at_3": all_support_at_3,
                "reciprocal_rank": reciprocal_rank,
            }
        )

    fieldnames = list(results[0].keys())

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    number_of_examples = len(results)

    average_precision_at_1 = sum(
        result["precision_at_1"]
        for result in results
    ) / number_of_examples

    average_recall_at_1 = sum(
        result["recall_at_1"]
        for result in results
    ) / number_of_examples

    average_precision_at_2 = sum(
        result["precision_at_2"]
        for result in results
    ) / number_of_examples

    average_recall_at_2 = sum(
        result["recall_at_2"]
        for result in results
    ) / number_of_examples

    average_precision_at_3 = sum(
        result["precision_at_3"]
        for result in results
    ) / number_of_examples

    average_recall_at_3 = sum(
        result["recall_at_3"]
        for result in results
    ) / number_of_examples

    mean_reciprocal_rank = sum(
        result["reciprocal_rank"]
        for result in results
    ) / number_of_examples

    complete_support_at_2 = sum(
        result["all_support_at_2"]
        for result in results
    )

    complete_support_at_3 = sum(
        result["all_support_at_3"]
        for result in results
    )

    print("BM25 retrieval evaluation completed.")
    print(f"Examples evaluated: {number_of_examples}")
    print(f"Results saved to: {OUTPUT_FILE}")

    print("\nAverage retrieval metrics:")
    print(f"Precision@1: {average_precision_at_1:.3f}")
    print(f"Recall@1: {average_recall_at_1:.3f}")
    print(f"Precision@2: {average_precision_at_2:.3f}")
    print(f"Recall@2: {average_recall_at_2:.3f}")
    print(f"Precision@3: {average_precision_at_3:.3f}")
    print(f"Recall@3: {average_recall_at_3:.3f}")
    print(f"Mean Reciprocal Rank: {mean_reciprocal_rank:.3f}")

    print("\nComplete supporting-document coverage:")
    print(
        f"All support found in top 2: "
        f"{complete_support_at_2}/{number_of_examples}"
    )
    print(
        f"All support found in top 3: "
        f"{complete_support_at_3}/{number_of_examples}"
    )


if __name__ == "__main__":
    main()