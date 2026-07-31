import csv
import json
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PILOT_FILE = (
    PROJECT_ROOT
    / "data"
    / "xrag_pilot_sample.jsonl"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "chunked_e5_retrieval_results.csv"
)

MODEL_NAME = "intfloat/multilingual-e5-small"

CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 80


def load_jsonl(file_path: Path) -> list[dict]:
    """Load all JSON objects from a JSONL file."""
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
    """Return readable article text."""
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


def get_support_articles(
    support_labels: list[Any],
) -> list[int]:
    """Return one-based supporting article numbers."""
    support_articles = []

    for index, label in enumerate(support_labels):
        try:
            is_support = float(label) > 0
        except (TypeError, ValueError):
            is_support = False

        if is_support:
            support_articles.append(index + 1)

    return support_articles


def chunk_article_text(
    tokenizer: Any,
    article_text: str,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Split one article into overlapping token chunks."""
    if chunk_size <= 0:
        raise ValueError(
            "Chunk size must be greater than zero."
        )

    if overlap < 0:
        raise ValueError(
            "Chunk overlap cannot be negative."
        )

    if overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be smaller than chunk size."
        )

    tokenized_article = tokenizer(
        article_text,
        add_special_tokens=False,
        truncation=False,
        verbose=False,
    )

    token_ids = tokenized_article["input_ids"]

    if not token_ids:
        return [""]

    step_size = chunk_size - overlap
    chunks = []

    for start_index in range(
        0,
        len(token_ids),
        step_size,
    ):
        end_index = start_index + chunk_size

        chunk_token_ids = token_ids[
            start_index:end_index
        ]

        chunk_text = tokenizer.decode(
            chunk_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        chunks.append(chunk_text)

        if end_index >= len(token_ids):
            break

    return chunks


def rank_articles_with_e5(
    model: SentenceTransformer,
    question: str,
    articles: list[Any],
) -> tuple[
    list[tuple[int, float]],
    list[int],
]:
    """
    Rank each article using its highest-scoring chunk.

    Returns:
        Ranked zero-based article indexes and scores.
        Number of chunks created for each article.
    """
    query_text = f"query: {question}"

    query_embedding = model.encode(
        [query_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]

    all_chunk_texts = []
    chunk_article_indexes = []
    chunks_per_article = []

    for article_index, article in enumerate(articles):
        article_text = get_article_text(article)

        article_chunks = chunk_article_text(
            tokenizer=model.tokenizer,
            article_text=article_text,
        )

        chunks_per_article.append(
            len(article_chunks)
        )

        for chunk in article_chunks:
            all_chunk_texts.append(
                f"passage: {chunk}"
            )

            chunk_article_indexes.append(
                article_index
            )

    chunk_embeddings = model.encode(
        all_chunk_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=16,
    )

    chunk_scores = (
        chunk_embeddings @ query_embedding
    )

    article_scores = [
        float("-inf")
        for _ in articles
    ]

    for article_index, score in zip(
        chunk_article_indexes,
        chunk_scores,
    ):
        numeric_score = float(score)

        if numeric_score > article_scores[article_index]:
            article_scores[article_index] = numeric_score

    ranked_articles = sorted(
        enumerate(article_scores),
        key=lambda item: item[1],
        reverse=True,
    )

    return ranked_articles, chunks_per_article


def calculate_metrics(
    ranked_article_numbers: list[int],
    true_support_articles: list[int],
    k: int,
) -> tuple[float, float, bool]:
    """Calculate Precision@K, Recall@K, and complete coverage."""
    top_k = ranked_article_numbers[:k]

    relevant_found = [
        article_number
        for article_number in top_k
        if article_number in true_support_articles
    ]

    precision = (
        len(relevant_found) / k
    )

    recall = (
        len(relevant_found)
        / len(true_support_articles)
        if true_support_articles
        else 0.0
    )

    all_support_found = set(
        true_support_articles
    ).issubset(
        set(top_k)
    )

    return precision, recall, all_support_found


def calculate_reciprocal_rank(
    ranked_article_numbers: list[int],
    true_support_articles: list[int],
) -> float:
    """Calculate reciprocal rank of the first support article."""
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

    if not examples:
        print("No XRAG examples were found.")
        return

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Loading model: {MODEL_NAME}")

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("Model loaded successfully.")
    print(f"Examples to evaluate: {len(examples)}")

    results = []

    for example_number, example in enumerate(
        examples,
        start=1,
    ):
        question = example.get("question", "")
        articles = example.get("articles", [])
        support_labels = example.get(
            "is_support",
            [],
        )

        print(
            f"Evaluating example "
            f"{example_number}/{len(examples)}..."
        )

        ranked_articles, chunks_per_article = (
            rank_articles_with_e5(
                model=model,
                question=question,
                articles=articles,
            )
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
                "crossdoc_type": example.get(
                    "crossdoc_type"
                ),
                "question": question,
                "true_support_articles": ",".join(
                    map(str, true_support_articles)
                ),
                "e5_full_ranking": ",".join(
                    map(str, ranked_article_numbers)
                ),
                "e5_scores": ",".join(
                    map(str, ranked_scores)
                ),
                "chunks_per_article": ",".join(
                    map(str, chunks_per_article)
                ),
                "top_1_article": ranked_article_numbers[0],
                "top_2_articles": ",".join(
                    map(
                        str,
                        ranked_article_numbers[:2],
                    )
                ),
                "top_3_articles": ",".join(
                    map(
                        str,
                        ranked_article_numbers[:3],
                    )
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

    fieldnames = list(
        results[0].keys()
    )

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
        int(result["all_support_at_2"])
        for result in results
    )

    complete_support_at_3 = sum(
        int(result["all_support_at_3"])
        for result in results
    )

    print("\nChunked E5 evaluation completed.")
    print(
        f"Examples evaluated: "
        f"{number_of_examples}"
    )
    print(
        f"Results saved to: "
        f"{OUTPUT_FILE}"
    )

    print("\nAverage retrieval metrics:")
    print(
        f"Precision@1: "
        f"{average_precision_at_1:.3f}"
    )
    print(
        f"Recall@1: "
        f"{average_recall_at_1:.3f}"
    )
    print(
        f"Precision@2: "
        f"{average_precision_at_2:.3f}"
    )
    print(
        f"Recall@2: "
        f"{average_recall_at_2:.3f}"
    )
    print(
        f"Precision@3: "
        f"{average_precision_at_3:.3f}"
    )
    print(
        f"Recall@3: "
        f"{average_recall_at_3:.3f}"
    )
    print(
        f"Mean Reciprocal Rank: "
        f"{mean_reciprocal_rank:.3f}"
    )

    print("\nComplete supporting-document coverage:")
    print(
        f"All support found in top 2: "
        f"{complete_support_at_2}/"
        f"{number_of_examples}"
    )
    print(
        f"All support found in top 3: "
        f"{complete_support_at_3}/"
        f"{number_of_examples}"
    )


if __name__ == "__main__":
    main()