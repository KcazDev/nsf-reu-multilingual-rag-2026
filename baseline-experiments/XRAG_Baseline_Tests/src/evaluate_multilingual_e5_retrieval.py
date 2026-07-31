import json
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PILOT_FILE = PROJECT_ROOT / "data" / "xrag_pilot_sample.jsonl"

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
    """Return article text from either a string or dictionary."""
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
    """
    Return the human-readable one-based article numbers
    marked as supporting.
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


def chunk_article_text(
    tokenizer: Any,
    article_text: str,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """
    Split an article into overlapping token-based chunks.

    Each chunk remains below the E5 model's 512-token limit
    after the passage prefix and special tokens are added.
    """
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if overlap < 0:
        raise ValueError("Chunk overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be smaller than chunk size."
        )

    token_ids = tokenizer.encode(
        article_text,
        add_special_tokens=False,
        truncation=False,
    )

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
) -> tuple[list[tuple[int, float]], list[int]]:
    """
    Rank articles using Multilingual E5.

    Each article is divided into overlapping chunks. The article's
    final score is the similarity score of its best matching chunk.

    Returns:
        ranked_articles:
            List of zero-based article indexes and similarity scores.

        chunks_per_article:
            Number of chunks generated for each article.
    """
    if not articles:
        return [], []

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

    if not all_chunk_texts:
        return [], chunks_per_article

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


def main() -> None:
    if not PILOT_FILE.exists():
        raise FileNotFoundError(
            f"Pilot dataset was not found:\n{PILOT_FILE}"
        )

    examples = load_jsonl(PILOT_FILE)

    if not examples:
        print("No XRAG examples were found.")
        return

    print(f"Loading model: {MODEL_NAME}")
    print("The first model download may take several minutes.")

    model = SentenceTransformer(MODEL_NAME)

    print("Model loaded successfully.")
    print(f"Examples loaded: {len(examples)}")

    first_example = examples[0]

    question = first_example.get("question", "")
    articles = first_example.get("articles", [])
    support_labels = first_example.get("is_support", [])

    if not question:
        raise ValueError(
            "The first XRAG example does not contain a question."
        )

    if not articles:
        raise ValueError(
            "The first XRAG example does not contain articles."
        )

    ranked_articles, chunks_per_article = (
        rank_articles_with_e5(
            model=model,
            question=question,
            articles=articles,
        )
    )

    true_support_articles = get_support_articles(
        support_labels
    )

    top_two_articles = [
        article_index + 1
        for article_index, score in ranked_articles[:2]
    ]

    supporting_articles_found = [
        article_number
        for article_number in top_two_articles
        if article_number in true_support_articles
    ]

    precision_at_2 = (
        len(supporting_articles_found) / 2
        if len(top_two_articles) == 2
        else 0.0
    )

    recall_at_2 = (
        len(supporting_articles_found)
        / len(true_support_articles)
        if true_support_articles
        else 0.0
    )

    all_support_found = set(
        true_support_articles
    ).issubset(
        set(top_two_articles)
    )

    print("\nFirst XRAG example:")
    print(f"ID: {first_example.get('id')}")
    print(f"Language: {first_example.get('language')}")
    print(f"Question: {question}")
    print(f"Number of articles: {len(articles)}")

    print("\nChunk information:")
    print(
        f"Chunk size: {CHUNK_SIZE_TOKENS} tokens"
    )
    print(
        f"Chunk overlap: {CHUNK_OVERLAP_TOKENS} tokens"
    )

    for article_number, chunk_count in enumerate(
        chunks_per_article,
        start=1,
    ):
        print(
            f"Article {article_number}: "
            f"{chunk_count} chunks"
        )

    print("\nChunked Multilingual E5 article ranking:")

    for rank_position, (
        article_index,
        score,
    ) in enumerate(
        ranked_articles,
        start=1,
    ):
        print(
            f"Rank {rank_position}: "
            f"Article {article_index + 1} | "
            f"Similarity score = {score:.4f}"
        )

    print("\nSupport comparison:")
    print(
        f"True supporting articles: "
        f"{true_support_articles}"
    )
    print(
        f"E5 top two articles: "
        f"{top_two_articles}"
    )
    print(
        f"Supporting articles found: "
        f"{supporting_articles_found}"
    )
    print(f"Precision@2: {precision_at_2:.2f}")
    print(f"Recall@2: {recall_at_2:.2f}")
    print(
        f"All supporting articles found: "
        f"{all_support_found}"
    )


if __name__ == "__main__":
    main()