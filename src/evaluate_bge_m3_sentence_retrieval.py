import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

import nltk
import numpy as np
import torch
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------
# Project paths and default settings
# ---------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"

DEFAULT_SAMPLE_FILE = DATA_DIR / "xrag_eval_sample_50.jsonl"

DEFAULT_CSV_FILE = (
    OUTPUTS_DIR / "bge_m3_sentence_retrieval_eval_50.csv"
)

DEFAULT_SUMMARY_FILE = (
    OUTPUTS_DIR / "bge_m3_sentence_retrieval_summary.txt"
)

DEFAULT_DETAILS_FILE = (
    OUTPUTS_DIR / "bge_m3_sentence_retrieval_details.jsonl"
)

DEFAULT_MODEL_NAME = "BAAI/bge-m3"

# These match the sentence-cap values evaluated in the CROSS paper.
DEFAULT_K_VALUES = [3, 5, 10, 20, 50]

# Sentence-level retrieval normally produces relatively short inputs.
DEFAULT_MAX_SEQUENCE_LENGTH = 512

# A smaller batch size is safer on CPU or systems with limited memory.
DEFAULT_BATCH_SIZE = 8


# ---------------------------------------------------------------------
# General file helpers
# ---------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Load a JSONL file into a list of dictionaries.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    records: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {error}"
                ) from error

    return records


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """
    Append one dictionary to a JSONL file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def remove_file_if_present(path: Path) -> None:
    """
    Delete a file when starting a fresh evaluation.
    """
    if path.exists():
        path.unlink()


def parse_k_values(raw_value: str) -> List[int]:
    """
    Convert a value such as '3,5,10,20,50' into a sorted integer list.
    """
    values: List[int] = []

    for piece in raw_value.split(","):
        piece = piece.strip()

        if not piece:
            continue

        try:
            value = int(piece)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"Invalid Top-K value: {piece}"
            ) from error

        if value <= 0:
            raise argparse.ArgumentTypeError(
                "Every Top-K value must be greater than zero."
            )

        values.append(value)

    if not values:
        raise argparse.ArgumentTypeError(
            "At least one Top-K value is required."
        )

    return sorted(set(values))


# ---------------------------------------------------------------------
# XRAG record helpers
# ---------------------------------------------------------------------

def get_question(record: Dict[str, Any]) -> str:
    """
    Read the question from an XRAG-style record.
    """
    question = (
        record.get("question")
        or record.get("query")
        or record.get("input")
        or ""
    )

    return str(question).strip()


def get_gold_answer(record: Dict[str, Any]) -> Any:
    """
    Read the gold answer from an XRAG-style record.
    """
    return (
        record.get("gold_answer")
        or record.get("answer")
        or record.get("answers")
        or ""
    )


def get_xrag_id(record: Dict[str, Any]) -> Any:
    """
    Read the example identifier.
    """
    return (
        record.get("xrag_id")
        or record.get("id")
        or record.get("_id")
        or record.get("example_id")
    )


def get_language(record: Dict[str, Any]) -> str:
    """
    Read the question language.
    """
    return str(
        record.get("language")
        or record.get("lang")
        or record.get("question_language")
        or ""
    )


def get_question_type(record: Dict[str, Any]) -> str:
    """
    XRAG uses q_type in the copied evaluation sample.
    """
    return str(
        record.get("question_type")
        or record.get("q_type")
        or ""
    )


def get_answer_type(record: Dict[str, Any]) -> str:
    """
    XRAG uses a_type in the copied evaluation sample.
    """
    return str(
        record.get("answer_type")
        or record.get("a_type")
        or ""
    )


def get_articles(record: Dict[str, Any]) -> List[str]:
    """
    Read the article list.

    The current XRAG evaluation sample stores each article as a string.
    This function also supports dictionary-style articles.
    """
    raw_articles = record.get("articles")

    if not isinstance(raw_articles, list) or not raw_articles:
        raise ValueError(
            "The record does not contain a non-empty articles list."
        )

    articles: List[str] = []

    for article in raw_articles:
        if isinstance(article, str):
            text = article

        elif isinstance(article, dict):
            text = (
                article.get("article_text")
                or article.get("text")
                or article.get("content")
                or article.get("body")
                or article.get("document")
                or ""
            )

        else:
            text = str(article)

        articles.append(text.strip())

    return articles


def get_parallel_list(
    record: Dict[str, Any],
    key: str,
    expected_length: int,
) -> List[Any]:
    """
    Read a list such as dates, URLs, or article_order.

    Missing entries are padded with None.
    """
    value = record.get(key)

    if not isinstance(value, list):
        return [None] * expected_length

    result = list(value[:expected_length])

    while len(result) < expected_length:
        result.append(None)

    return result


def is_positive_support_value(value: Any) -> bool:
    """
    XRAG support labels may be numeric, Boolean, or strings.

    The copied sample uses values such as:
        [0, 2, 0, 0, 0, 0, 0, 0]
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value > 0

    if isinstance(value, str):
        normalized = value.strip().lower()

        return normalized in {
            "1",
            "2",
            "true",
            "yes",
            "support",
            "relevant",
            "gold",
        }

    return False


def get_support_article_numbers_one_based(
    record: Dict[str, Any],
    article_count: int,
) -> List[int]:
    """
    Convert the XRAG is_support list into one-based article numbers.
    """
    support_values = record.get("is_support")

    if not isinstance(support_values, list):
        return []

    support_numbers: List[int] = []

    for article_number, value in enumerate(
        support_values[:article_count],
        start=1,
    ):
        if is_positive_support_value(value):
            support_numbers.append(article_number)

    return support_numbers


# ---------------------------------------------------------------------
# Sentence tokenization
# ---------------------------------------------------------------------

def prepare_nltk_resources() -> None:
    """
    Download Punkt resources when they are not already present.

    The script still has a regex fallback if downloading is unavailable.
    """
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]

    for resource_path, download_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(download_name, quiet=True)
            except Exception:
                # The regex fallback will be used if Punkt is unavailable.
                pass


def normalize_whitespace(text: str) -> str:
    """
    Normalize spaces while keeping paragraph boundaries available.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def choose_punkt_language(
    question_language: str,
    article_text: str,
) -> str:
    """
    Choose an available Punkt model.

    NLTK does not provide a dedicated Arabic Punkt package. English Punkt
    still handles common punctuation, and the script has a multilingual
    regex fallback for Arabic question marks and other punctuation.
    """
    normalized_language = question_language.lower().strip()

    if normalized_language in {"es", "spa", "spanish"}:
        return "spanish"

    return "english"


def regex_sentence_split(text: str) -> List[str]:
    """
    Language-flexible fallback sentence splitter.

    It recognizes Latin punctuation, Arabic question marks, and several
    common East Asian sentence-ending characters.
    """
    text = normalize_whitespace(text)

    if not text:
        return []

    paragraphs = re.split(r"\n+", text)
    sentences: List[str] = []

    sentence_boundary = re.compile(
        r"(?<=[.!?؟。！？])\s+"
    )

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        parts = sentence_boundary.split(paragraph)

        for part in parts:
            part = part.strip()

            if part:
                sentences.append(part)

    return sentences


def split_oversized_unit(
    text: str,
    max_words: int = 250,
    overlap_words: int = 30,
) -> List[str]:
    """
    Split an unusually long unpunctuated sentence into smaller fallback
    units so it does not dominate memory or exceed the encoder limit.

    Normal sentences are not changed.
    """
    words = text.split()

    if len(words) <= max_words:
        return [text]

    units: List[str] = []
    step = max(1, max_words - overlap_words)
    start = 0

    while start < len(words):
        end = min(start + max_words, len(words))
        unit = " ".join(words[start:end]).strip()

        if unit:
            units.append(unit)

        if end >= len(words):
            break

        start += step

    return units


def split_article_into_sentences(
    article_text: str,
    question_language: str,
    minimum_characters: int,
) -> List[str]:
    """
    Split one article into sentence-level retrieval units.

    Punkt is attempted first to stay close to the CROSS methodology.
    A multilingual regex method is used as a fallback.
    """
    article_text = normalize_whitespace(article_text)

    if not article_text:
        return []

    punkt_language = choose_punkt_language(
        question_language=question_language,
        article_text=article_text,
    )

    sentences: List[str] = []

    try:
        punkt_sentences = sent_tokenize(
            article_text,
            language=punkt_language,
        )

        sentences = [
            sentence.strip()
            for sentence in punkt_sentences
            if sentence.strip()
        ]

    except Exception:
        sentences = regex_sentence_split(article_text)

    # Punkt can occasionally return one enormous unit for unusual text.
    if len(sentences) <= 1 and len(article_text.split()) > 250:
        sentences = regex_sentence_split(article_text)

    final_units: List[str] = []

    for sentence in sentences:
        sentence = normalize_whitespace(sentence)

        if not sentence:
            continue

        oversized_units = split_oversized_unit(sentence)

        for unit in oversized_units:
            unit = unit.strip()

            if len(unit) >= minimum_characters:
                final_units.append(unit)

    # Do not throw away an article just because it contains a short text.
    if not final_units and article_text:
        final_units = [article_text]

    return final_units


def build_sentence_index(
    record: Dict[str, Any],
    minimum_characters: int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Split all eight articles into sentence-level units.

    Returns:
        sentence_texts
        sentence_metadata
    """
    articles = get_articles(record)
    language = get_language(record)

    dates = get_parallel_list(
        record,
        key="dates",
        expected_length=len(articles),
    )

    urls = get_parallel_list(
        record,
        key="urls",
        expected_length=len(articles),
    )

    article_order = get_parallel_list(
        record,
        key="article_order",
        expected_length=len(articles),
    )

    support_articles = set(
        get_support_article_numbers_one_based(
            record=record,
            article_count=len(articles),
        )
    )

    sentence_texts: List[str] = []
    sentence_metadata: List[Dict[str, Any]] = []

    for article_number, article_text in enumerate(
        articles,
        start=1,
    ):
        sentences = split_article_into_sentences(
            article_text=article_text,
            question_language=language,
            minimum_characters=minimum_characters,
        )

        for sentence_number, sentence in enumerate(
            sentences,
            start=1,
        ):
            sentence_texts.append(sentence)

            sentence_metadata.append(
                {
                    "article_number_one_based": article_number,
                    "article_order_value": article_order[
                        article_number - 1
                    ],
                    "sentence_number_within_article": sentence_number,
                    "is_support_article": article_number
                    in support_articles,
                    "publication_date": dates[article_number - 1],
                    "source_url": urls[article_number - 1],
                }
            )

    return sentence_texts, sentence_metadata


# ---------------------------------------------------------------------
# BGE-M3 model and ranking
# ---------------------------------------------------------------------

def resolve_device(requested_device: str) -> str:
    """
    Resolve auto, cpu, cuda, or mps.
    """
    normalized = requested_device.lower().strip()

    if normalized != "auto":
        return normalized

    if torch.cuda.is_available():
        return "cuda"

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


def load_embedding_model(
    model_name: str,
    device: str,
    max_sequence_length: int,
) -> SentenceTransformer:
    """
    Load BGE-M3 through Sentence Transformers.
    """
    print()
    print("Loading embedding model...")
    print(f"Model: {model_name}")
    print(f"Device: {device}")

    model = SentenceTransformer(
        model_name,
        device=device,
    )

    if max_sequence_length > 0:
        model.max_seq_length = max_sequence_length

    print(
        f"Maximum sequence length used: "
        f"{model.max_seq_length}"
    )

    return model


def encode_and_rank_sentences(
    model: SentenceTransformer,
    question: str,
    sentence_texts: Sequence[str],
    batch_size: int,
    show_progress: bool,
) -> Tuple[List[int], List[float]]:
    """
    Embed the question and sentences and return sentence indices ranked
    by cosine similarity.

    Because embeddings are normalized, their dot product is cosine
    similarity.
    """
    if not question:
        raise ValueError("Question text is empty.")

    if not sentence_texts:
        raise ValueError(
            "No sentence units were produced for this example."
        )

    query_embedding = model.encode(
        [question],
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    sentence_embeddings = model.encode(
        list(sentence_texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )

    scores = sentence_embeddings @ query_embedding

    ranked_indices = np.argsort(-scores).tolist()

    ranked_scores = [
        float(scores[index])
        for index in ranked_indices
    ]

    return ranked_indices, ranked_scores


# ---------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------

def calculate_metrics_at_k(
    ranked_indices: Sequence[int],
    sentence_metadata: Sequence[Dict[str, Any]],
    support_article_numbers: Sequence[int],
    k: int,
) -> Dict[str, Any]:
    """
    Evaluate sentence retrieval using XRAG's article-level support labels.

    Important:
    XRAG labels support articles, not the exact gold sentence. Therefore,
    these metrics measure whether retrieved sentences came from support
    articles. They do not prove that the precise answer-bearing sentence
    was retrieved.
    """
    effective_k = min(k, len(ranked_indices))
    selected_indices = list(ranked_indices[:effective_k])

    support_set = set(support_article_numbers)

    selected_article_numbers = [
        int(
            sentence_metadata[index][
                "article_number_one_based"
            ]
        )
        for index in selected_indices
    ]

    unique_articles_retrieved = sorted(
        set(selected_article_numbers)
    )

    support_articles_retrieved = sorted(
        set(selected_article_numbers).intersection(support_set)
    )

    support_sentence_count = sum(
        1
        for article_number in selected_article_numbers
        if article_number in support_set
    )

    sentence_precision = (
        support_sentence_count / effective_k
        if effective_k
        else 0.0
    )

    support_article_recall = (
        len(support_articles_retrieved) / len(support_set)
        if support_set
        else None
    )

    complete_support = (
        support_set.issubset(
            set(support_articles_retrieved)
        )
        if support_set
        else False
    )

    return {
        "requested_k": k,
        "effective_k": effective_k,
        "support_sentence_count": support_sentence_count,
        "sentence_precision_at_k": sentence_precision,
        "unique_articles_retrieved": unique_articles_retrieved,
        "support_articles_retrieved": support_articles_retrieved,
        "support_article_recall_at_k": support_article_recall,
        "complete_support_at_k": complete_support,
    }


def find_first_support_sentence_rank(
    ranked_indices: Sequence[int],
    sentence_metadata: Sequence[Dict[str, Any]],
    support_article_numbers: Sequence[int],
) -> Optional[int]:
    """
    Find the first rank containing a sentence from a support article.
    """
    support_set = set(support_article_numbers)

    if not support_set:
        return None

    for rank, sentence_index in enumerate(
        ranked_indices,
        start=1,
    ):
        article_number = sentence_metadata[sentence_index][
            "article_number_one_based"
        ]

        if article_number in support_set:
            return rank

    return None


def reciprocal_rank(
    first_support_rank: Optional[int],
) -> float:
    """
    Compute reciprocal rank from the first sentence belonging to a
    support article.
    """
    if first_support_rank is None or first_support_rank <= 0:
        return 0.0

    return 1.0 / first_support_rank


# ---------------------------------------------------------------------
# Per-example processing
# ---------------------------------------------------------------------

def create_configuration_id(
    model_name: str,
    k_values: Sequence[int],
    max_sequence_length: int,
    minimum_characters: int,
) -> str:
    """
    Identify the evaluation configuration for resume support.
    """
    return (
        f"model={model_name}|"
        f"k={','.join(str(value) for value in k_values)}|"
        f"max_length={max_sequence_length}|"
        f"min_chars={minimum_characters}"
    )


def process_example(
    record: Dict[str, Any],
    example_number: int,
    model: SentenceTransformer,
    model_name: str,
    device: str,
    k_values: Sequence[int],
    batch_size: int,
    max_sequence_length: int,
    minimum_characters: int,
    show_progress: bool,
    configuration_id: str,
) -> Dict[str, Any]:
    """
    Run BGE-M3 sentence retrieval for one XRAG example.
    """
    question = get_question(record)
    articles = get_articles(record)

    support_article_numbers = (
        get_support_article_numbers_one_based(
            record=record,
            article_count=len(articles),
        )
    )

    sentence_texts, sentence_metadata = build_sentence_index(
        record=record,
        minimum_characters=minimum_characters,
    )

    ranked_indices, ranked_scores = (
        encode_and_rank_sentences(
            model=model,
            question=question,
            sentence_texts=sentence_texts,
            batch_size=batch_size,
            show_progress=show_progress,
        )
    )

    first_support_rank = find_first_support_sentence_rank(
        ranked_indices=ranked_indices,
        sentence_metadata=sentence_metadata,
        support_article_numbers=support_article_numbers,
    )

    metrics_by_k: Dict[str, Dict[str, Any]] = {}

    for k in k_values:
        metrics_by_k[str(k)] = calculate_metrics_at_k(
            ranked_indices=ranked_indices,
            sentence_metadata=sentence_metadata,
            support_article_numbers=support_article_numbers,
            k=k,
        )

    max_k = min(max(k_values), len(ranked_indices))
    top_sentence_results: List[Dict[str, Any]] = []

    for rank_position in range(max_k):
        sentence_index = ranked_indices[rank_position]
        metadata = sentence_metadata[sentence_index]

        top_sentence_results.append(
            {
                "rank": rank_position + 1,
                "score": round(
                    ranked_scores[rank_position],
                    8,
                ),
                "article_number_one_based": metadata[
                    "article_number_one_based"
                ],
                "article_order_value": metadata[
                    "article_order_value"
                ],
                "sentence_number_within_article": metadata[
                    "sentence_number_within_article"
                ],
                "is_support_article": metadata[
                    "is_support_article"
                ],
                "publication_date": metadata[
                    "publication_date"
                ],
                "source_url": metadata["source_url"],
                "sentence_text": sentence_texts[
                    sentence_index
                ],
            }
        )

    sentence_counts_by_article: Dict[str, int] = defaultdict(int)

    for metadata in sentence_metadata:
        article_number = str(
            metadata["article_number_one_based"]
        )
        sentence_counts_by_article[article_number] += 1

    return {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "configuration_id": configuration_id,
        "system_name": "Multilingual_RAG_System",
        "retrieval_system": "CROSS_style_BGE_M3_dense",
        "retrieval_granularity": "sentence",
        "model_name": model_name,
        "device": device,
        "max_sequence_length": max_sequence_length,
        "batch_size": batch_size,
        "example_number": example_number,
        "xrag_id": get_xrag_id(record),
        "language": get_language(record),
        "task": record.get("task"),
        "question_type": get_question_type(record),
        "answer_type": get_answer_type(record),
        "crossdoc_type": record.get("crossdoc_type"),
        "question": question,
        "gold_answer": get_gold_answer(record),
        "article_count": len(articles),
        "support_article_numbers_one_based":
            support_article_numbers,
        "total_sentence_units": len(sentence_texts),
        "sentence_counts_by_article":
            dict(sentence_counts_by_article),
        "first_support_sentence_rank":
            first_support_rank,
        "support_article_sentence_reciprocal_rank":
            reciprocal_rank(first_support_rank),
        "metrics_by_k": metrics_by_k,
        "top_sentence_results": top_sentence_results,
        "evaluation_note": (
            "XRAG provides article-level support labels. "
            "A retrieved sentence is counted as relevant when it "
            "comes from a labeled support article."
        ),
    }


# ---------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------

def load_matching_details(
    details_file: Path,
    configuration_id: str,
) -> List[Dict[str, Any]]:
    """
    Load completed examples matching the current configuration.
    """
    if not details_file.exists():
        return []

    matching_records: List[Dict[str, Any]] = []

    with details_file.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if record.get("configuration_id") == configuration_id:
                matching_records.append(record)

    latest_by_example: Dict[int, Dict[str, Any]] = {}

    for record in matching_records:
        example_number = record.get("example_number")

        if example_number is None:
            continue

        latest_by_example[int(example_number)] = record

    return [
        latest_by_example[number]
        for number in sorted(latest_by_example)
    ]


# ---------------------------------------------------------------------
# CSV creation
# ---------------------------------------------------------------------

def detail_to_csv_row(
    detail: Dict[str, Any],
    k_values: Sequence[int],
) -> Dict[str, Any]:
    """
    Flatten a detailed result into one CSV row.
    """
    top_results = detail.get(
        "top_sentence_results",
        [],
    )

    row: Dict[str, Any] = {
        "example_number": detail.get("example_number"),
        "xrag_id": detail.get("xrag_id"),
        "language": detail.get("language"),
        "task": detail.get("task"),
        "question_type": detail.get("question_type"),
        "answer_type": detail.get("answer_type"),
        "crossdoc_type": detail.get("crossdoc_type"),
        "question": detail.get("question"),
        "gold_answer": detail.get("gold_answer"),
        "article_count": detail.get("article_count"),
        "support_article_numbers_one_based": json.dumps(
            detail.get(
                "support_article_numbers_one_based",
                [],
            ),
            ensure_ascii=False,
        ),
        "total_sentence_units": detail.get(
            "total_sentence_units"
        ),
        "sentence_counts_by_article": json.dumps(
            detail.get(
                "sentence_counts_by_article",
                {},
            ),
            ensure_ascii=False,
        ),
        "first_support_sentence_rank": detail.get(
            "first_support_sentence_rank"
        ),
        "support_article_sentence_reciprocal_rank":
            detail.get(
                "support_article_sentence_reciprocal_rank"
            ),
        "top_ranked_article_numbers": json.dumps(
            [
                result.get("article_number_one_based")
                for result in top_results
            ],
            ensure_ascii=False,
        ),
        "top_ranked_sentence_scores": json.dumps(
            [
                result.get("score")
                for result in top_results
            ],
            ensure_ascii=False,
        ),
        "model_name": detail.get("model_name"),
        "retrieval_system": detail.get(
            "retrieval_system"
        ),
        "retrieval_granularity": detail.get(
            "retrieval_granularity"
        ),
        "device": detail.get("device"),
        "max_sequence_length": detail.get(
            "max_sequence_length"
        ),
        "batch_size": detail.get("batch_size"),
        "timestamp_utc": detail.get("timestamp_utc"),
    }

    metrics_by_k = detail.get("metrics_by_k", {})

    for k in k_values:
        metrics = metrics_by_k.get(str(k), {})

        row[f"effective_k_{k}"] = metrics.get(
            "effective_k"
        )

        row[f"support_sentence_count_at_{k}"] = (
            metrics.get("support_sentence_count")
        )

        row[f"sentence_precision_at_{k}"] = (
            metrics.get("sentence_precision_at_k")
        )

        row[f"support_article_recall_at_{k}"] = (
            metrics.get("support_article_recall_at_k")
        )

        row[f"complete_support_at_{k}"] = (
            metrics.get("complete_support_at_k")
        )

        row[f"unique_articles_retrieved_at_{k}"] = (
            json.dumps(
                metrics.get(
                    "unique_articles_retrieved",
                    [],
                ),
                ensure_ascii=False,
            )
        )

        row[f"support_articles_retrieved_at_{k}"] = (
            json.dumps(
                metrics.get(
                    "support_articles_retrieved",
                    [],
                ),
                ensure_ascii=False,
            )
        )

    return row


def write_csv(
    details: Sequence[Dict[str, Any]],
    csv_file: Path,
    k_values: Sequence[int],
) -> None:
    """
    Write one row per XRAG example.
    """
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        detail_to_csv_row(detail, k_values)
        for detail in details
    ]

    if not rows:
        raise ValueError("No evaluation rows were created.")

    fieldnames = list(rows[0].keys())

    with csv_file.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# Summary calculations
# ---------------------------------------------------------------------

def average_optional(
    values: Sequence[Optional[float]],
) -> float:
    """
    Average numeric values while ignoring None.
    """
    valid_values = [
        float(value)
        for value in values
        if value is not None
    ]

    return mean(valid_values) if valid_values else 0.0


def summarize_group(
    records: Sequence[Dict[str, Any]],
    k_values: Sequence[int],
) -> Dict[str, Any]:
    """
    Calculate summary metrics for a set of examples.
    """
    summary: Dict[str, Any] = {
        "examples": len(records),
        "average_total_sentence_units": (
            mean(
                float(
                    record.get(
                        "total_sentence_units",
                        0,
                    )
                )
                for record in records
            )
            if records
            else 0.0
        ),
        "support_article_sentence_mrr": (
            mean(
                float(
                    record.get(
                        "support_article_sentence_reciprocal_rank",
                        0.0,
                    )
                )
                for record in records
            )
            if records
            else 0.0
        ),
        "metrics_by_k": {},
    }

    for k in k_values:
        k_key = str(k)

        sentence_precision_values: List[
            Optional[float]
        ] = []

        article_recall_values: List[
            Optional[float]
        ] = []

        complete_count = 0
        any_support_count = 0

        for record in records:
            metrics = record.get(
                "metrics_by_k",
                {},
            ).get(k_key, {})

            sentence_precision_values.append(
                metrics.get("sentence_precision_at_k")
            )

            article_recall_values.append(
                metrics.get(
                    "support_article_recall_at_k"
                )
            )

            if metrics.get("complete_support_at_k"):
                complete_count += 1

            if metrics.get("support_sentence_count", 0) > 0:
                any_support_count += 1

        summary["metrics_by_k"][k_key] = {
            "average_sentence_precision_at_k":
                average_optional(
                    sentence_precision_values
                ),
            "average_support_article_recall_at_k":
                average_optional(
                    article_recall_values
                ),
            "complete_support_count":
                complete_count,
            "complete_support_rate": (
                complete_count / len(records)
                if records
                else 0.0
            ),
            "any_support_count": any_support_count,
            "any_support_rate": (
                any_support_count / len(records)
                if records
                else 0.0
            ),
        }

    return summary


def group_records(
    records: Sequence[Dict[str, Any]],
    key: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group result records by language, task, or question type.
    """
    grouped: Dict[
        str,
        List[Dict[str, Any]],
    ] = defaultdict(list)

    for record in records:
        value = str(record.get(key) or "unknown")
        grouped[value].append(record)

    return dict(grouped)


def add_group_summary_lines(
    lines: List[str],
    title: str,
    grouped_records: Dict[
        str,
        List[Dict[str, Any]],
    ],
    k_values: Sequence[int],
) -> None:
    """
    Add grouped metrics to the text summary.
    """
    lines.append(title)
    lines.append("-" * 78)

    for group_name in sorted(grouped_records):
        records = grouped_records[group_name]
        summary = summarize_group(records, k_values)

        lines.append(
            f"{group_name} ({len(records)} examples)"
        )

        for k in k_values:
            metrics = summary["metrics_by_k"][str(k)]

            lines.append(
                f"  Top-{k}: "
                f"Sentence precision="
                f"{metrics['average_sentence_precision_at_k']:.3f}, "
                f"Support-article recall="
                f"{metrics['average_support_article_recall_at_k']:.3f}, "
                f"Complete="
                f"{metrics['complete_support_count']}/"
                f"{len(records)}"
            )

        lines.append(
            f"  Support-article sentence MRR: "
            f"{summary['support_article_sentence_mrr']:.3f}"
        )

    lines.append("")


def write_summary(
    records: Sequence[Dict[str, Any]],
    summary_file: Path,
    sample_file: Path,
    model_name: str,
    device: str,
    k_values: Sequence[int],
    max_sequence_length: int,
) -> None:
    """
    Write a human-readable retrieval summary.
    """
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    overall = summarize_group(records, k_values)

    lines: List[str] = []

    lines.append(
        "BGE-M3 CROSS-STYLE SENTENCE RETRIEVAL SUMMARY"
    )
    lines.append("=" * 78)
    lines.append(
        f"Created UTC: "
        f"{datetime.now(timezone.utc).isoformat()}"
    )
    lines.append(f"Sample file: {sample_file}")
    lines.append(f"Examples evaluated: {len(records)}")
    lines.append(f"Embedding model: {model_name}")
    lines.append(f"Device: {device}")
    lines.append(
        f"Retrieval granularity: sentence"
    )
    lines.append(
        f"Maximum sequence length: "
        f"{max_sequence_length}"
    )
    lines.append(
        f"Sentence caps tested: "
        f"{list(k_values)}"
    )
    lines.append("")

    lines.append("Important evaluation note")
    lines.append("-" * 78)
    lines.append(
        "XRAG provides article-level support labels rather than "
        "exact answer-sentence labels."
    )
    lines.append(
        "A retrieved sentence is therefore counted as relevant "
        "when it comes from a labeled support article."
    )
    lines.append(
        "These metrics measure support-article coverage from "
        "sentence retrieval, not exact gold-sentence recall."
    )
    lines.append("")

    lines.append("Overall metrics")
    lines.append("-" * 78)
    lines.append(
        f"Average sentence units per example: "
        f"{overall['average_total_sentence_units']:.1f}"
    )
    lines.append(
        f"Support-article sentence MRR: "
        f"{overall['support_article_sentence_mrr']:.3f}"
    )

    for k in k_values:
        metrics = overall["metrics_by_k"][str(k)]

        lines.append("")
        lines.append(f"Top-{k}")
        lines.append(
            f"  Average sentence precision: "
            f"{metrics['average_sentence_precision_at_k']:.3f}"
        )
        lines.append(
            f"  Average support-article recall: "
            f"{metrics['average_support_article_recall_at_k']:.3f}"
        )
        lines.append(
            f"  Any support retrieved: "
            f"{metrics['any_support_count']}/{len(records)} "
            f"({metrics['any_support_rate'] * 100:.1f}%)"
        )
        lines.append(
            f"  Complete support retrieved: "
            f"{metrics['complete_support_count']}/{len(records)} "
            f"({metrics['complete_support_rate'] * 100:.1f}%)"
        )

    lines.append("")

    add_group_summary_lines(
        lines=lines,
        title="Performance by language",
        grouped_records=group_records(
            records,
            "language",
        ),
        k_values=k_values,
    )

    add_group_summary_lines(
        lines=lines,
        title="Performance by retrieval task",
        grouped_records=group_records(
            records,
            "task",
        ),
        k_values=k_values,
    )

    add_group_summary_lines(
        lines=lines,
        title="Performance by question type",
        grouped_records=group_records(
            records,
            "question_type",
        ),
        k_values=k_values,
    )

    lines.append("Interpretation guidance")
    lines.append("-" * 78)
    lines.append(
        "A larger sentence cap can improve support coverage, "
        "but it can also introduce more distractor sentences."
    )
    lines.append(
        "The best cap should be selected using both complete-support "
        "coverage and downstream answer-generation accuracy."
    )
    lines.append(
        "This evaluator isolates the BGE-M3 retrieval stage. "
        "No language-model API calls are made."
    )

    with summary_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write("\n".join(lines))


# ---------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------

def print_example_result(
    detail: Dict[str, Any],
    k_values: Sequence[int],
) -> None:
    """
    Print a compact result for one example.
    """
    print()
    print("#" * 78)
    print(
        f"Example {detail['example_number']} | "
        f"XRAG ID: {detail['xrag_id']}"
    )
    print(
        f"Language: {detail['language']} | "
        f"Task: {detail['task']} | "
        f"Question type: {detail['question_type']}"
    )
    print(f"Question: {detail['question']}")
    print(
        f"Support articles: "
        f"{detail['support_article_numbers_one_based']}"
    )
    print(
        f"Sentence units indexed: "
        f"{detail['total_sentence_units']}"
    )
    print(
        f"First support-article sentence rank: "
        f"{detail['first_support_sentence_rank']}"
    )

    top_results = detail.get(
        "top_sentence_results",
        [],
    )

    print()
    print("Top five retrieved sentences:")

    for result in top_results[:5]:
        preview = result["sentence_text"].replace(
            "\n",
            " ",
        )

        if len(preview) > 180:
            preview = preview[:177] + "..."

        support_marker = (
            "SUPPORT ARTICLE"
            if result["is_support_article"]
            else "non-support article"
        )

        print(
            f"  Rank {result['rank']}: "
            f"Article {result['article_number_one_based']} | "
            f"Score {result['score']:.4f} | "
            f"{support_marker}"
        )
        print(f"    {preview}")

    print()
    print("Metrics:")

    for k in k_values:
        metrics = detail["metrics_by_k"][str(k)]

        recall = metrics["support_article_recall_at_k"]

        recall_text = (
            f"{recall:.3f}"
            if recall is not None
            else "N/A"
        )

        print(
            f"  Top-{k}: "
            f"sentence precision="
            f"{metrics['sentence_precision_at_k']:.3f}, "
            f"support-article recall={recall_text}, "
            f"complete="
            f"{metrics['complete_support_at_k']}"
        )


# ---------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate CROSS-style BGE-M3 sentence retrieval "
            "on an XRAG JSONL sample."
        )
    )

    parser.add_argument(
        "--sample-file",
        default=str(DEFAULT_SAMPLE_FILE),
        help="Path to the XRAG evaluation sample.",
    )

    parser.add_argument(
        "--csv-file",
        default=str(DEFAULT_CSV_FILE),
        help="Path for the flattened CSV results.",
    )

    parser.add_argument(
        "--summary-file",
        default=str(DEFAULT_SUMMARY_FILE),
        help="Path for the text summary.",
    )

    parser.add_argument(
        "--details-file",
        default=str(DEFAULT_DETAILS_FILE),
        help="Path for detailed JSONL retrieval results.",
    )

    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Sentence Transformers embedding model.",
    )

    parser.add_argument(
        "--k-values",
        type=parse_k_values,
        default=DEFAULT_K_VALUES,
        help=(
            "Comma-separated sentence caps. "
            "Example: 3,5,10,20,50"
        ),
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=3,
        help=(
            "Number of examples to evaluate. "
            "Use 50 for the complete sample."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Sentence encoding batch size.",
    )

    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=DEFAULT_MAX_SEQUENCE_LENGTH,
        help=(
            "Maximum encoder sequence length. "
            "Sentence-level units normally fit within 512."
        ),
    )

    parser.add_argument(
        "--minimum-characters",
        type=int,
        default=10,
        help=(
            "Minimum sentence length after tokenization."
        ),
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Embedding device.",
    )

    parser.add_argument(
        "--show-progress",
        action="store_true",
        help=(
            "Show Sentence Transformers encoding progress bars."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue an interrupted run using the details JSONL."
        ),
    )

    args = parser.parse_args()

    sample_file = Path(args.sample_file)
    csv_file = Path(args.csv_file)
    summary_file = Path(args.summary_file)
    details_file = Path(args.details_file)

    k_values: List[int] = list(args.k_values)

    if args.max_examples <= 0:
        raise ValueError(
            "--max-examples must be greater than zero."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be greater than zero."
        )

    OUTPUTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prepare_nltk_resources()

    records = load_jsonl(sample_file)
    selected_records = records[: args.max_examples]

    device = resolve_device(args.device)

    configuration_id = create_configuration_id(
        model_name=args.model_name,
        k_values=k_values,
        max_sequence_length=args.max_sequence_length,
        minimum_characters=args.minimum_characters,
    )

    if not args.resume:
        remove_file_if_present(details_file)
        remove_file_if_present(csv_file)
        remove_file_if_present(summary_file)

    existing_details = load_matching_details(
        details_file=details_file,
        configuration_id=configuration_id,
    )

    completed_examples = {
        int(detail["example_number"])
        for detail in existing_details
    }

    print("BGE-M3 CROSS-STYLE SENTENCE RETRIEVAL")
    print("=" * 78)
    print(f"Root directory: {ROOT_DIR}")
    print(f"Sample file: {sample_file}")
    print(f"Examples requested: {args.max_examples}")
    print(f"Examples available: {len(records)}")
    print(f"Model: {args.model_name}")
    print(f"Device: {device}")
    print(f"Sentence caps: {k_values}")
    print(f"Batch size: {args.batch_size}")
    print(f"Resume mode: {args.resume}")
    print(
        f"Already completed for this configuration: "
        f"{len(completed_examples)}"
    )

    model = load_embedding_model(
        model_name=args.model_name,
        device=device,
        max_sequence_length=args.max_sequence_length,
    )

    processed_count = 0
    skipped_count = 0

    for example_number, record in enumerate(
        selected_records,
        start=1,
    ):
        if example_number in completed_examples:
            print(
                f"Skipping completed example "
                f"{example_number}"
            )
            skipped_count += 1
            continue

        try:
            detail = process_example(
                record=record,
                example_number=example_number,
                model=model,
                model_name=args.model_name,
                device=device,
                k_values=k_values,
                batch_size=args.batch_size,
                max_sequence_length=args.max_sequence_length,
                minimum_characters=args.minimum_characters,
                show_progress=args.show_progress,
                configuration_id=configuration_id,
            )

        except RuntimeError as error:
            error_message = str(error).lower()

            if (
                "out of memory" in error_message
                or "cuda" in error_message
            ):
                raise RuntimeError(
                    "The embedding model ran out of memory. "
                    "Try rerunning with --batch-size 2 or "
                    "--device cpu."
                ) from error

            raise

        append_jsonl(details_file, detail)
        print_example_result(detail, k_values)

        processed_count += 1

    all_details = load_matching_details(
        details_file=details_file,
        configuration_id=configuration_id,
    )

    allowed_example_numbers = set(
        range(1, len(selected_records) + 1)
    )

    all_details = [
        detail
        for detail in all_details
        if int(detail.get("example_number", -1))
        in allowed_example_numbers
    ]

    write_csv(
        details=all_details,
        csv_file=csv_file,
        k_values=k_values,
    )

    write_summary(
        records=all_details,
        summary_file=summary_file,
        sample_file=sample_file,
        model_name=args.model_name,
        device=device,
        k_values=k_values,
        max_sequence_length=args.max_sequence_length,
    )

    print()
    print("=" * 78)
    print("BGE-M3 SENTENCE RETRIEVAL FINISHED")
    print(f"Processed this run: {processed_count}")
    print(f"Skipped this run: {skipped_count}")
    print(
        f"Total completed examples: "
        f"{len(all_details)}"
    )
    print(f"CSV saved to: {csv_file}")
    print(f"Summary saved to: {summary_file}")
    print(f"Detailed JSONL saved to: {details_file}")


if __name__ == "__main__":
    main()