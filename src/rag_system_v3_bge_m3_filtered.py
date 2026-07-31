"""
V3 retrieval-only system.

Pipeline:
    XRAG question
    -> BGE-M3 raw Top-20 sentence retrieval
    -> exact duplicate removal
    -> semantic duplicate removal
    -> article diversity and MMR filtering
    -> final Top-10 evidence
    -> retrieval evaluation

This version does not call GPT. We first verify that filtering preserves
the required support evidence before adding answer generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SAMPLE_FILE = (
    PROJECT_ROOT
    / "data"
    / "xrag_eval_sample_50.jsonl"
)

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "v3"
)


# ============================================================================
# V3 SETTINGS
# ============================================================================

MODEL_NAME = "BAAI/bge-m3"

RAW_TOP_K = 20
FINAL_TOP_K = 12

SEMANTIC_DUPLICATE_THRESHOLD = 0.92

MAX_SENTENCES_PER_ARTICLE = 4

MMR_LAMBDA = 0.85

MIN_SENTENCE_CHARACTERS = 10


# ============================================================================
# SENTENCE DATA STRUCTURE
# ============================================================================

@dataclass
class SentenceUnit:
    """
    Stores one sentence and all metadata needed for retrieval and evaluation.
    """

    article_index: int
    sentence_index: int
    text: str
    is_support_article: bool

    publication_date: str = ""
    source_url: str = ""

    embedding: np.ndarray | None = None

    relevance_score: float = 0.0
    raw_rank: int | None = None

    mmr_score: float | None = None
    final_rank: int | None = None

    @property
    def article_number(self) -> int:
        """
        Return a human-readable one-based article number.
        """

        return self.article_index + 1

    @property
    def sentence_number(self) -> int:
        """
        Return a human-readable one-based sentence number.
        """

        return self.sentence_index + 1

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the sentence into a JSON-safe dictionary.
        """

        return {
            "raw_rank": self.raw_rank,
            "final_rank": self.final_rank,
            "article_number": self.article_number,
            "sentence_number": self.sentence_number,
            "relevance_score": round(
                float(self.relevance_score),
                8,
            ),
            "mmr_score": (
                round(float(self.mmr_score), 8)
                if self.mmr_score is not None
                else None
            ),
            "is_support_article": self.is_support_article,
            "publication_date": self.publication_date,
            "source_url": self.source_url,
            "text": self.text,
        }


# ============================================================================
# COMMAND-LINE ARGUMENTS
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run V3 BGE-M3 retrieval filtering on the XRAG sample."
        )
    )

    parser.add_argument(
        "--sample-file",
        type=Path,
        default=DEFAULT_SAMPLE_FILE,
        help="Path to the XRAG evaluation JSONL file.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory where V3 output files will be saved.",
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=3,
        help=(
            "Maximum number of examples to run. "
            "Use 0 to run the entire file."
        ),
    )

    parser.add_argument(
        "--preview-count",
        type=int,
        default=3,
        help=(
            "Number of examples whose raw and filtered evidence "
            "will be printed."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Embedding batch size.",
    )

    parser.add_argument(
        "--device",
        default=None,
        help='Optional device such as "cpu" or "cuda".',
    )

    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show SentenceTransformer progress bars.",
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if args.max_examples < 0:
        raise ValueError(
            "--max-examples cannot be negative."
        )

    if args.preview_count < 0:
        raise ValueError(
            "--preview-count cannot be negative."
        )

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be at least 1."
        )

    if RAW_TOP_K < 1:
        raise ValueError(
            "RAW_TOP_K must be at least 1."
        )

    if FINAL_TOP_K < 1:
        raise ValueError(
            "FINAL_TOP_K must be at least 1."
        )

    if FINAL_TOP_K > RAW_TOP_K:
        raise ValueError(
            "FINAL_TOP_K cannot be greater than RAW_TOP_K."
        )

    if not 0.0 <= SEMANTIC_DUPLICATE_THRESHOLD <= 1.0:
        raise ValueError(
            "SEMANTIC_DUPLICATE_THRESHOLD must be between 0 and 1."
        )

    if MAX_SENTENCES_PER_ARTICLE < 1:
        raise ValueError(
            "MAX_SENTENCES_PER_ARTICLE must be at least 1."
        )

    if not 0.0 <= MMR_LAMBDA <= 1.0:
        raise ValueError(
            "MMR_LAMBDA must be between 0 and 1."
        )


# ============================================================================
# JSONL LOADING
# ============================================================================

def load_jsonl(
    file_path: Path,
) -> list[dict[str, Any]]:
    """
    Load the XRAG JSONL evaluation file.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find the sample file:\n{file_path}"
        )

    examples: list[dict[str, Any]] = []

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as input_file:

        for line_number, line in enumerate(
            input_file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                example = json.loads(line)

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

            if not isinstance(example, dict):
                raise ValueError(
                    f"Line {line_number} is not a JSON object."
                )

            examples.append(example)

    if not examples:
        raise ValueError(
            f"No examples were found in:\n{file_path}"
        )

    return examples


# ============================================================================
# XRAG FIELD HELPERS
# ============================================================================

def get_first_value(
    example: dict[str, Any],
    *possible_keys: str,
    default: Any = "",
) -> Any:
    """
    Return the first available field from a list of possible names.
    """

    for key in possible_keys:
        value = example.get(key)

        if value is not None:
            return value

    return default


def get_question(
    example: dict[str, Any],
) -> str:
    return str(
        get_first_value(
            example,
            "question",
            "query",
        )
    ).strip()


def get_gold_answer(
    example: dict[str, Any],
) -> str:
    return str(
        get_first_value(
            example,
            "answer",
            "gold_answer",
        )
    ).strip()


def get_language(
    example: dict[str, Any],
) -> str:
    return str(
        get_first_value(
            example,
            "language",
            "lang",
        )
    ).strip()


def get_question_type(
    example: dict[str, Any],
) -> str:
    return str(
        get_first_value(
            example,
            "q_type",
            "question_type",
        )
    ).strip()


def get_answer_type(
    example: dict[str, Any],
) -> str:
    return str(
        get_first_value(
            example,
            "a_type",
            "answer_type",
        )
    ).strip()


def get_task(
    example: dict[str, Any],
) -> str:
    return str(
        get_first_value(
            example,
            "task",
            "retrieval_type",
            "crossdoc_type",
        )
    ).strip()


def get_article_text(
    article: Any,
) -> str:
    """
    Support articles stored as either plain strings or dictionaries.
    """

    if isinstance(article, str):
        return article.strip()

    if isinstance(article, dict):
        parts: list[str] = []

        for key in (
            "title",
            "text",
            "content",
            "body",
            "article",
        ):
            value = article.get(key)

            if isinstance(value, str) and value.strip():
                parts.append(value.strip())

        return "\n".join(parts)

    if article is None:
        return ""

    return str(article).strip()


def get_safe_list_value(
    values: Any,
    index: int,
) -> str:
    """
    Safely retrieve one value from a possible list.
    """

    if not isinstance(values, list):
        return ""

    if not 0 <= index < len(values):
        return ""

    value = values[index]

    if value is None:
        return ""

    return str(value)


def get_support_indexes(
    example: dict[str, Any],
    article_count: int,
) -> list[int]:
    """
    Convert XRAG's article-level is_support values into zero-based indexes.

    Any positive value counts as support.
    """

    labels = example.get(
        "is_support",
        [],
    )

    if not isinstance(labels, list):
        return []

    support_indexes: list[int] = []

    limit = min(
        article_count,
        len(labels),
    )

    for index in range(limit):
        try:
            support_value = float(labels[index])

        except (TypeError, ValueError):
            continue

        if support_value > 0:
            support_indexes.append(index)

    return support_indexes


# ============================================================================
# SENTENCE SPLITTING
# ============================================================================

def regex_sentence_split(
    text: str,
) -> list[str]:
    """
    Multilingual fallback sentence splitter.

    Handles common Latin, Arabic, Chinese, and Japanese punctuation.
    """

    text = re.sub(
        r"\r\n?",
        "\n",
        text,
    )

    pieces = re.split(
        r"(?<=[.!?؟。！？])\s+|\n+",
        text,
        flags=re.UNICODE,
    )

    return [
        piece.strip()
        for piece in pieces
        if piece.strip()
    ]


def split_sentences(
    text: str,
    language: str,
) -> list[str]:
    """
    Use NLTK when an appropriate tokenizer is available.

    Arabic falls back to the multilingual regular-expression splitter.
    """

    language_map = {
        "en": "english",
        "es": "spanish",
        "de": "german",
    }

    nltk_language = language_map.get(
        language.lower()
    )

    if nltk_language:
        try:
            from nltk.tokenize import sent_tokenize

            sentences = sent_tokenize(
                text,
                language=nltk_language,
            )

            cleaned_sentences = [
                sentence.strip()
                for sentence in sentences
                if sentence.strip()
            ]

            if cleaned_sentences:
                return cleaned_sentences

        except (ImportError, LookupError):
            pass

    return regex_sentence_split(text)


def build_sentence_units(
    example: dict[str, Any],
    support_indexes: list[int],
) -> list[SentenceUnit]:
    """
    Convert the eight XRAG articles into individual sentence units.
    """

    articles = example.get(
        "articles",
        [],
    )

    if not isinstance(articles, list) or not articles:
        raise ValueError(
            "Example does not contain a usable articles list."
        )

    language = get_language(example)

    publication_dates = get_first_value(
        example,
        "dates",
        "article_dates",
        default=[],
    )

    source_urls = get_first_value(
        example,
        "urls",
        "article_urls",
        default=[],
    )

    support_set = set(support_indexes)

    sentence_units: list[SentenceUnit] = []

    for article_index, article in enumerate(articles):
        full_article_text = get_article_text(article)

        sentences = split_sentences(
            full_article_text,
            language,
        )

        publication_date = get_safe_list_value(
            publication_dates,
            article_index,
        )

        source_url = get_safe_list_value(
            source_urls,
            article_index,
        )

        for sentence_index, sentence in enumerate(sentences):
            sentence = sentence.strip()

            if len(sentence) < MIN_SENTENCE_CHARACTERS:
                continue

            sentence_units.append(
                SentenceUnit(
                    article_index=article_index,
                    sentence_index=sentence_index,
                    text=sentence,
                    is_support_article=(
                        article_index in support_set
                    ),
                    publication_date=publication_date,
                    source_url=source_url,
                )
            )

    if not sentence_units:
        raise ValueError(
            "Sentence splitting produced no usable sentences."
        )

    return sentence_units


# ============================================================================
# BGE-M3 EMBEDDING AND RAW TOP-20 RETRIEVAL
# ============================================================================

def embed_question_and_sentences(
    model: SentenceTransformer,
    question: str,
    sentence_units: list[SentenceUnit],
    batch_size: int,
    show_progress: bool,
) -> np.ndarray:
    """
    Embed the question and all candidate sentences.

    normalize_embeddings=True makes every embedding have unit length.
    Therefore, dot product can be used as cosine similarity.
    """

    texts = [
        question,
        *[
            sentence_unit.text
            for sentence_unit in sentence_units
        ],
    ]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if embeddings.ndim != 2:
        raise RuntimeError(
            "The embedding output was not a two-dimensional array."
        )

    if embeddings.shape[0] != len(texts):
        raise RuntimeError(
            "The embedding count does not match the input count."
        )

    question_embedding = embeddings[0]

    sentence_embeddings = embeddings[1:]

    for sentence_unit, embedding in zip(
        sentence_units,
        sentence_embeddings,
        strict=True,
    ):
        sentence_unit.embedding = embedding

    return question_embedding


def retrieve_raw_top_20(
    question_embedding: np.ndarray,
    sentence_units: list[SentenceUnit],
) -> list[SentenceUnit]:
    """
    Rank all sentences by cosine similarity to the question.
    """

    if any(
        sentence_unit.embedding is None
        for sentence_unit in sentence_units
    ):
        raise RuntimeError(
            "At least one sentence is missing its embedding."
        )

    embedding_matrix = np.stack(
        [
            sentence_unit.embedding
            for sentence_unit in sentence_units
        ]
    )

    similarity_scores = (
        embedding_matrix
        @ question_embedding
    )

    ranked_indexes = np.argsort(
        -similarity_scores,
        kind="stable",
    )

    selected_indexes = ranked_indexes[
        : min(
            RAW_TOP_K,
            len(sentence_units),
        )
    ]

    selected_sentences: list[SentenceUnit] = []

    for rank, sentence_index in enumerate(
        selected_indexes,
        start=1,
    ):
        sentence_unit = sentence_units[
            int(sentence_index)
        ]

        sentence_unit.relevance_score = float(
            similarity_scores[int(sentence_index)]
        )

        sentence_unit.raw_rank = rank

        selected_sentences.append(
            sentence_unit
        )

    return selected_sentences


# ============================================================================
# EXACT DUPLICATE REMOVAL
# ============================================================================

def normalize_sentence_text(
    text: str,
) -> str:
    """
    Normalize text for exact duplicate comparison.
    """

    normalized = unicodedata.normalize(
        "NFKC",
        text,
    )

    normalized = normalized.casefold()

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()

    normalized = re.sub(
        r"([.!?؟،,:;؛])\1+",
        r"\1",
        normalized,
    )

    normalized = normalized.strip(
        " \t\r\n.!?؟،,:;؛-–—"
    )

    return normalized


def remove_exact_duplicates(
    candidates: list[SentenceUnit],
) -> tuple[
    list[SentenceUnit],
    list[dict[str, Any]],
]:
    """
    Keep the highest-ranked copy of an exactly repeated sentence.
    """

    seen_sentences: dict[
        str,
        SentenceUnit,
    ] = {}

    kept_sentences: list[SentenceUnit] = []

    removed_sentences: list[
        dict[str, Any]
    ] = []

    for candidate in candidates:
        normalized_text = normalize_sentence_text(
            candidate.text
        )

        if not normalized_text:
            kept_sentences.append(candidate)
            continue

        previous_sentence = seen_sentences.get(
            normalized_text
        )

        if previous_sentence is None:
            seen_sentences[
                normalized_text
            ] = candidate

            kept_sentences.append(candidate)
            continue

        removed_sentences.append(
            {
                "reason": "exact_duplicate",
                "removed_raw_rank": candidate.raw_rank,
                "kept_raw_rank": previous_sentence.raw_rank,
                "removed_article": candidate.article_number,
                "kept_article": previous_sentence.article_number,
                "removed_text": candidate.text,
            }
        )

    return kept_sentences, removed_sentences


# ============================================================================
# SEMANTIC NEAR-DUPLICATE REMOVAL
# ============================================================================

def get_numeric_signature(
    text: str,
) -> tuple[str, ...]:
    """
    Extract numbers and numeric date patterns from a sentence.

    Similar-looking sentences with different numbers are preserved.
    This reduces the risk of deleting conflicting casualty counts,
    dates, times, or quantities.
    """

    normalized = unicodedata.normalize(
        "NFKC",
        text,
    )

    return tuple(
        re.findall(
            r"\d+(?:[.,:/-]\d+)*",
            normalized,
        )
    )


def cosine_similarity(
    first_embedding: np.ndarray,
    second_embedding: np.ndarray,
) -> float:
    """
    The embeddings were normalized, so dot product equals cosine similarity.
    """

    return float(
        np.dot(
            first_embedding,
            second_embedding,
        )
    )


def remove_semantic_duplicates(
    candidates: list[SentenceUnit],
) -> tuple[
    list[SentenceUnit],
    list[dict[str, Any]],
]:
    """
    Remove lower-ranked sentences that are extremely similar.

    Sentences containing different numeric information are not merged.
    """

    kept_sentences: list[SentenceUnit] = []

    removed_sentences: list[
        dict[str, Any]
    ] = []

    for candidate in candidates:
        if candidate.embedding is None:
            raise RuntimeError(
                "A candidate is missing its embedding."
            )

        duplicate_of: SentenceUnit | None = None

        highest_similarity = -1.0

        candidate_numbers = get_numeric_signature(
            candidate.text
        )

        for existing_sentence in kept_sentences:
            if existing_sentence.embedding is None:
                raise RuntimeError(
                    "A kept sentence is missing its embedding."
                )

            existing_numbers = get_numeric_signature(
                existing_sentence.text
            )

            # Preserve similar statements when their numeric facts differ.
            if candidate_numbers != existing_numbers:
                continue

            similarity = cosine_similarity(
                candidate.embedding,
                existing_sentence.embedding,
            )

            if (
                similarity
                >= SEMANTIC_DUPLICATE_THRESHOLD
                and similarity > highest_similarity
            ):
                duplicate_of = existing_sentence
                highest_similarity = similarity

        if duplicate_of is None:
            kept_sentences.append(candidate)
            continue

        removed_sentences.append(
            {
                "reason": "semantic_near_duplicate",
                "similarity": round(
                    highest_similarity,
                    8,
                ),
                "threshold": (
                    SEMANTIC_DUPLICATE_THRESHOLD
                ),
                "removed_raw_rank": candidate.raw_rank,
                "kept_raw_rank": duplicate_of.raw_rank,
                "removed_article": candidate.article_number,
                "kept_article": duplicate_of.article_number,
                "removed_text": candidate.text,
                "kept_text": duplicate_of.text,
            }
        )

    return kept_sentences, removed_sentences


# ============================================================================
# MMR AND ARTICLE DIVERSITY
# ============================================================================

def select_final_evidence_with_mmr(
    candidates: list[SentenceUnit],
) -> list[SentenceUnit]:
    """
    Select the final Top-10 using Maximum Marginal Relevance.

    MMR score:

        lambda * question relevance
        -
        (1 - lambda) * similarity to selected evidence

    No article may contribute more than three final sentences.
    """

    selected_sentences: list[
        SentenceUnit
    ] = []

    remaining_candidates = list(candidates)

    article_counts: Counter[int] = Counter()

    while (
        remaining_candidates
        and len(selected_sentences) < FINAL_TOP_K
    ):
        eligible_candidates = [
            candidate
            for candidate in remaining_candidates
            if (
                article_counts[
                    candidate.article_index
                ]
                < MAX_SENTENCES_PER_ARTICLE
            )
        ]

        if not eligible_candidates:
            break

        best_candidate: SentenceUnit | None = None

        best_comparison_key: (
            tuple[float, float, int]
            | None
        ) = None

        best_mmr_score = 0.0

        for candidate in eligible_candidates:
            if candidate.embedding is None:
                raise RuntimeError(
                    "An MMR candidate is missing its embedding."
                )

            relevance = candidate.relevance_score

            if not selected_sentences:
                redundancy = 0.0

            else:
                redundancy = max(
                    cosine_similarity(
                        candidate.embedding,
                        selected_sentence.embedding,
                    )
                    for selected_sentence in selected_sentences
                    if selected_sentence.embedding is not None
                )

            mmr_score = (
                MMR_LAMBDA
                * relevance
                -
                (1.0 - MMR_LAMBDA)
                * redundancy
            )

            comparison_key = (
                mmr_score,
                relevance,
                -(
                    candidate.raw_rank
                    if candidate.raw_rank is not None
                    else 999999
                ),
            )

            if (
                best_comparison_key is None
                or comparison_key
                > best_comparison_key
            ):
                best_candidate = candidate
                best_comparison_key = comparison_key
                best_mmr_score = mmr_score

        if best_candidate is None:
            break

        best_candidate.mmr_score = float(
            best_mmr_score
        )

        best_candidate.final_rank = (
            len(selected_sentences)
            + 1
        )

        selected_sentences.append(
            best_candidate
        )

        article_counts[
            best_candidate.article_index
        ] += 1

        remaining_candidates.remove(
            best_candidate
        )

    return selected_sentences


# ============================================================================
# RETRIEVAL EVALUATION
# ============================================================================

def calculate_retrieval_metrics(
    selected_sentences: list[SentenceUnit],
    support_indexes: list[int],
) -> dict[str, Any]:
    """
    Evaluate article-level support coverage.

    XRAG provides article-level labels, not exact supporting-sentence labels.
    """

    retrieved_article_indexes = {
        sentence.article_index
        for sentence in selected_sentences
    }

    support_set = set(
        support_indexes
    )

    found_support_articles = (
        retrieved_article_indexes
        .intersection(support_set)
    )

    support_recall = (
        len(found_support_articles)
        / len(support_set)
        if support_set
        else 0.0
    )

    complete_support = (
        support_set.issubset(
            retrieved_article_indexes
        )
        if support_set
        else False
    )

    return {
        "effective_k": len(
            selected_sentences
        ),
        "unique_article_count": len(
            retrieved_article_indexes
        ),
        "retrieved_articles": sorted(
            index + 1
            for index
            in retrieved_article_indexes
        ),
        "support_articles_found": sorted(
            index + 1
            for index
            in found_support_articles
        ),
        "support_recall": support_recall,
        "complete_support": complete_support,
        "any_support": bool(
            found_support_articles
        ),
    }


def get_article_counts(
    selected_sentences: list[SentenceUnit],
) -> dict[str, int]:
    """
    Count how many selected sentences came from each article.
    """

    counts = Counter(
        sentence.article_number
        for sentence in selected_sentences
    )

    return {
        str(article_number): count
        for article_number, count
        in sorted(counts.items())
    }


# ============================================================================
# TERMINAL PREVIEW
# ============================================================================

def print_example_preview(
    example_number: int,
    example: dict[str, Any],
    support_indexes: list[int],
    raw_candidates: list[SentenceUnit],
    filtered_evidence: list[SentenceUnit],
    exact_duplicate_count: int,
    semantic_duplicate_count: int,
) -> None:
    """
    Print the raw Top-20 and filtered Top-10 for manual inspection.
    """

    print(
        "\n"
        + "#"
        * 100
    )

    print(
        f"EXAMPLE {example_number}"
    )

    print(
        f"XRAG ID: "
        f"{get_first_value(example, 'id', 'xrag_id')}"
    )

    print(
        f"Language: {get_language(example)}"
    )

    print(
        f"Task: {get_task(example)}"
    )

    print(
        f"Question type: "
        f"{get_question_type(example)}"
    )

    print(
        f"Question: {get_question(example)}"
    )

    print(
        "True support articles: "
        f"{[
            index + 1
            for index
            in support_indexes
        ]}"
    )

    print(
        "Duplicates removed: "
        f"exact={exact_duplicate_count}, "
        f"semantic={semantic_duplicate_count}"
    )

    print(
        f"\nRAW TOP-{len(raw_candidates)}"
    )

    for sentence in raw_candidates:
        evidence_label = (
            "SUPPORT"
            if sentence.is_support_article
            else "distractor"
        )

        print(
            f"{sentence.raw_rank:>2}. "
            f"A{sentence.article_number} "
            f"S{sentence.sentence_number} "
            f"score="
            f"{sentence.relevance_score:.4f} "
            f"[{evidence_label}] "
            f"{sentence.text[:180]}"
        )

    print(
        f"\nFILTERED TOP-{len(filtered_evidence)}"
    )

    for sentence in filtered_evidence:
        evidence_label = (
            "SUPPORT"
            if sentence.is_support_article
            else "distractor"
        )

        print(
            f"{sentence.final_rank:>2}. "
            f"A{sentence.article_number} "
            f"S{sentence.sentence_number} "
            f"raw={sentence.raw_rank} "
            f"rel="
            f"{sentence.relevance_score:.4f} "
            f"mmr="
            f"{sentence.mmr_score:.4f} "
            f"[{evidence_label}] "
            f"{sentence.text[:180]}"
        )


# ============================================================================
# OUTPUT HELPERS
# ============================================================================

def write_jsonl(
    output_path: Path,
    records: list[dict[str, Any]],
) -> None:
    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        for record in records:
            output_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_csv(
    output_path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as output_file:

        writer = csv.DictWriter(
            output_file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(rows)


def create_summary(
    rows: list[dict[str, Any]],
) -> str:
    """
    Create the overall raw-versus-filtered summary.
    """

    example_count = len(rows)

    if example_count == 0:
        return "No V3 results were produced."

    average_raw_recall = (
        sum(
            float(
                row["raw_support_recall"]
            )
            for row in rows
        )
        / example_count
    )

    average_filtered_recall = (
        sum(
            float(
                row["filtered_support_recall"]
            )
            for row in rows
        )
        / example_count
    )

    raw_complete_count = sum(
        bool(
            row["raw_complete_support"]
        )
        for row in rows
    )

    filtered_complete_count = sum(
        bool(
            row["filtered_complete_support"]
        )
        for row in rows
    )

    raw_any_support_count = sum(
        bool(
            row["raw_any_support"]
        )
        for row in rows
    )

    filtered_any_support_count = sum(
        bool(
            row["filtered_any_support"]
        )
        for row in rows
    )

    support_loss_count = sum(
        bool(
            row["support_lost_after_filtering"]
        )
        for row in rows
    )

    average_exact_duplicates = (
        sum(
            int(
                row["exact_duplicates_removed"]
            )
            for row in rows
        )
        / example_count
    )

    average_semantic_duplicates = (
        sum(
            int(
                row["semantic_duplicates_removed"]
            )
            for row in rows
        )
        / example_count
    )

    average_unique_articles = (
        sum(
            int(
                row["filtered_unique_article_count"]
            )
            for row in rows
        )
        / example_count
    )

    summary_lines = [
        "SYSTEM V3 FILTERED RETRIEVAL SUMMARY",
        "=" * 78,
        f"Examples evaluated: {example_count}",
        f"Embedding model: {MODEL_NAME}",
        f"Raw candidate pool: Top-{RAW_TOP_K}",
        f"Final evidence size: Top-{FINAL_TOP_K}",
        (
            "Semantic duplicate threshold: "
            f"{SEMANTIC_DUPLICATE_THRESHOLD}"
        ),
        (
            "Maximum sentences per article: "
            f"{MAX_SENTENCES_PER_ARTICLE}"
        ),
        f"MMR lambda: {MMR_LAMBDA}",
        "",
        "RAW TOP-20",
        "-" * 78,
        (
            "Average support recall: "
            f"{average_raw_recall:.3f}"
        ),
        (
            "Complete support: "
            f"{raw_complete_count}/{example_count}"
        ),
        (
            "Any support: "
            f"{raw_any_support_count}/{example_count}"
        ),
        "",
        f"FILTERED TOP-{FINAL_TOP_K}",
        "-" * 78,
        (
            "Average support recall: "
            f"{average_filtered_recall:.3f}"
        ),
        (
            "Complete support: "
            f"{filtered_complete_count}/{example_count}"
        ),
        (
            "Any support: "
            f"{filtered_any_support_count}/{example_count}"
        ),
        (
            "Support lost after filtering: "
            f"{support_loss_count}/{example_count}"
        ),
        (
            "Average exact duplicates removed: "
            f"{average_exact_duplicates:.2f}"
        ),
        (
            "Average semantic duplicates removed: "
            f"{average_semantic_duplicates:.2f}"
        ),
        (
            "Average unique articles in final evidence: "
            f"{average_unique_articles:.2f}"
        ),
        "",
        "IMPORTANT LIMITATION",
        "-" * 78,
        (
            "XRAG provides article-level support labels rather than "
            "exact supporting-sentence labels."
        ),
        (
            "A retrieved sentence counts as support-related when it "
            "comes from a labeled support article."
        ),
    ]

    return "\n".join(
        summary_lines
    )


# ============================================================================
# MAIN PROGRAM
# ============================================================================

def main() -> None:
    args = parse_arguments()

    validate_arguments(args)

    examples = load_jsonl(
        args.sample_file
    )

    if args.max_examples > 0:
        examples = examples[
            : args.max_examples
        ]

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "SYSTEM V3: BGE-M3 FILTERED RETRIEVAL"
    )

    print(
        "="
        * 100
    )

    print(
        f"Sample file: {args.sample_file}"
    )

    print(
        f"Output directory: {args.output_dir}"
    )

    print(
        f"Examples: {len(examples)}"
    )

    print(
        f"Embedding model: {MODEL_NAME}"
    )

    print(
        f"Raw candidates: Top-{RAW_TOP_K}"
    )

    print(
        f"Final evidence: Top-{FINAL_TOP_K}"
    )

    print(
        "Filtering: "
        "exact duplicates -> "
        "semantic duplicates -> "
        "article diversity -> "
        "MMR"
    )

    print(
        "\nLoading BGE-M3..."
    )

    model = SentenceTransformer(
        MODEL_NAME,
        device=args.device,
    )

    print(
        "BGE-M3 loaded successfully."
    )

    detailed_records: list[
        dict[str, Any]
    ] = []

    evaluation_rows: list[
        dict[str, Any]
    ] = []

    for example_number, example in enumerate(
        examples,
        start=1,
    ):
        question = get_question(example)

        if not question:
            raise ValueError(
                f"Example {example_number} "
                "does not contain a usable question."
            )

        articles = example.get(
            "articles",
            [],
        )

        if not isinstance(articles, list):
            raise ValueError(
                f"Example {example_number} "
                "does not contain a usable articles list."
            )

        support_indexes = get_support_indexes(
            example,
            article_count=len(articles),
        )

        sentence_units = build_sentence_units(
            example,
            support_indexes,
        )

        question_embedding = (
            embed_question_and_sentences(
                model=model,
                question=question,
                sentence_units=sentence_units,
                batch_size=args.batch_size,
                show_progress=args.show_progress,
            )
        )

        raw_candidates = retrieve_raw_top_20(
            question_embedding,
            sentence_units,
        )

        (
            after_exact_deduplication,
            exact_duplicate_removals,
        ) = remove_exact_duplicates(
            raw_candidates
        )

        (
            after_semantic_deduplication,
            semantic_duplicate_removals,
        ) = remove_semantic_duplicates(
            after_exact_deduplication
        )

        filtered_evidence = (
            select_final_evidence_with_mmr(
                after_semantic_deduplication
            )
        )

        raw_metrics = (
            calculate_retrieval_metrics(
                raw_candidates,
                support_indexes,
            )
        )

        filtered_metrics = (
            calculate_retrieval_metrics(
                filtered_evidence,
                support_indexes,
            )
        )

        raw_support_articles_found = set(
            raw_metrics[
                "support_articles_found"
            ]
        )

        filtered_support_articles_found = set(
            filtered_metrics[
                "support_articles_found"
            ]
        )

        support_articles_removed = sorted(
            raw_support_articles_found
            -
            filtered_support_articles_found
        )

        support_lost_after_filtering = bool(
            raw_metrics[
                "complete_support"
            ]
            and not filtered_metrics[
                "complete_support"
            ]
        )

        detailed_record = {
            "system_version": (
                "V3_retrieval_only"
            ),
            "embedding_model": MODEL_NAME,
            "configuration": {
                "raw_top_k": RAW_TOP_K,
                "final_top_k": FINAL_TOP_K,
                "semantic_duplicate_threshold": (
                    SEMANTIC_DUPLICATE_THRESHOLD
                ),
                "max_sentences_per_article": (
                    MAX_SENTENCES_PER_ARTICLE
                ),
                "mmr_lambda": MMR_LAMBDA,
            },
            "example_number": example_number,
            "xrag_id": get_first_value(
                example,
                "id",
                "xrag_id",
            ),
            "language": get_language(example),
            "task": get_task(example),
            "question_type": (
                get_question_type(example)
            ),
            "answer_type": (
                get_answer_type(example)
            ),
            "question": question,
            "gold_answer": (
                get_gold_answer(example)
            ),
            "support_articles": [
                index + 1
                for index
                in support_indexes
            ],
            "total_sentence_units": len(
                sentence_units
            ),
            "raw_metrics": raw_metrics,
            "filtered_metrics": (
                filtered_metrics
            ),
            "support_lost_after_filtering": (
                support_lost_after_filtering
            ),
            "support_articles_removed": (
                support_articles_removed
            ),
            "raw_article_counts": (
                get_article_counts(
                    raw_candidates
                )
            ),
            "filtered_article_counts": (
                get_article_counts(
                    filtered_evidence
                )
            ),
            "exact_duplicate_removals": (
                exact_duplicate_removals
            ),
            "semantic_duplicate_removals": (
                semantic_duplicate_removals
            ),
            "raw_candidates": [
                sentence.to_dict()
                for sentence
                in raw_candidates
            ],
            "filtered_evidence": [
                sentence.to_dict()
                for sentence
                in filtered_evidence
            ],
        }

        detailed_records.append(
            detailed_record
        )

        evaluation_row = {
            "example_number": (
                example_number
            ),
            "xrag_id": get_first_value(
                example,
                "id",
                "xrag_id",
            ),
            "language": (
                get_language(example)
            ),
            "task": get_task(example),
            "question_type": (
                get_question_type(example)
            ),
            "answer_type": (
                get_answer_type(example)
            ),
            "question": question,
            "gold_answer": (
                get_gold_answer(example)
            ),
            "support_articles": ",".join(
                str(index + 1)
                for index
                in support_indexes
            ),
            "total_sentence_units": len(
                sentence_units
            ),
            "raw_support_recall": (
                raw_metrics[
                    "support_recall"
                ]
            ),
            "raw_complete_support": (
                raw_metrics[
                    "complete_support"
                ]
            ),
            "raw_any_support": (
                raw_metrics[
                    "any_support"
                ]
            ),
            "raw_unique_article_count": (
                raw_metrics[
                    "unique_article_count"
                ]
            ),
            "exact_duplicates_removed": (
                len(
                    exact_duplicate_removals
                )
            ),
            "semantic_duplicates_removed": (
                len(
                    semantic_duplicate_removals
                )
            ),
            "filtered_support_recall": (
                filtered_metrics[
                    "support_recall"
                ]
            ),
            "filtered_complete_support": (
                filtered_metrics[
                    "complete_support"
                ]
            ),
            "filtered_any_support": (
                filtered_metrics[
                    "any_support"
                ]
            ),
            "filtered_unique_article_count": (
                filtered_metrics[
                    "unique_article_count"
                ]
            ),
            "support_lost_after_filtering": (
                support_lost_after_filtering
            ),
            "support_articles_removed": (
                ",".join(
                    str(article_number)
                    for article_number
                    in support_articles_removed
                )
            ),
            "filtered_raw_ranks": ",".join(
                str(sentence.raw_rank)
                for sentence
                in filtered_evidence
            ),
            "filtered_article_sequence": (
                ",".join(
                    str(
                        sentence.article_number
                    )
                    for sentence
                    in filtered_evidence
                )
            ),
        }

        evaluation_rows.append(
            evaluation_row
        )

        if (
            example_number
            <= args.preview_count
        ):
            print_example_preview(
                example_number=example_number,
                example=example,
                support_indexes=support_indexes,
                raw_candidates=raw_candidates,
                filtered_evidence=filtered_evidence,
                exact_duplicate_count=len(
                    exact_duplicate_removals
                ),
                semantic_duplicate_count=len(
                    semantic_duplicate_removals
                ),
            )

        print(
            f"Finished "
            f"{example_number}/"
            f"{len(examples)} | "
            f"raw recall="
            f"{raw_metrics['support_recall']:.3f} | "
            f"filtered recall="
            f"{filtered_metrics['support_recall']:.3f} | "
            f"filtered complete="
            f"{filtered_metrics['complete_support']} | "
            f"final evidence="
            f"{len(filtered_evidence)}"
        )

    detailed_output_path = (
        args.output_dir
        / "v3_filtered_retrieval_details.jsonl"
    )

    evaluation_output_path = (
        args.output_dir
        / "v3_filtered_retrieval_eval.csv"
    )

    summary_output_path = (
        args.output_dir
        / "v3_filtered_retrieval_summary.txt"
    )

    write_jsonl(
        detailed_output_path,
        detailed_records,
    )

    write_csv(
        evaluation_output_path,
        evaluation_rows,
    )

    summary_text = create_summary(
        evaluation_rows
    )

    summary_output_path.write_text(
        summary_text,
        encoding="utf-8",
    )

    print(
        "\n"
        + summary_text
    )

    print(
        "\nFILES CREATED"
    )

    print(
        "-"
        * 100
    )

    print(
        detailed_output_path
    )

    print(
        evaluation_output_path
    )

    print(
        summary_output_path
    )


if __name__ == "__main__":
    main()