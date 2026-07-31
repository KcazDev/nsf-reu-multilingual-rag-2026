"""
RAG System V2: BGE-M3 Sentence Retrieval + OpenAI Generation

This script:

1. Loads the XRAG 50-question sample.
2. Splits all eight candidate articles into sentences.
3. Embeds the question and sentences with BGE-M3.
4. Retrieves Top-10 and Top-20 sentences.
5. Sends each retrieval condition to the same OpenAI model.
6. Saves detailed JSONL results.
7. Creates a manual review CSV.
8. Creates an automatic summary.

Run preview:
    python .\\src\\rag_system_v2_bge_m3.py --max-examples 1

Run first 10 with generation:
    python .\\src\\rag_system_v2_bge_m3.py --max-examples 10 --generate

Resume an interrupted run:
    python .\\src\\rag_system_v2_bge_m3.py --max-examples 10 --generate --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from openai import OpenAI

# Reuse the tested BGE-M3 retrieval functions from the evaluator
# we created in the previous step.
from evaluate_bge_m3_sentence_retrieval import (
    build_sentence_index,
    calculate_metrics_at_k,
    encode_and_rank_sentences,
    get_answer_type,
    get_articles,
    get_gold_answer,
    get_language,
    get_question,
    get_question_type,
    get_support_article_numbers_one_based,
    get_xrag_id,
    load_embedding_model,
    load_jsonl,
    prepare_nltk_resources,
    resolve_device,
)


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"

DEFAULT_SAMPLE_FILE = DATA_DIR / "xrag_eval_sample_50.jsonl"

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"

# Keep this the same model used in the earlier System V1 experiment
# so the comparison isolates the effect of retrieval.
DEFAULT_LLM_MODEL = "gpt-5.4"

DEFAULT_TOP_K_VALUES = [10, 20]
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_SEQUENCE_LENGTH = 512
DEFAULT_MAX_OUTPUT_TOKENS = 256


# ---------------------------------------------------------------------
# Controlled generation prompt
# ---------------------------------------------------------------------

SYSTEM_INSTRUCTIONS = """
You are a multilingual question-answering assistant.

Answer the question using only the evidence that directly matches the
question. The evidence may contain distractors, repeated facts, similar
events, and later updates. Ignore evidence that does not refer to the exact
event, entities, dates, or quantities asked about.

Requirements:
- Answer in the same language as the question.
- Return exactly one direct, concise sentence that answers only what was asked.
- Use no more than 35 words.
- Do not use headings, bullets, citations, explanations, caveats, or refusals.
- Preserve exact names, dates, numbers, comparisons, and sets when supported.
- For event-order questions, state only which event occurred first unless the evidence clearly supports an additional chronological explanation.
- Do not add facts that are not supported by the supplied evidence.
""".strip()


# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------

def parse_top_k_values(raw_value: str) -> List[int]:
    """
    Convert a value such as '10,20' into a sorted integer list.
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
# Output paths
# ---------------------------------------------------------------------

def create_condition_name(top_k: int) -> str:
    """
    Create a readable name for one retrieval condition.
    """
    return f"bge_m3_top{top_k}_sentences"


def create_output_stem(top_k_values: Sequence[int]) -> str:
    """
    Create a shared filename stem for the selected conditions.
    """
    top_k_text = "_".join(
        f"top{top_k}"
        for top_k in top_k_values
    )

    return f"rag_system_v2_bge_m3_{top_k_text}"


def resolve_output_paths(
    top_k_values: Sequence[int],
    details_file_argument: Optional[str],
    review_file_argument: Optional[str],
    summary_file_argument: Optional[str],
) -> Tuple[Path, Path, Path]:
    """
    Resolve custom or default output locations.
    """
    output_stem = create_output_stem(top_k_values)

    details_file = (
        Path(details_file_argument)
        if details_file_argument
        else OUTPUTS_DIR / f"{output_stem}_outputs.jsonl"
    )

    review_file = (
        Path(review_file_argument)
        if review_file_argument
        else OUTPUTS_DIR / f"{output_stem}_review.csv"
    )

    summary_file = (
        Path(summary_file_argument)
        if summary_file_argument
        else OUTPUTS_DIR / f"{output_stem}_summary.txt"
    )

    return details_file, review_file, summary_file


# ---------------------------------------------------------------------
# General file helpers
# ---------------------------------------------------------------------

def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """
    Append one result record to a JSONL file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )


def remove_file_if_present(path: Path) -> None:
    """
    Remove an old output file before a fresh run.
    """
    if path.exists():
        path.unlink()


def load_generated_records(
    details_file: Path,
    configuration_id: str,
) -> List[Dict[str, Any]]:
    """
    Load completed generation records matching this configuration.

    The latest record for each example and Top-K condition is kept.
    """
    if not details_file.exists():
        return []

    latest_records: Dict[
        Tuple[int, int],
        Dict[str, Any],
    ] = {}

    with details_file.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if record.get("configuration_id") != configuration_id:
                continue

            example_number = record.get("example_number")
            top_k = record.get("top_k")

            if example_number is None or top_k is None:
                continue

            key = (
                int(example_number),
                int(top_k),
            )

            latest_records[key] = record

    return [
        latest_records[key]
        for key in sorted(latest_records)
    ]


# ---------------------------------------------------------------------
# Retrieved sentence preparation
# ---------------------------------------------------------------------

def build_selected_sentence_results(
    ranked_indices: Sequence[int],
    ranked_scores: Sequence[float],
    sentence_texts: Sequence[str],
    sentence_metadata: Sequence[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Convert the top-ranked sentence indices into saved result objects.
    """
    effective_k = min(
        top_k,
        len(ranked_indices),
    )

    selected_results: List[Dict[str, Any]] = []

    for rank_position in range(effective_k):
        sentence_index = ranked_indices[rank_position]
        metadata = sentence_metadata[sentence_index]

        selected_results.append(
            {
                "rank": rank_position + 1,
                "similarity_score": round(
                    float(ranked_scores[rank_position]),
                    8,
                ),
                "article_number_one_based": metadata[
                    "article_number_one_based"
                ],
                "article_order_value": metadata.get(
                    "article_order_value"
                ),
                "sentence_number_within_article": metadata[
                    "sentence_number_within_article"
                ],
                "publication_date": metadata.get(
                    "publication_date"
                ),
                "source_url": metadata.get("source_url"),
                # This is saved for evaluation only.
                # It will not be shown to the language model.
                "is_support_article": metadata[
                    "is_support_article"
                ],
                "sentence_text": sentence_texts[
                    sentence_index
                ],
            }
        )

    return selected_results


def build_retrieved_context(
    selected_sentences: Sequence[Dict[str, Any]],
) -> str:
    """
    Format selected sentences for the generation prompt.

    Support-label information is intentionally excluded.
    """
    context_blocks: List[str] = []

    for result in selected_sentences:
        rank = result["rank"]
        article_number = result[
            "article_number_one_based"
        ]
        publication_date = result.get(
            "publication_date"
        )

        label = (
            f"Retrieved sentence {rank} "
            f"| Article {article_number}"
        )

        if publication_date:
            label += f" | Date: {publication_date}"

        sentence_text = str(
            result["sentence_text"]
        ).strip()

        context_blocks.append(
            f"[{label}]\n{sentence_text}"
        )

    return "\n\n".join(context_blocks)


def build_generation_prompt(
    question: str,
    retrieved_context: str,
) -> str:
    """
    Create the user message sent to the answer-generation model.
    """
    return f"""
QUESTION

{question}

RETRIEVED NEWS EVIDENCE

{retrieved_context}

FINAL ANSWER
""".strip()


# ---------------------------------------------------------------------
# OpenAI generation
# ---------------------------------------------------------------------

def extract_response_text(response: Any) -> str:
    """
    Extract text from an OpenAI Responses API result.
    """
    output_text = getattr(
        response,
        "output_text",
        None,
    )

    if output_text:
        return str(output_text).strip()

    text_parts: List[str] = []

    for output_item in getattr(
        response,
        "output",
        [],
    ):
        for content_item in getattr(
            output_item,
            "content",
            [],
        ):
            content_type = getattr(
                content_item,
                "type",
                "",
            )

            if content_type == "output_text":
                text = getattr(
                    content_item,
                    "text",
                    "",
                )

                if text:
                    text_parts.append(str(text))

    return "\n".join(text_parts).strip()


def normalize_generated_answer(answer: str) -> str:
    """
    Normalize extra whitespace without changing the answer wording.
    """
    return re.sub(
        r"\s+",
        " ",
        answer,
    ).strip()


def generate_answer(
    client: OpenAI,
    model_name: str,
    question: str,
    retrieved_context: str,
    max_output_tokens: int,
    max_retries: int,
) -> str:
    """
    Generate one answer using the OpenAI Responses API.
    """
    user_prompt = build_generation_prompt(
        question=question,
        retrieved_context=retrieved_context,
    )

    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.responses.create(
                model=model_name,
                instructions=SYSTEM_INSTRUCTIONS,
                input=user_prompt,
                max_output_tokens=max_output_tokens,
            )

            answer = extract_response_text(response)
            answer = normalize_generated_answer(answer)

            if not answer:
                raise RuntimeError(
                    "The model returned an empty answer."
                )

            return answer

        except Exception as error:
            last_error = error

            if attempt >= max_retries:
                break

            wait_seconds = 2 ** attempt

            print(
                f"    API attempt {attempt} failed: "
                f"{error}"
            )
            print(
                f"    Retrying in {wait_seconds} seconds..."
            )

            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Generation failed after {max_retries} attempts: "
        f"{last_error}"
    )


# ---------------------------------------------------------------------
# Answer measurements
# ---------------------------------------------------------------------

def count_words(text: str) -> int:
    """
    Count whitespace-separated words.

    This is suitable for the Arabic and Spanish answers in this sample.
    """
    return len(
        re.findall(
            r"\S+",
            text.strip(),
        )
    )


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

def create_configuration_id(
    sample_file: Path,
    embedding_model: str,
    llm_model: str,
    top_k_values: Sequence[int],
    max_sequence_length: int,
) -> str:
    """
    Create a stable configuration identifier for resume support.
    """
    return "|".join(
        [
            f"sample={sample_file.name}",
            f"embedding={embedding_model}",
            f"llm={llm_model}",
            (
                "top_k="
                + ",".join(
                    str(value)
                    for value in top_k_values
                )
            ),
            f"max_length={max_sequence_length}",
            "granularity=sentence",
            "prompt=controlled_v2",
        ]
    )


# ---------------------------------------------------------------------
# One-condition result creation
# ---------------------------------------------------------------------

def create_generation_record(
    record: Dict[str, Any],
    example_number: int,
    top_k: int,
    embedding_model: str,
    llm_model: str,
    device: str,
    configuration_id: str,
    sentence_count: int,
    support_article_numbers: Sequence[int],
    retrieval_metrics: Dict[str, Any],
    selected_sentences: Sequence[Dict[str, Any]],
    generated_answer: str,
) -> Dict[str, Any]:
    """
    Create one saved result for one question and one Top-K condition.
    """
    answer_word_count = count_words(
        generated_answer
    )

    return {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "configuration_id": configuration_id,
        "system_version": "V2",
        "system_name": "BGE-M3 CROSS-style RAG",
        "condition": create_condition_name(top_k),
        "retrieval_method": "BGE-M3 dense retrieval",
        "retrieval_granularity": "sentence",
        "embedding_model": embedding_model,
        "generation_model": llm_model,
        "device": device,
        "top_k": top_k,
        "example_number": example_number,
        "xrag_id": get_xrag_id(record),
        "language": get_language(record),
        "task": record.get("task"),
        "question_type": get_question_type(record),
        "answer_type": get_answer_type(record),
        "crossdoc_type": record.get(
            "crossdoc_type"
        ),
        "question": get_question(record),
        "gold_answer": get_gold_answer(record),
        "article_count": len(
            get_articles(record)
        ),
        "total_sentence_units": sentence_count,
        "support_article_numbers_one_based": list(
            support_article_numbers
        ),
        "retrieval_metrics": retrieval_metrics,
        "retrieved_sentences": list(
            selected_sentences
        ),
        "model_answer": generated_answer,
        "answer_word_count": answer_word_count,
        "over_35_words": answer_word_count > 35,
        "generation_error": "",
    }


# ---------------------------------------------------------------------
# Terminal display
# ---------------------------------------------------------------------

def sentence_preview(
    text: str,
    maximum_length: int = 170,
) -> str:
    """
    Create a compact one-line sentence preview.
    """
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if len(text) <= maximum_length:
        return text

    return text[: maximum_length - 3] + "..."


def print_retrieval_condition(
    top_k: int,
    retrieval_metrics: Dict[str, Any],
    selected_sentences: Sequence[Dict[str, Any]],
) -> None:
    """
    Print retrieval metrics and the first few retrieved sentences.
    """
    recall = retrieval_metrics.get(
        "support_article_recall_at_k"
    )

    recall_text = (
        f"{float(recall):.3f}"
        if recall is not None
        else "N/A"
    )

    print()
    print(f"  BGE-M3 Top-{top_k}")
    print(
        f"    Sentence precision: "
        f"{retrieval_metrics['sentence_precision_at_k']:.3f}"
    )
    print(
        f"    Support-article recall: {recall_text}"
    )
    print(
        f"    Complete support: "
        f"{retrieval_metrics['complete_support_at_k']}"
    )
    print(
        f"    Support articles retrieved: "
        f"{retrieval_metrics['support_articles_retrieved']}"
    )

    print("    First three retrieved sentences:")

    for result in selected_sentences[:3]:
        marker = (
            "support article"
            if result["is_support_article"]
            else "non-support article"
        )

        print(
            f"      Rank {result['rank']} | "
            f"Article "
            f"{result['article_number_one_based']} | "
            f"{marker} | "
            f"score={result['similarity_score']:.4f}"
        )

        print(
            "        "
            + sentence_preview(
                result["sentence_text"]
            )
        )


# ---------------------------------------------------------------------
# Review CSV
# ---------------------------------------------------------------------

def record_to_review_row(
    record: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Flatten one generated record for manual review.
    """
    retrieval_metrics = record.get(
        "retrieval_metrics",
        {},
    )

    retrieved_sentences = record.get(
        "retrieved_sentences",
        [],
    )

    return {
        "example_number": record.get(
            "example_number"
        ),
        "xrag_id": record.get("xrag_id"),
        "condition": record.get("condition"),
        "top_k": record.get("top_k"),
        "language": record.get("language"),
        "task": record.get("task"),
        "question_type": record.get(
            "question_type"
        ),
        "answer_type": record.get(
            "answer_type"
        ),
        "crossdoc_type": record.get(
            "crossdoc_type"
        ),
        "question": record.get("question"),
        "gold_answer": record.get(
            "gold_answer"
        ),
        "model_answer": record.get(
            "model_answer"
        ),
        "answer_word_count": record.get(
            "answer_word_count"
        ),
        "over_35_words": record.get(
            "over_35_words"
        ),
        "support_articles": json.dumps(
            record.get(
                "support_article_numbers_one_based",
                [],
            ),
            ensure_ascii=False,
        ),
        "support_articles_retrieved": json.dumps(
            retrieval_metrics.get(
                "support_articles_retrieved",
                [],
            ),
            ensure_ascii=False,
        ),
        "support_article_recall": (
            retrieval_metrics.get(
                "support_article_recall_at_k"
            )
        ),
        "complete_support": (
            retrieval_metrics.get(
                "complete_support_at_k"
            )
        ),
        "sentence_precision": (
            retrieval_metrics.get(
                "sentence_precision_at_k"
            )
        ),
        "retrieved_article_sequence": json.dumps(
            [
                item.get(
                    "article_number_one_based"
                )
                for item in retrieved_sentences
            ],
            ensure_ascii=False,
        ),
        "retrieved_sentence_texts": json.dumps(
            [
                item.get("sentence_text")
                for item in retrieved_sentences
            ],
            ensure_ascii=False,
        ),
        "manual_score_0_to_2": "",
        "correctness_label": "",
        "language_correct_yes_no": "",
        "retrieval_error_type": "",
        "generation_error_type": "",
        "notes": "",
    }


def write_review_csv(
    records: Sequence[Dict[str, Any]],
    review_file: Path,
) -> None:
    """
    Write the manual scoring worksheet.
    """
    review_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sorted_records = sorted(
        records,
        key=lambda item: (
            int(item.get("example_number", 0)),
            int(item.get("top_k", 0)),
        ),
    )

    rows = [
        record_to_review_row(record)
        for record in sorted_records
    ]

    if not rows:
        return

    with review_file.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# Automatic summary
# ---------------------------------------------------------------------

def write_automatic_summary(
    records: Sequence[Dict[str, Any]],
    summary_file: Path,
    sample_file: Path,
    embedding_model: str,
    llm_model: str,
    top_k_values: Sequence[int],
) -> None:
    """
    Write retrieval and format statistics.

    Correctness is not calculated until manual scoring is completed.
    """
    summary_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines: List[str] = []

    lines.append(
        "RAG SYSTEM V2: BGE-M3 TOP-10 VS TOP-20"
    )
    lines.append("=" * 78)
    lines.append(
        f"Created UTC: "
        f"{datetime.now(timezone.utc).isoformat()}"
    )
    lines.append(f"Sample file: {sample_file}")
    lines.append(
        f"Embedding model: {embedding_model}"
    )
    lines.append(
        f"Generation model: {llm_model}"
    )
    lines.append(
        f"Retrieval granularity: sentence"
    )
    lines.append(
        f"Conditions: {list(top_k_values)}"
    )
    lines.append(
        f"Total generated records: {len(records)}"
    )
    lines.append("")

    for top_k in top_k_values:
        condition_records = [
            record
            for record in records
            if int(record.get("top_k", -1))
            == top_k
        ]

        lines.append(f"BGE-M3 Top-{top_k}")
        lines.append("-" * 78)

        if not condition_records:
            lines.append("No completed records.")
            lines.append("")
            continue

        recall_values: List[float] = []
        complete_count = 0
        any_support_count = 0
        over_35_count = 0
        word_counts: List[int] = []

        for record in condition_records:
            metrics = record.get(
                "retrieval_metrics",
                {},
            )

            recall = metrics.get(
                "support_article_recall_at_k"
            )

            if recall is not None:
                recall_values.append(
                    float(recall)
                )

            if metrics.get(
                "complete_support_at_k"
            ):
                complete_count += 1

            if metrics.get(
                "support_sentence_count",
                0,
            ) > 0:
                any_support_count += 1

            if record.get("over_35_words"):
                over_35_count += 1

            word_counts.append(
                int(
                    record.get(
                        "answer_word_count",
                        0,
                    )
                )
            )

        record_count = len(
            condition_records
        )

        average_recall = (
            mean(recall_values)
            if recall_values
            else 0.0
        )

        average_word_count = (
            mean(word_counts)
            if word_counts
            else 0.0
        )

        lines.append(
            f"Examples generated: {record_count}"
        )
        lines.append(
            f"Average support-article recall: "
            f"{average_recall:.3f}"
        )
        lines.append(
            f"Any support retrieved: "
            f"{any_support_count}/{record_count}"
        )
        lines.append(
            f"Complete support retrieved: "
            f"{complete_count}/{record_count}"
        )
        lines.append(
            f"Average answer word count: "
            f"{average_word_count:.1f}"
        )
        lines.append(
            f"Answers over 35 words: "
            f"{over_35_count}/{record_count}"
        )
        lines.append("")

    lines.append("Manual scoring still required")
    lines.append("-" * 78)
    lines.append(
        "Use the review CSV to assign each answer a score:"
    )
    lines.append(
        "2 = correct, 1 = partially correct, 0 = incorrect."
    )
    lines.append(
        "Compare Top-10 and Top-20 answer accuracy after "
        "the same questions have been scored."
    )

    with summary_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write("\n".join(lines))


# ---------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run BGE-M3 sentence retrieval followed by "
            "OpenAI answer generation on XRAG."
        )
    )

    parser.add_argument(
        "--sample-file",
        default=str(DEFAULT_SAMPLE_FILE),
        help="Path to the XRAG JSONL sample.",
    )

    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Sentence Transformers embedding model.",
    )

    parser.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
        help=(
            "OpenAI generation model. Keep this identical "
            "to System V1 for a fair comparison."
        ),
    )

    parser.add_argument(
        "--top-k-values",
        type=parse_top_k_values,
        default=DEFAULT_TOP_K_VALUES,
        help=(
            "Comma-separated sentence caps. "
            "Default: 10,20"
        ),
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=1,
        help=(
            "Number of XRAG examples to process. "
            "Start with 1, then use 10."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="BGE-M3 sentence encoding batch size.",
    )

    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=DEFAULT_MAX_SEQUENCE_LENGTH,
        help="Maximum BGE-M3 encoder sequence length.",
    )

    parser.add_argument(
        "--minimum-characters",
        type=int,
        default=10,
        help="Minimum sentence length after splitting.",
    )

    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Maximum generation output tokens.",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum API attempts per generation.",
    )

    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.5,
        help=(
            "Delay in seconds between successful API calls."
        ),
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device used for BGE-M3.",
    )

    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show embedding progress bars.",
    )

    parser.add_argument(
        "--generate",
        action="store_true",
        help=(
            "Make OpenAI API calls. Without this flag, "
            "the script only previews retrieval."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted generation run without "
            "repeating completed API calls."
        ),
    )

    parser.add_argument(
        "--details-file",
        default=None,
        help="Optional custom JSONL output path.",
    )

    parser.add_argument(
        "--review-file",
        default=None,
        help="Optional custom review CSV path.",
    )

    parser.add_argument(
        "--summary-file",
        default=None,
        help="Optional custom summary path.",
    )

    args = parser.parse_args()

    if args.max_examples <= 0:
        raise ValueError(
            "--max-examples must be greater than zero."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be greater than zero."
        )

    if args.max_retries <= 0:
        raise ValueError(
            "--max-retries must be greater than zero."
        )

    sample_file = Path(args.sample_file)
    top_k_values = list(args.top_k_values)

    details_file, review_file, summary_file = (
        resolve_output_paths(
            top_k_values=top_k_values,
            details_file_argument=args.details_file,
            review_file_argument=args.review_file,
            summary_file_argument=args.summary_file,
        )
    )

    OUTPUTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    load_dotenv(
        dotenv_path=ROOT_DIR / ".env"
    )

    if args.generate and not os.getenv(
        "OPENAI_API_KEY"
    ):
        raise RuntimeError(
            "OPENAI_API_KEY was not found. "
            "Make sure it is saved in the project .env file."
        )

    prepare_nltk_resources()

    all_sample_records = load_jsonl(
        sample_file
    )

    selected_records = all_sample_records[
        : args.max_examples
    ]

    device = resolve_device(args.device)

    configuration_id = create_configuration_id(
        sample_file=sample_file,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        top_k_values=top_k_values,
        max_sequence_length=args.max_sequence_length,
    )

    if args.generate and not args.resume:
        remove_file_if_present(details_file)
        remove_file_if_present(review_file)
        remove_file_if_present(summary_file)

    completed_records = (
        load_generated_records(
            details_file=details_file,
            configuration_id=configuration_id,
        )
        if args.generate and args.resume
        else []
    )

    completed_keys = {
        (
            int(record["example_number"]),
            int(record["top_k"]),
        )
        for record in completed_records
    }

    print()
    print("RAG SYSTEM V2: BGE-M3 SENTENCE RETRIEVAL")
    print("=" * 78)
    print(f"Sample file: {sample_file}")
    print(
        f"Examples selected: "
        f"{len(selected_records)}"
    )
    print(
        f"Embedding model: "
        f"{args.embedding_model}"
    )
    print(f"Generation model: {args.llm_model}")
    print(f"Device: {device}")
    print(
        f"Sentence caps: {top_k_values}"
    )
    print(f"Generation enabled: {args.generate}")
    print(f"Resume enabled: {args.resume}")
    print(
        f"Previously completed conditions: "
        f"{len(completed_keys)}"
    )

    embedding_model = load_embedding_model(
        model_name=args.embedding_model,
        device=device,
        max_sequence_length=args.max_sequence_length,
    )

    client: Optional[OpenAI] = None

    if args.generate:
        client = OpenAI()

    generated_this_run = 0
    failed_this_run = 0
    skipped_this_run = 0

    for example_number, record in enumerate(
        selected_records,
        start=1,
    ):
        pending_top_k_values = [
            top_k
            for top_k in top_k_values
            if (
                example_number,
                top_k,
            )
            not in completed_keys
        ]

        if args.generate and not pending_top_k_values:
            print()
            print(
                f"Skipping Example {example_number}: "
                f"all conditions are complete."
            )

            skipped_this_run += len(
                top_k_values
            )
            continue

        if not args.generate:
            pending_top_k_values = top_k_values

        question = get_question(record)
        articles = get_articles(record)

        support_article_numbers = (
            get_support_article_numbers_one_based(
                record=record,
                article_count=len(articles),
            )
        )

        print()
        print("#" * 78)
        print(
            f"Example {example_number} | "
            f"XRAG ID: {get_xrag_id(record)}"
        )
        print(
            f"Language: {get_language(record)} | "
            f"Task: {record.get('task')} | "
            f"Question type: "
            f"{get_question_type(record)}"
        )
        print(f"Question: {question}")
        print(
            f"True support articles: "
            f"{support_article_numbers}"
        )

        sentence_texts, sentence_metadata = (
            build_sentence_index(
                record=record,
                minimum_characters=(
                    args.minimum_characters
                ),
            )
        )

        print(
            f"Sentence units indexed: "
            f"{len(sentence_texts)}"
        )

        ranked_indices, ranked_scores = (
            encode_and_rank_sentences(
                model=embedding_model,
                question=question,
                sentence_texts=sentence_texts,
                batch_size=args.batch_size,
                show_progress=args.show_progress,
            )
        )

        for top_k in pending_top_k_values:
            retrieval_metrics = (
                calculate_metrics_at_k(
                    ranked_indices=ranked_indices,
                    sentence_metadata=(
                        sentence_metadata
                    ),
                    support_article_numbers=(
                        support_article_numbers
                    ),
                    k=top_k,
                )
            )

            selected_sentences = (
                build_selected_sentence_results(
                    ranked_indices=ranked_indices,
                    ranked_scores=ranked_scores,
                    sentence_texts=sentence_texts,
                    sentence_metadata=(
                        sentence_metadata
                    ),
                    top_k=top_k,
                )
            )

            print_retrieval_condition(
                top_k=top_k,
                retrieval_metrics=retrieval_metrics,
                selected_sentences=selected_sentences,
            )

            if not args.generate:
                continue

            retrieved_context = (
                build_retrieved_context(
                    selected_sentences
                )
            )

            print(
                f"    Generating answer with "
                f"{args.llm_model}..."
            )

            try:
                assert client is not None

                generated_answer = generate_answer(
                    client=client,
                    model_name=args.llm_model,
                    question=question,
                    retrieved_context=(
                        retrieved_context
                    ),
                    max_output_tokens=(
                        args.max_output_tokens
                    ),
                    max_retries=args.max_retries,
                )

            except Exception as error:
                failed_this_run += 1

                print(
                    f"    GENERATION FAILED: {error}"
                )
                print(
                    "    This condition was not saved. "
                    "Use --resume to retry it."
                )

                continue

            generation_record = (
                create_generation_record(
                    record=record,
                    example_number=example_number,
                    top_k=top_k,
                    embedding_model=(
                        args.embedding_model
                    ),
                    llm_model=args.llm_model,
                    device=device,
                    configuration_id=(
                        configuration_id
                    ),
                    sentence_count=len(
                        sentence_texts
                    ),
                    support_article_numbers=(
                        support_article_numbers
                    ),
                    retrieval_metrics=(
                        retrieval_metrics
                    ),
                    selected_sentences=(
                        selected_sentences
                    ),
                    generated_answer=(
                        generated_answer
                    ),
                )
            )

            append_jsonl(
                details_file,
                generation_record,
            )

            completed_keys.add(
                (
                    example_number,
                    top_k,
                )
            )

            generated_this_run += 1

            print(
                f"    Answer: {generated_answer}"
            )
            print(
                f"    Word count: "
                f"{generation_record['answer_word_count']}"
            )

            if args.request_delay > 0:
                time.sleep(args.request_delay)

    if not args.generate:
        print()
        print("=" * 78)
        print("RETRIEVAL PREVIEW FINISHED")
        print(
            "No API calls were made and no generation "
            "records were saved."
        )
        return

    final_records = load_generated_records(
        details_file=details_file,
        configuration_id=configuration_id,
    )

    allowed_example_numbers = set(
        range(
            1,
            len(selected_records) + 1,
        )
    )

    final_records = [
        record
        for record in final_records
        if int(
            record.get(
                "example_number",
                -1,
            )
        )
        in allowed_example_numbers
    ]

    write_review_csv(
        records=final_records,
        review_file=review_file,
    )

    write_automatic_summary(
        records=final_records,
        summary_file=summary_file,
        sample_file=sample_file,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        top_k_values=top_k_values,
    )

    print()
    print("=" * 78)
    print("RAG SYSTEM V2 RUN FINISHED")
    print(
        f"Generated this run: "
        f"{generated_this_run}"
    )
    print(
        f"Failed this run: "
        f"{failed_this_run}"
    )
    print(
        f"Skipped this run: "
        f"{skipped_this_run}"
    )
    print(
        f"Total completed records: "
        f"{len(final_records)}"
    )
    print(f"Detailed results: {details_file}")
    print(f"Review CSV: {review_file}")
    print(f"Automatic summary: {summary_file}")


if __name__ == "__main__":
    main()