"""
System V3: filtered BGE-M3 retrieval followed by GPT generation.

Pipeline:
    XRAG question
    -> BGE-M3 raw Top-20 retrieval
    -> exact duplicate removal
    -> semantic duplicate removal
    -> article diversity and MMR filtering
    -> final Top-12 evidence
    -> GPT answer generation
    -> CSV for manual 2/1/0 scoring

This script imports the tested retrieval functions from:
    rag_system_v3_bge_m3_filtered.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from sentence_transformers import SentenceTransformer

import rag_system_v3_bge_m3_filtered as v3


# ============================================================================
# PATHS AND GENERATION SETTINGS
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
    / "v3_generation"
)

GENERATION_MODEL = "gpt-5.4"

MAX_OUTPUT_TOKENS = 300

MAX_API_RETRIES = 4


# ============================================================================
# CONTROLLED GENERATION PROMPT
# ============================================================================

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


# ============================================================================
# COMMAND-LINE ARGUMENTS
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run V3 filtered Top-12 retrieval and GPT answer generation."
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
        help="Directory where V3 generation files will be saved.",
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help=(
            "Maximum number of examples to process. "
            "Use 0 to process the entire sample."
        ),
    )

    parser.add_argument(
        "--preview-count",
        type=int,
        default=1,
        help="Number of examples whose evidence will be printed.",
    )

    parser.add_argument(
        "--generation-model",
        default=GENERATION_MODEL,
        help="OpenAI model used for answer generation.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="BGE-M3 embedding batch size.",
    )

    parser.add_argument(
        "--device",
        default=None,
        help='Optional embedding device such as "cpu" or "cuda".',
    )

    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show BGE-M3 embedding progress bars.",
    )

    parser.add_argument(
        "--generate",
        action="store_true",
        help=(
            "Call the OpenAI API. Without this flag, "
            "the script only previews retrieval."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip examples already stored in the output JSONL file.",
    )

    return parser.parse_args()


def validate_arguments(
    args: argparse.Namespace,
) -> None:
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

    # These are the selected V3 settings.
    expected_settings = {
        "RAW_TOP_K": 20,
        "FINAL_TOP_K": 12,
        "SEMANTIC_DUPLICATE_THRESHOLD": 0.92,
        "MAX_SENTENCES_PER_ARTICLE": 4,
        "MMR_LAMBDA": 0.85,
    }

    actual_settings = {
        "RAW_TOP_K": v3.RAW_TOP_K,
        "FINAL_TOP_K": v3.FINAL_TOP_K,
        "SEMANTIC_DUPLICATE_THRESHOLD": (
            v3.SEMANTIC_DUPLICATE_THRESHOLD
        ),
        "MAX_SENTENCES_PER_ARTICLE": (
            v3.MAX_SENTENCES_PER_ARTICLE
        ),
        "MMR_LAMBDA": v3.MMR_LAMBDA,
    }

    mismatches: list[str] = []

    for setting_name, expected_value in expected_settings.items():
        actual_value = actual_settings[setting_name]

        if actual_value != expected_value:
            mismatches.append(
                f"{setting_name}: "
                f"expected {expected_value}, "
                f"found {actual_value}"
            )

    if mismatches:
        raise ValueError(
            "The V3 retrieval file does not contain the selected "
            "final configuration:\n- "
            + "\n- ".join(mismatches)
        )


# ============================================================================
# OPENAI CLIENT
# ============================================================================

def create_openai_client() -> OpenAI:
    load_dotenv(
        PROJECT_ROOT
        / ".env"
    )

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY was not found. "
            "Make sure it is stored in the project .env file."
        )

    return OpenAI(
        api_key=api_key
    )


# ============================================================================
# PROMPT CONSTRUCTION
# ============================================================================

def build_evidence_text(
    filtered_evidence: list[v3.SentenceUnit],
) -> str:
    evidence_blocks: list[str] = []

    for sentence in filtered_evidence:
        publication_date = (
            sentence.publication_date.strip()
            or "unknown"
        )

        evidence_blocks.append(
            (
                f"[Evidence {sentence.final_rank} | "
                f"Article {sentence.article_number} | "
                f"Date: {publication_date}]\n"
                f"{sentence.text}"
            )
        )

    return "\n\n".join(
        evidence_blocks
    )


def build_user_prompt(
    example: dict[str, Any],
    filtered_evidence: list[v3.SentenceUnit],
) -> str:
    language = v3.get_language(
        example
    )

    question = v3.get_question(
        example
    )

    evidence_text = build_evidence_text(
        filtered_evidence
    )

    return (
        f"Question language: {language}\n\n"
        f"Question:\n"
        f"{question}\n\n"
        f"Retrieved evidence:\n"
        f"{evidence_text}\n\n"
        f"Return only the final answer."
    )


# ============================================================================
# GPT GENERATION
# ============================================================================

def generate_answer(
    client: OpenAI,
    model_name: str,
    user_prompt: str,
) -> tuple[str, str]:
    """
    Generate one answer and return the answer plus the API response ID.
    """

    last_error: Exception | None = None

    for attempt in range(
        1,
        MAX_API_RETRIES + 1,
    ):
        try:
            response = client.responses.create(
                model=model_name,
                instructions=SYSTEM_INSTRUCTIONS,
                input=user_prompt,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                store=False,
            )

            answer = (
                response.output_text
                or ""
            ).strip()

            if not answer:
                raise RuntimeError(
                    "The API returned an empty answer."
                )

            return (
                answer,
                response.id,
            )

        except (
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
        ) as error:
            last_error = error

            if attempt == MAX_API_RETRIES:
                break

            wait_seconds = (
                2 ** attempt
            )

            print(
                f"Temporary API error on attempt "
                f"{attempt}/{MAX_API_RETRIES}: "
                f"{error}"
            )

            print(
                f"Retrying in "
                f"{wait_seconds} seconds..."
            )

            time.sleep(
                wait_seconds
            )

    raise RuntimeError(
        f"Generation failed after "
        f"{MAX_API_RETRIES} attempts: "
        f"{last_error}"
    )


# ============================================================================
# OUTPUT HELPERS
# ============================================================================

def count_words(
    text: str,
) -> int:
    return len(
        text.strip().split()
    )


def read_existing_jsonl(
    file_path: Path,
) -> list[dict[str, Any]]:
    if not file_path.exists():
        return []

    records: list[dict[str, Any]] = []

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        for line in input_file:
            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    return records


def write_jsonl(
    output_path: Path,
    records: list[dict[str, Any]],
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def write_review_csv(
    output_path: Path,
    records: list[dict[str, Any]],
) -> None:
    fieldnames = [
        "example_number",
        "xrag_id",
        "language",
        "task",
        "question_type",
        "answer_type",
        "question",
        "gold_answer",
        "generated_answer",
        "manual_score_2_1_0",
        "error_category",
        "review_notes",
        "correct_language",
        "answer_word_count",
        "over_35_words",
        "raw_support_recall",
        "raw_complete_support",
        "filtered_support_recall",
        "filtered_complete_support",
        "support_lost_after_filtering",
        "support_articles",
        "raw_support_articles_found",
        "filtered_support_articles_found",
        "filtered_article_sequence",
        "filtered_raw_ranks",
        "exact_duplicates_removed",
        "semantic_duplicates_removed",
        "generation_model",
        "response_id",
    ]

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        sorted_records = sorted(
            records,
            key=lambda record: (
                record["example_number"]
            ),
        )

        for record in sorted_records:
            retrieval = record[
                "retrieval"
            ]

            raw_metrics = retrieval[
                "raw_metrics"
            ]

            filtered_metrics = retrieval[
                "filtered_metrics"
            ]

            writer.writerow(
                {
                    "example_number": (
                        record["example_number"]
                    ),
                    "xrag_id": (
                        record["xrag_id"]
                    ),
                    "language": (
                        record["language"]
                    ),
                    "task": (
                        record["task"]
                    ),
                    "question_type": (
                        record["question_type"]
                    ),
                    "answer_type": (
                        record["answer_type"]
                    ),
                    "question": (
                        record["question"]
                    ),
                    "gold_answer": (
                        record["gold_answer"]
                    ),
                    "generated_answer": (
                        record["generated_answer"]
                    ),
                    "manual_score_2_1_0": "",
                    "error_category": "",
                    "review_notes": "",
                    "correct_language": "",
                    "answer_word_count": (
                        record["answer_word_count"]
                    ),
                    "over_35_words": (
                        record["over_35_words"]
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
                    "support_lost_after_filtering": (
                        retrieval[
                            "support_lost_after_filtering"
                        ]
                    ),
                    "support_articles": ",".join(
                        str(value)
                        for value
                        in record[
                            "support_articles"
                        ]
                    ),
                    "raw_support_articles_found": (
                        ",".join(
                            str(value)
                            for value
                            in raw_metrics[
                                "support_articles_found"
                            ]
                        )
                    ),
                    "filtered_support_articles_found": (
                        ",".join(
                            str(value)
                            for value
                            in filtered_metrics[
                                "support_articles_found"
                            ]
                        )
                    ),
                    "filtered_article_sequence": (
                        ",".join(
                            str(
                                evidence[
                                    "article_number"
                                ]
                            )
                            for evidence
                            in record[
                                "filtered_evidence"
                            ]
                        )
                    ),
                    "filtered_raw_ranks": (
                        ",".join(
                            str(
                                evidence[
                                    "raw_rank"
                                ]
                            )
                            for evidence
                            in record[
                                "filtered_evidence"
                            ]
                        )
                    ),
                    "exact_duplicates_removed": (
                        retrieval[
                            "exact_duplicates_removed"
                        ]
                    ),
                    "semantic_duplicates_removed": (
                        retrieval[
                            "semantic_duplicates_removed"
                        ]
                    ),
                    "generation_model": (
                        record[
                            "generation_model"
                        ]
                    ),
                    "response_id": (
                        record[
                            "response_id"
                        ]
                    ),
                }
            )


def create_summary(
    records: list[dict[str, Any]],
    generation_model: str,
) -> str:
    if not records:
        return (
            "No V3 generation records "
            "were produced."
        )

    example_count = len(
        records
    )

    average_raw_recall = (
        sum(
            record[
                "retrieval"
            ][
                "raw_metrics"
            ][
                "support_recall"
            ]
            for record
            in records
        )
        / example_count
    )

    average_filtered_recall = (
        sum(
            record[
                "retrieval"
            ][
                "filtered_metrics"
            ][
                "support_recall"
            ]
            for record
            in records
        )
        / example_count
    )

    raw_complete_count = sum(
        bool(
            record[
                "retrieval"
            ][
                "raw_metrics"
            ][
                "complete_support"
            ]
        )
        for record
        in records
    )

    filtered_complete_count = sum(
        bool(
            record[
                "retrieval"
            ][
                "filtered_metrics"
            ][
                "complete_support"
            ]
        )
        for record
        in records
    )

    support_loss_count = sum(
        bool(
            record[
                "retrieval"
            ][
                "support_lost_after_filtering"
            ]
        )
        for record
        in records
    )

    average_answer_words = (
        sum(
            record[
                "answer_word_count"
            ]
            for record
            in records
        )
        / example_count
    )

    over_35_count = sum(
        bool(
            record[
                "over_35_words"
            ]
        )
        for record
        in records
    )

    summary_lines = [
        "SYSTEM V3 FILTERED TOP-12 GENERATION SUMMARY",
        "=" * 78,
        f"Generated examples: {example_count}",
        f"Embedding model: {v3.MODEL_NAME}",
        f"Generation model: {generation_model}",
        f"Raw candidate pool: Top-{v3.RAW_TOP_K}",
        (
            "Final filtered evidence: "
            f"Top-{v3.FINAL_TOP_K}"
        ),
        (
            "Semantic duplicate threshold: "
            f"{v3.SEMANTIC_DUPLICATE_THRESHOLD}"
        ),
        (
            "Maximum sentences per article: "
            f"{v3.MAX_SENTENCES_PER_ARTICLE}"
        ),
        f"MMR lambda: {v3.MMR_LAMBDA}",
        "",
        "RETRIEVAL FOR GENERATED EXAMPLES",
        "-" * 78,
        (
            "Raw Top-20 average support recall: "
            f"{average_raw_recall:.3f}"
        ),
        (
            "Raw Top-20 complete support: "
            f"{raw_complete_count}/"
            f"{example_count}"
        ),
        (
            "Filtered Top-12 average support recall: "
            f"{average_filtered_recall:.3f}"
        ),
        (
            "Filtered Top-12 complete support: "
            f"{filtered_complete_count}/"
            f"{example_count}"
        ),
        (
            "Support lost after filtering: "
            f"{support_loss_count}/"
            f"{example_count}"
        ),
        "",
        "GENERATION FORMAT",
        "-" * 78,
        (
            "Average answer length: "
            f"{average_answer_words:.1f} words"
        ),
        (
            "Answers over 35 words: "
            f"{over_35_count}/"
            f"{example_count}"
        ),
        "",
        "MANUAL REVIEW",
        "-" * 78,
        "Open the review CSV and score each answer:",
        "2 = fully correct",
        "1 = partially correct",
        "0 = incorrect",
        "",
        "Suggested error categories:",
        "wrong_number",
        "wrong_date",
        "wrong_event",
        "incomplete_answer",
        "unsupported_addition",
        "retrieval_failure",
        "filtering_failure",
        "generation_failure",
        "reasoning_failure",
    ]

    return "\n".join(
        summary_lines
    )


# ============================================================================
# TERMINAL PREVIEW
# ============================================================================

def print_preview(
    example_number: int,
    example: dict[str, Any],
    filtered_evidence: list[v3.SentenceUnit],
    generated_answer: str | None,
) -> None:
    print(
        "\n"
        + "#"
        * 100
    )

    print(
        f"EXAMPLE {example_number}"
    )

    print(
        f"Language: "
        f"{v3.get_language(example)}"
    )

    print(
        f"Question: "
        f"{v3.get_question(example)}"
    )

    print(
        f"Gold answer: "
        f"{v3.get_gold_answer(example)}"
    )

    print(
        f"\nFILTERED TOP-"
        f"{len(filtered_evidence)} EVIDENCE"
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
            f"raw={sentence.raw_rank} "
            f"rel="
            f"{sentence.relevance_score:.4f} "
            f"mmr="
            f"{sentence.mmr_score:.4f} "
            f"[{evidence_label}] "
            f"{sentence.text[:200]}"
        )

    if generated_answer is not None:
        print(
            "\nGENERATED ANSWER:"
        )

        print(
            generated_answer
        )


# ============================================================================
# MAIN PROGRAM
# ============================================================================

def main() -> None:
    args = parse_arguments()

    validate_arguments(
        args
    )

    examples = v3.load_jsonl(
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

    output_jsonl_path = (
        args.output_dir
        / "v3_filtered_top12_generation_outputs.jsonl"
    )

    review_csv_path = (
        args.output_dir
        / "v3_filtered_top12_generation_review.csv"
    )

    summary_path = (
        args.output_dir
        / "v3_filtered_top12_generation_summary.txt"
    )

    if args.resume:
        generated_records = read_existing_jsonl(
            output_jsonl_path
        )

    else:
        generated_records = []

    completed_example_numbers = {
        int(
            record[
                "example_number"
            ]
        )
        for record
        in generated_records
    }

    print(
        "SYSTEM V3: FILTERED TOP-12 "
        "+ GPT GENERATION"
    )

    print(
        "="
        * 100
    )

    print(
        f"Sample file: "
        f"{args.sample_file}"
    )

    print(
        f"Examples requested: "
        f"{len(examples)}"
    )

    print(
        f"Embedding model: "
        f"{v3.MODEL_NAME}"
    )

    print(
        f"Generation model: "
        f"{args.generation_model}"
    )

    print(
        f"Raw candidates: "
        f"Top-{v3.RAW_TOP_K}"
    )

    print(
        f"Final evidence: "
        f"Top-{v3.FINAL_TOP_K}"
    )

    print(
        f"Generate answers: "
        f"{args.generate}"
    )

    print(
        f"Resume: "
        f"{args.resume}"
    )

    print(
        "\nLoading BGE-M3..."
    )

    embedding_model = SentenceTransformer(
        v3.MODEL_NAME,
        device=args.device,
    )

    print(
        "BGE-M3 loaded successfully."
    )

    openai_client = (
        create_openai_client()
        if args.generate
        else None
    )

    for example_number, example in enumerate(
        examples,
        start=1,
    ):
        if (
            example_number
            in completed_example_numbers
        ):
            print(
                f"Skipping completed example "
                f"{example_number}."
            )

            continue

        question = v3.get_question(
            example
        )

        articles = example.get(
            "articles",
            [],
        )

        if not isinstance(
            articles,
            list,
        ) or not articles:
            raise ValueError(
                f"Example {example_number} "
                "does not contain usable articles."
            )

        support_indexes = (
            v3.get_support_indexes(
                example,
                article_count=len(articles),
            )
        )

        sentence_units = (
            v3.build_sentence_units(
                example,
                support_indexes,
            )
        )

        question_embedding = (
            v3.embed_question_and_sentences(
                model=embedding_model,
                question=question,
                sentence_units=sentence_units,
                batch_size=args.batch_size,
                show_progress=args.show_progress,
            )
        )

        raw_candidates = (
            v3.retrieve_raw_top_20(
                question_embedding,
                sentence_units,
            )
        )

        (
            exact_kept,
            exact_duplicate_removals,
        ) = v3.remove_exact_duplicates(
            raw_candidates
        )

        (
            semantic_kept,
            semantic_duplicate_removals,
        ) = v3.remove_semantic_duplicates(
            exact_kept
        )

        filtered_evidence = (
            v3.select_final_evidence_with_mmr(
                semantic_kept
            )
        )

        raw_metrics = (
            v3.calculate_retrieval_metrics(
                raw_candidates,
                support_indexes,
            )
        )

        filtered_metrics = (
            v3.calculate_retrieval_metrics(
                filtered_evidence,
                support_indexes,
            )
        )

        support_lost_after_filtering = bool(
            raw_metrics[
                "complete_support"
            ]
            and not filtered_metrics[
                "complete_support"
            ]
        )

        generated_answer: str | None = None

        response_id = ""

        user_prompt = build_user_prompt(
            example,
            filtered_evidence,
        )

        if args.generate:
            if openai_client is None:
                raise RuntimeError(
                    "The OpenAI client "
                    "was not initialized."
                )

            (
                generated_answer,
                response_id,
            ) = generate_answer(
                client=openai_client,
                model_name=(
                    args.generation_model
                ),
                user_prompt=user_prompt,
            )

            answer_length = count_words(
                generated_answer
            )

            generated_record = {
                "system_version": (
                    "V3_filtered_top12_generation"
                ),
                "example_number": (
                    example_number
                ),
                "xrag_id": (
                    v3.get_first_value(
                        example,
                        "id",
                        "xrag_id",
                    )
                ),
                "language": (
                    v3.get_language(example)
                ),
                "task": (
                    v3.get_task(example)
                ),
                "question_type": (
                    v3.get_question_type(
                        example
                    )
                ),
                "answer_type": (
                    v3.get_answer_type(
                        example
                    )
                ),
                "question": question,
                "gold_answer": (
                    v3.get_gold_answer(
                        example
                    )
                ),
                "support_articles": [
                    index + 1
                    for index
                    in support_indexes
                ],
                "generation_model": (
                    args.generation_model
                ),
                "response_id": (
                    response_id
                ),
                "generated_answer": (
                    generated_answer
                ),
                "answer_word_count": (
                    answer_length
                ),
                "over_35_words": (
                    answer_length > 35
                ),
                "retrieval": {
                    "raw_metrics": (
                        raw_metrics
                    ),
                    "filtered_metrics": (
                        filtered_metrics
                    ),
                    "support_lost_after_filtering": (
                        support_lost_after_filtering
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
                },
                "filtered_evidence": [
                    sentence.to_dict()
                    for sentence
                    in filtered_evidence
                ],
                "prompt": {
                    "instructions": (
                        SYSTEM_INSTRUCTIONS
                    ),
                    "input": (
                        user_prompt
                    ),
                },
            }

            generated_records.append(
                generated_record
            )

            # Save after every answer so progress is not lost.
            write_jsonl(
                output_jsonl_path,
                generated_records,
            )

            write_review_csv(
                review_csv_path,
                generated_records,
            )

            summary_path.write_text(
                create_summary(
                    generated_records,
                    args.generation_model,
                ),
                encoding="utf-8",
            )

        if (
            example_number
            <= args.preview_count
        ):
            print_preview(
                example_number=example_number,
                example=example,
                filtered_evidence=(
                    filtered_evidence
                ),
                generated_answer=(
                    generated_answer
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
            f"generated="
            f"{generated_answer is not None}"
        )

    if args.generate:
        generated_records = sorted(
            generated_records,
            key=lambda record: (
                record["example_number"]
            ),
        )

        write_jsonl(
            output_jsonl_path,
            generated_records,
        )

        write_review_csv(
            review_csv_path,
            generated_records,
        )

        summary_text = create_summary(
            generated_records,
            args.generation_model,
        )

        summary_path.write_text(
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
            output_jsonl_path
        )

        print(
            review_csv_path
        )

        print(
            summary_path
        )

    else:
        print(
            "\nRetrieval preview completed "
            "without API calls."
        )

        print(
            "Add --generate when you are "
            "ready to generate answers."
        )


if __name__ == "__main__":
    main()