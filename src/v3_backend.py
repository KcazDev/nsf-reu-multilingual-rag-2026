from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------
# Import the existing tested V3 scripts
# ---------------------------------------------------------------------
SRC_DIRECTORY = Path(__file__).resolve().parent
PROJECT_DIRECTORY = SRC_DIRECTORY.parent

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

import rag_system_v3_bge_m3_filtered as v3
import rag_system_v3_generate as generation


# ---------------------------------------------------------------------
# Fixed final V3 configuration
# ---------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
GENERATION_MODEL_NAME = "gpt-5.4"

RAW_TOP_K = 20
FINAL_TOP_K = 12
SEMANTIC_DUPLICATE_THRESHOLD = 0.92
MAX_SENTENCES_PER_ARTICLE = 4
MMR_LAMBDA = 0.85


def validate_v3_configuration() -> None:
    """
    Confirm that the frontend is using the same V3 settings that were
    evaluated on the 50-question XRAG sample.
    """

    expected_settings = {
        "RAW_TOP_K": RAW_TOP_K,
        "FINAL_TOP_K": FINAL_TOP_K,
        "SEMANTIC_DUPLICATE_THRESHOLD": SEMANTIC_DUPLICATE_THRESHOLD,
        "MAX_SENTENCES_PER_ARTICLE": MAX_SENTENCES_PER_ARTICLE,
        "MMR_LAMBDA": MMR_LAMBDA,
    }

    mismatches: list[str] = []

    for setting_name, expected_value in expected_settings.items():
        actual_value = getattr(v3, setting_name, None)

        if actual_value != expected_value:
            mismatches.append(
                f"{setting_name}: expected {expected_value}, "
                f"found {actual_value}"
            )

    if mismatches:
        mismatch_text = "\n".join(mismatches)

        raise RuntimeError(
            "The frontend V3 settings do not match the evaluated system:\n"
            f"{mismatch_text}"
        )


# ---------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------
def detect_language_code(question: str) -> str:
    """
    Provide a basic language code for sentence splitting.

    Arabic script is detected directly. Questions containing common
    Spanish punctuation or accented characters are marked Spanish.
    Other questions default to English.
    """

    if re.search(r"[\u0600-\u06FF]", question):
        return "ar"

    if re.search(r"[¿¡áéíóúüñÁÉÍÓÚÜÑ]", question):
        return "es"

    return "en"


def split_pasted_articles(text: str) -> list[str]:
    """
    Split pasted text into candidate articles.

    Preferred separator:
        ---

    When no divider is present, blank lines are treated as article
    boundaries.
    """

    cleaned_text = text.strip()

    if not cleaned_text:
        return []

    divider_parts = re.split(
        r"\n\s*---+\s*\n",
        cleaned_text,
    )

    if len(divider_parts) > 1:
        return [
            part.strip()
            for part in divider_parts
            if part.strip()
        ]

    return [
        part.strip()
        for part in re.split(
            r"\n\s*\n",
            cleaned_text,
        )
        if part.strip()
    ]


def extract_articles_from_uploaded_file(
    filename: str,
    file_bytes: bytes,
) -> list[str]:
    """
    Extract article text from TXT, JSON, or JSONL uploads.
    """

    decoded_text = file_bytes.decode(
        "utf-8-sig",
        errors="replace",
    ).strip()

    if not decoded_text:
        return []

    extension = Path(filename).suffix.lower()

    if extension == ".txt":
        articles = split_pasted_articles(decoded_text)

        if articles:
            return articles

        return [decoded_text]

    if extension == ".jsonl":
        articles: list[str] = []

        for line in decoded_text.splitlines():
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)
            articles.extend(
                extract_article_texts_from_json(record)
            )

        return articles

    if extension == ".json":
        data = json.loads(decoded_text)
        return extract_article_texts_from_json(data)

    raise ValueError(
        f"Unsupported file type: {extension}"
    )


def extract_article_texts_from_json(
    data: Any,
) -> list[str]:
    """
    Extract article strings from several simple JSON structures.
    """

    if isinstance(data, str):
        text = data.strip()
        return [text] if text else []

    if isinstance(data, list):
        articles: list[str] = []

        for item in data:
            articles.extend(
                extract_article_texts_from_json(item)
            )

        return articles

    if isinstance(data, dict):
        for key in (
            "text",
            "content",
            "article",
            "article_text",
            "body",
        ):
            value = data.get(key)

            if isinstance(value, str) and value.strip():
                return [value.strip()]

        for key in (
            "articles",
            "documents",
            "contexts",
            "evidence",
        ):
            value = data.get(key)

            if value is not None:
                return extract_article_texts_from_json(value)

    return []


# ---------------------------------------------------------------------
# Sentence-unit construction
# ---------------------------------------------------------------------
def build_frontend_sentence_units(
    articles: list[str],
    language: str,
) -> list[v3.SentenceUnit]:
    """
    Convert uploaded or pasted articles into the same SentenceUnit
    objects used by the evaluated V3 retriever.
    """

    sentence_units: list[v3.SentenceUnit] = []

    for article_index, article_text in enumerate(articles):
        cleaned_article = article_text.strip()

        if not cleaned_article:
            continue

        sentences = v3.split_sentences(
            cleaned_article,
            language,
        )

        for sentence_index, sentence_text in enumerate(sentences):
            cleaned_sentence = sentence_text.strip()

            if not cleaned_sentence:
                continue

            sentence_units.append(
                v3.SentenceUnit(
                    article_index=article_index,
                    sentence_index=sentence_index,
                    text=cleaned_sentence,
                    is_support_article=False,
                    publication_date="",
                    source_url="",
                    embedding=None,
                    relevance_score=0.0,
                    raw_rank=None,
                    mmr_score=None,
                    final_rank=None,
                )
            )

    return sentence_units


def build_frontend_user_prompt(
    question: str,
    language: str,
    filtered_evidence: list[v3.SentenceUnit],
) -> str:
    """
    Build the generation prompt using the final filtered evidence.
    """

    evidence_blocks: list[str] = []

    for unit in filtered_evidence:
        evidence_blocks.append(
            f"[Article {unit.article_index + 1}, "
            f"Sentence {unit.sentence_index + 1}]\n"
            f"{unit.text}"
        )

    evidence_text = "\n\n".join(evidence_blocks)

    return (
        f"Question language: {language}\n\n"
        f"Question:\n"
        f"{question}\n\n"
        f"Retrieved evidence:\n"
        f"{evidence_text}\n\n"
        f"Return only the final answer."
    )


# ---------------------------------------------------------------------
# Main backend service
# ---------------------------------------------------------------------
class V3Backend:
    """
    Reusable backend service for the Streamlit frontend.
    """

    def __init__(
        self,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        generation_model_name: str = GENERATION_MODEL_NAME,
    ) -> None:
        validate_v3_configuration()

        load_dotenv(
            PROJECT_DIRECTORY / ".env"
        )

        self.embedding_model_name = embedding_model_name
        self.generation_model_name = generation_model_name

        self.embedding_model = SentenceTransformer(
            embedding_model_name
        )

        self.openai_client = OpenAI()

    def retrieve(
        self,
        question: str,
        articles: list[str],
    ) -> dict[str, Any]:
        """
        Run the complete V3 retrieval and filtering pipeline.
        """

        cleaned_question = question.strip()
        cleaned_articles = [
            article.strip()
            for article in articles
            if article.strip()
        ]

        if not cleaned_question:
            raise ValueError(
                "The question cannot be empty."
            )

        if not cleaned_articles:
            raise ValueError(
                "At least one article is required."
            )

        language = detect_language_code(
            cleaned_question
        )

        sentence_units = build_frontend_sentence_units(
            cleaned_articles,
            language,
        )

        if not sentence_units:
            raise ValueError(
                "No usable sentences were found in the supplied articles."
            )

        question_embedding = (
            v3.embed_question_and_sentences(
                model=self.embedding_model,
                question=cleaned_question,
                sentence_units=sentence_units,
                batch_size=16,
                show_progress=False,
            )
        )

        raw_candidates = v3.retrieve_raw_top_20(
            question_embedding=question_embedding,
            sentence_units=sentence_units,
        )

        exact_filtered_candidates, exact_removed = (
            v3.remove_exact_duplicates(
                raw_candidates
            )
        )

        semantic_filtered_candidates, semantic_removed = (
            v3.remove_semantic_duplicates(
                exact_filtered_candidates
            )
        )

        final_evidence = (
            v3.select_final_evidence_with_mmr(
                semantic_filtered_candidates
            )
        )

        return {
            "question": cleaned_question,
            "language": language,
            "article_count": len(cleaned_articles),
            "sentence_count": len(sentence_units),
            "raw_candidate_count": len(raw_candidates),
            "exact_duplicates_removed": len(exact_removed),
            "semantic_duplicates_removed": len(semantic_removed),
            "final_evidence": final_evidence,
        }

    def answer(
        self,
        question: str,
        articles: list[str],
    ) -> dict[str, Any]:
        """
        Retrieve evidence and generate the final multilingual answer.
        """

        retrieval_result = self.retrieve(
            question=question,
            articles=articles,
        )

        final_evidence = retrieval_result[
            "final_evidence"
        ]

        if not final_evidence:
            raise RuntimeError(
                "The retriever did not return any final evidence."
            )

        user_prompt = build_frontend_user_prompt(
            question=retrieval_result["question"],
            language=retrieval_result["language"],
            filtered_evidence=final_evidence,
        )

        generated_answer, response_id = (
            generation.generate_answer(
                client=self.openai_client,
                model_name=self.generation_model_name,
                user_prompt=user_prompt,
            )
        )

        evidence_for_display = []

        for unit in final_evidence:
            evidence_for_display.append(
                {
                    "final_rank": unit.final_rank,
                    "raw_rank": unit.raw_rank,
                    "article_number": unit.article_index + 1,
                    "sentence_number": unit.sentence_index + 1,
                    "text": unit.text,
                    "relevance_score": float(
                        unit.relevance_score
                    ),
                    "mmr_score": (
                        float(unit.mmr_score)
                        if unit.mmr_score is not None
                        else None
                    ),
                }
            )

        return {
            "answer": generated_answer,
            "response_id": response_id,
            "embedding_model": self.embedding_model_name,
            "generation_model": self.generation_model_name,
            "language": retrieval_result["language"],
            "article_count": retrieval_result["article_count"],
            "sentence_count": retrieval_result["sentence_count"],
            "raw_candidate_count": retrieval_result[
                "raw_candidate_count"
            ],
            "exact_duplicates_removed": retrieval_result[
                "exact_duplicates_removed"
            ],
            "semantic_duplicates_removed": retrieval_result[
                "semantic_duplicates_removed"
            ],
            "final_evidence_count": len(
                evidence_for_display
            ),
            "evidence": evidence_for_display,
        }