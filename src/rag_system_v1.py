import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
ENV_FILE = ROOT_DIR / ".env"

DEFAULT_SAMPLE_FILE = DATA_DIR / "xrag_eval_sample_50.jsonl"
DEFAULT_OUTPUT_FILE = OUTPUTS_DIR / "rag_system_v1_outputs.jsonl"

DEFAULT_MODEL = "gpt-5.4-2026-03-05"
PROMPT_VERSION = "rag_system_v1_controlled"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_number}: {e}") from e

    return records


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def tokenize(text: str) -> List[str]:
    if not text:
        return []

    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def get_question(record: Dict[str, Any]) -> str:
    return (
        record.get("question")
        or record.get("query")
        or record.get("input")
        or ""
    )


def get_gold_answer(record: Dict[str, Any]) -> Any:
    return (
        record.get("gold_answer")
        or record.get("answer")
        or record.get("answers")
        or ""
    )


def get_xrag_id(record: Dict[str, Any]) -> Any:
    return (
        record.get("xrag_id")
        or record.get("id")
        or record.get("_id")
        or record.get("example_id")
    )


def get_language(record: Dict[str, Any]) -> str:
    return (
        record.get("language")
        or record.get("lang")
        or record.get("question_language")
        or ""
    )


def text_from_article_candidate(article: Any) -> str:
    """
    Used while deciding which record field actually contains the article texts.
    """
    if isinstance(article, str):
        return article

    if not isinstance(article, dict):
        return str(article)

    return (
        article.get("article_text")
        or article.get("text")
        or article.get("content")
        or article.get("body")
        or article.get("passage")
        or article.get("document")
        or article.get("article")
        or ""
    )


def score_article_list_candidate(value: Any) -> int:
    """
    Scores a possible article/document list.

    This prevents the system from accidentally choosing article_order when
    article_order is only a list of IDs or short labels.
    """
    if not isinstance(value, list) or not value:
        return -1

    score = 0

    for item in value[:8]:
        text = text_from_article_candidate(item)

        if len(text) > 200:
            score += 3
        elif len(text) > 50:
            score += 2
        elif len(text) > 0:
            score += 1

        if isinstance(item, dict):
            useful_keys = {
                "article_text",
                "text",
                "content",
                "body",
                "source_url",
                "url",
                "publication_date",
                "date",
                "is_support",
                "support",
                "relevance",
            }

            score += len(useful_keys.intersection(set(item.keys())))

    return score


def get_articles(record: Dict[str, Any]) -> List[Any]:
    """
    Finds the article/document list inside an XRAG-style record.

    Supports:
    - list of article dictionaries
    - list of plain article strings
    - several possible field names
    """

    possible_keys = [
        "articles",
        "documents",
        "docs",
        "contexts",
        "ctxs",
        "passages",
        "ordered_articles",
        "article_order",
    ]

    best_key = None
    best_value = None
    best_score = -1

    for key in possible_keys:
        value = record.get(key)
        score = score_article_list_candidate(value)

        if score > best_score:
            best_key = key
            best_value = value
            best_score = score

    if isinstance(best_value, list) and best_value:
        return best_value

    raise ValueError(
        "Could not find article/document list in record. "
        "Expected one of: articles, documents, docs, contexts, ctxs, passages, ordered_articles, article_order."
    )


def get_article_text(article: Any) -> str:
    """
    Gets article text whether the article is stored as:
    - a plain string
    - a dictionary with text fields
    - another object type
    """
    if isinstance(article, str):
        return article

    if not isinstance(article, dict):
        return str(article)

    return (
        article.get("article_text")
        or article.get("text")
        or article.get("content")
        or article.get("body")
        or article.get("passage")
        or article.get("document")
        or article.get("article")
        or ""
    )


def get_article_date(article: Any) -> str:
    if not isinstance(article, dict):
        return ""

    return (
        str(article.get("publication_date"))
        if article.get("publication_date") is not None
        else str(article.get("date") or article.get("published_date") or "")
    )


def get_article_url(article: Any) -> str:
    if not isinstance(article, dict):
        return ""

    return (
        article.get("source_url")
        or article.get("url")
        or article.get("link")
        or ""
    )


def is_support_article(article: Any) -> bool:
    """
    XRAG uses support/relevance style labels.
    This handles several possible label names.
    """
    if not isinstance(article, dict):
        return False

    possible_keys = [
        "is_support",
        "support",
        "is_gold",
        "gold",
        "relevant",
        "relevance",
        "label",
    ]

    for key in possible_keys:
        if key not in article:
            continue

        value = article.get(key)

        if isinstance(value, bool):
            return value

        if isinstance(value, int) or isinstance(value, float):
            return value > 0

        if isinstance(value, str):
            lowered = value.strip().lower()
            return lowered in {"1", "true", "yes", "support", "relevant", "gold"}

    return False


def normalize_support_numbers(value: Any) -> List[int]:
    """
    Converts possible support-number formats into article numbers.
    """
    if value is None:
        return []

    if isinstance(value, int):
        return [value]

    if isinstance(value, str):
        pieces = re.findall(r"\d+", value)
        return [int(piece) for piece in pieces]

    if isinstance(value, list):
        numbers = []

        for item in value:
            if isinstance(item, int):
                numbers.append(item)

            elif isinstance(item, str):
                found = re.findall(r"\d+", item)
                numbers.extend(int(x) for x in found)

            elif isinstance(item, dict):
                for possible_key in [
                    "article_number",
                    "article_index",
                    "index",
                    "id",
                    "doc_id",
                    "document_id",
                ]:
                    if possible_key in item:
                        try:
                            numbers.append(int(item[possible_key]))
                        except Exception:
                            pass

        return numbers

    return []


def is_positive_support_value(value: Any) -> bool:
    """
    Returns True when a value marks an article as support/relevant.
    XRAG stores support as numbers like 0 or 2.
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, int) or isinstance(value, float):
        return value > 0

    if isinstance(value, str):
        lowered = value.strip().lower()
        return lowered in {"1", "2", "true", "yes", "support", "relevant", "gold"}

    return False


def support_flags_to_article_numbers(flags: Any, articles: List[Any]) -> List[int]:
    """
    Converts a list like [0, 2, 0, 0, 0, 0, 0, 0]
    into one-based support article numbers like [2].
    """
    if not isinstance(flags, list):
        return []

    if len(flags) != len(articles):
        return []

    support_numbers = []

    for index, value in enumerate(flags, start=1):
        if is_positive_support_value(value):
            support_numbers.append(index)

    return support_numbers


def get_support_article_numbers_from_record(
    record: Dict[str, Any],
    articles: List[Any],
) -> List[int]:
    """
    Tries record-level support fields.
    XRAG sample stores support labels as:
        is_support: [0, 2, 0, 0, 0, 0, 0, 0]
    """

    if "is_support" in record:
        numbers = support_flags_to_article_numbers(record.get("is_support"), articles)

        if numbers:
            return numbers

    record_level_keys = [
        "support_article_numbers_one_based",
        "support_articles_one_based",
        "support_article_numbers",
        "support_articles",
        "support_idxs",
        "support_indices",
        "support_ids",
        "gold_articles",
        "gold_article_numbers",
        "relevant_articles",
        "evidence_articles",
        "supporting_articles",
    ]

    for key in record_level_keys:
        if key in record:
            numbers = normalize_support_numbers(record.get(key))

            if numbers:
                if min(numbers) == 0:
                    numbers = [number + 1 for number in numbers]

                return sorted(set(numbers))

    return []

def get_support_article_numbers_from_articles(articles: List[Any]) -> List[int]:
    """
    Tries article-level labels like is_support > 0.
    """
    support_numbers = []

    for index, article in enumerate(articles, start=1):
        if is_support_article(article):
            support_numbers.append(index)

    return support_numbers


def get_support_article_numbers_one_based(
    record: Dict[str, Any],
    articles: List[Any],
) -> List[int]:
    """
    Finds true support article numbers.

    First checks the record-level XRAG is_support list.
    Then checks article-level labels.
    """

    record_level_support = get_support_article_numbers_from_record(
        record=record,
        articles=articles,
    )

    if record_level_support:
        return record_level_support

    article_level_support = get_support_article_numbers_from_articles(articles)

    if article_level_support:
        return article_level_support

    return []

def rank_articles_bm25(
    question: str,
    articles: List[Any],
) -> Tuple[List[int], List[float]]:
    """
    Returns:
        ranking_one_based: article numbers ranked best to worst
        scores_by_rank: BM25 scores in ranked order
    """
    article_texts = [get_article_text(article) for article in articles]
    tokenized_articles = [tokenize(text) for text in article_texts]
    tokenized_question = tokenize(question)

    bm25 = BM25Okapi(tokenized_articles)
    scores = bm25.get_scores(tokenized_question)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    ranking_one_based = [i + 1 for i in ranked_indices]
    scores_by_rank = [round(float(scores[i]), 6) for i in ranked_indices]

    return ranking_one_based, scores_by_rank


def calculate_retrieval_metrics(
    selected_article_numbers: List[int],
    support_article_numbers: List[int],
) -> Dict[str, Any]:
    selected_set = set(selected_article_numbers)
    support_set = set(support_article_numbers)

    found_support = selected_set.intersection(support_set)

    precision = len(found_support) / len(selected_set) if selected_set else 0.0
    recall = len(found_support) / len(support_set) if support_set else 0.0
    complete = support_set.issubset(selected_set) if support_set else False

    return {
        "retrieval_precision": precision,
        "retrieval_recall": recall,
        "complete_support_retrieved": complete,
        "support_found": len(found_support),
    }


def format_retrieved_documents(
    articles: List[Any],
    selected_article_numbers: List[int],
) -> str:
    blocks = []

    for display_number, original_article_number in enumerate(selected_article_numbers, start=1):
        article = articles[original_article_number - 1]

        text = get_article_text(article)
        date = get_article_date(article)
        url = get_article_url(article)

        block = (
            f"DOCUMENT {display_number}\n"
            f"Original article number: {original_article_number}\n"
            f"Publication date: {date}\n"
            f"Source URL: {url}\n"
            f"Article text:\n{text}"
        )

        blocks.append(block)

    separator = "\n\n" + ("-" * 70) + "\n\n"
    return separator.join(blocks)


def build_prompt(
    question: str,
    retrieved_documents_text: str,
) -> str:
    return f"""You are a multilingual retrieval-augmented question answering system.

Use the retrieved documents to answer the question.

Rules:
1. Answer in the same language as the question.
2. Use only information supported by the retrieved documents.
3. Ignore irrelevant or distracting documents.
4. Give one direct sentence.
5. Keep the answer under 35 words.
6. Do not use bullet points, headings, citations, or explanations.
7. If the retrieved documents do not contain enough information, say that the available documents do not provide enough information, in the same language as the question.

RETRIEVED DOCUMENTS:
{retrieved_documents_text}

QUESTION:
{question}

ANSWER:"""


def count_words(text: Optional[str]) -> Optional[int]:
    if text is None:
        return None

    cleaned = text.strip()

    if not cleaned:
        return 0

    return len(cleaned.split())


def generate_answer(
    prompt: str,
    model: str,
    max_output_tokens: int,
) -> Dict[str, Any]:
    client = OpenAI()

    response = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=max_output_tokens,
    )

    answer = response.output_text.strip()
    usage = getattr(response, "usage", None)

    input_tokens = getattr(usage, "input_tokens", None) if usage else None
    output_tokens = getattr(usage, "output_tokens", None) if usage else None

    return {
        "model_answer": answer,
        "word_count": count_words(answer),
        "stop_reason": getattr(response, "status", None),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def load_completed_keys(output_file: Path) -> set:
    """
    Used so rerunning the script does not duplicate finished records.
    """
    completed = set()

    if not output_file.exists():
        return completed

    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            key = (
                record.get("mode"),
                record.get("example_number"),
                record.get("top_k"),
                record.get("model"),
            )

            completed.add(key)

    return completed


def process_example(
    record: Dict[str, Any],
    example_number: int,
    top_k: int,
    generate: bool,
    model: str,
    max_output_tokens: int,
) -> Dict[str, Any]:
    question = get_question(record)
    gold_answer = get_gold_answer(record)
    articles = get_articles(record)

    bm25_ranking, bm25_scores_by_rank = rank_articles_bm25(question, articles)

    selected_article_numbers = bm25_ranking[:top_k]

    support_article_numbers = get_support_article_numbers_one_based(
        record=record,
        articles=articles,
    )

    retrieval_metrics = calculate_retrieval_metrics(
        selected_article_numbers=selected_article_numbers,
        support_article_numbers=support_article_numbers,
    )

    retrieved_documents_text = format_retrieved_documents(
        articles=articles,
        selected_article_numbers=selected_article_numbers,
    )

    prompt = build_prompt(
        question=question,
        retrieved_documents_text=retrieved_documents_text,
    )

    output_record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "system_name": "Multilingual_RAG_System",
        "system_version": "v1",
        "mode": "generate" if generate else "preview",
        "example_number": example_number,
        "xrag_id": get_xrag_id(record),
        "language": get_language(record),
        "task": record.get("task"),
        "question_type": record.get("question_type") or record.get("q_type"),
        "answer_type": record.get("answer_type") or record.get("a_type"),
        "crossdoc_type": record.get("crossdoc_type"),
        "question": question,
        "gold_answer": gold_answer,
        "retriever": "BM25Okapi",
        "top_k": top_k,
        "selected_article_numbers_one_based": selected_article_numbers,
        "support_article_numbers_one_based": support_article_numbers,
        "bm25_full_ranking_one_based": bm25_ranking,
        "bm25_scores_by_rank": bm25_scores_by_rank,
        **retrieval_metrics,
        "prompt_version": PROMPT_VERSION,
        "model": model if generate else None,
        "model_answer": None,
        "word_count": None,
        "stop_reason": None,
        "input_tokens": None,
        "output_tokens": None,
        "prompt_preview_first_1500_chars": prompt[:1500],
    }

    if generate:
        generation_result = generate_answer(
            prompt=prompt,
            model=model,
            max_output_tokens=max_output_tokens,
        )

        output_record.update(generation_result)

    return output_record


def print_example_output(record: Dict[str, Any]) -> None:
    print()
    print("#" * 70)
    print(f"Example {record['example_number']}")
    print(f"XRAG ID: {record['xrag_id']}")
    print(f"Language: {record['language']}")
    print(f"Task: {record.get('task')}")
    print(f"Question type: {record.get('question_type')}")
    print(f"Question: {record['question']}")
    print(f"BM25 ranking: {record['bm25_full_ranking_one_based']}")
    print(f"Selected Top-{record['top_k']}: {record['selected_article_numbers_one_based']}")
    print(f"True support: {record['support_article_numbers_one_based']}")
    print(
        "Retrieval precision/recall/complete: "
        f"{record['retrieval_precision']:.3f} / "
        f"{record['retrieval_recall']:.3f} / "
        f"{record['complete_support_retrieved']}"
    )

    if record["mode"] == "generate":
        print()
        print("MODEL ANSWER:")
        print(record["model_answer"])
        print(f"Word count: {record['word_count']}")
    else:
        print()
        print("PROMPT PREVIEW FIRST 1500 CHARS:")
        print(record["prompt_preview_first_1500_chars"])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample-file",
        default=str(DEFAULT_SAMPLE_FILE),
        help="Path to XRAG sample JSONL file.",
    )

    parser.add_argument(
        "--output-file",
        default=str(DEFAULT_OUTPUT_FILE),
        help="Path to save JSONL outputs.",
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=3,
        help="Number of examples to process.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of BM25 documents to retrieve.",
    )

    parser.add_argument(
        "--generate",
        action="store_true",
        help="Actually call the OpenAI model. Without this flag, the script only previews prompts.",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenAI model name.",
    )

    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=180,
        help="Maximum output tokens for generation.",
    )

    args = parser.parse_args()

    load_dotenv(ENV_FILE)

    sample_file = Path(args.sample_file)
    output_file = Path(args.output_file)
    generate = args.generate

    if generate and not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY not found. Make sure your .env file exists and contains OPENAI_API_KEY."
        )

    records = load_jsonl(sample_file)
    selected_records = records[: args.max_examples]

    completed_keys = load_completed_keys(output_file)

    print("MULTILINGUAL RAG SYSTEM V1")
    print("=" * 70)
    print(f"Root directory: {ROOT_DIR}")
    print(f"Sample file: {sample_file}")
    print(f"Output file: {output_file}")
    print(f"Mode: {'generate' if generate else 'preview'}")
    print(f"Top-K: {args.top_k}")
    print(f"Examples requested: {args.max_examples}")
    print(f"Examples available: {len(records)}")

    processed_count = 0
    skipped_count = 0

    for example_number, record in enumerate(selected_records, start=1):
        model_for_key = args.model if generate else None

        key = (
            "generate" if generate else "preview",
            example_number,
            args.top_k,
            model_for_key,
        )

        if key in completed_keys:
            print(f"Skipping completed example {example_number}")
            skipped_count += 1
            continue

        output_record = process_example(
            record=record,
            example_number=example_number,
            top_k=args.top_k,
            generate=generate,
            model=args.model,
            max_output_tokens=args.max_output_tokens,
        )

        append_jsonl(output_file, output_record)
        print_example_output(output_record)

        processed_count += 1

    print()
    print("=" * 70)
    print("RAG SYSTEM V1 FINISHED")
    print(f"Processed: {processed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()