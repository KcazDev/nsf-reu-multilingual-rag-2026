import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLE = PROJECT_ROOT / "data" / "xrag_eval_sample_50.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "rag_pipeline_v1_outputs.jsonl"
MODEL_NAME = "gpt-5.4-2026-03-05"
MAX_OUTPUT_TOKENS = 180
PROMPT_VERSION = "rag_pipeline_v1_controlled"
PAUSE_BETWEEN_CALLS_SECONDS = 1

SYSTEM_PROMPT = (
    "You are completing a multilingual factual question-answering task using retrieved news documents. "
    "Return exactly one sentence in the same language as the question. Use no more than 35 words. "
    "Do not use headings, bullet points, caveats, refusals, or explanations. Use the retrieved documents "
    "that directly answer the question. Ignore distractors and later events that do not answer the original question."
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Sample file was not found:\n{path}")
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
    return examples


def get_article_text(article: Any) -> str:
    if isinstance(article, str):
        return article
    if isinstance(article, dict):
        parts = []
        for key in ["title", "text", "content", "article", "body"]:
            value = article.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
        return " ".join(parts)
    return str(article)


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def rank_with_bm25(question: str, articles: list[Any]) -> list[tuple[int, float]]:
    tokenized_articles = [tokenize(get_article_text(article)) for article in articles]
    tokenized_question = tokenize(question)
    if not tokenized_question:
        raise ValueError("Question produced no BM25 tokens.")
    bm25 = BM25Okapi(tokenized_articles)
    scores = bm25.get_scores(tokenized_question)
    return sorted([(i, float(s)) for i, s in enumerate(scores)], key=lambda item: item[1], reverse=True)


def get_support_indexes(example: dict[str, Any]) -> list[int]:
    labels = example.get("is_support", [])
    article_order = example.get("article_order") or list(range(len(labels)))
    support_indexes = []
    for article_index in article_order:
        try:
            if float(labels[article_index]) > 0:
                support_indexes.append(int(article_index))
        except (IndexError, TypeError, ValueError):
            continue
    return support_indexes


def retrieval_metrics(selected: list[int], support: list[int]) -> dict[str, Any]:
    selected_set = set(selected)
    support_set = set(support)
    found = selected_set.intersection(support_set)
    return {
        "retrieval_precision": len(found) / len(selected) if selected else 0.0,
        "retrieval_recall": len(found) / len(support_set) if support_set else 0.0,
        "complete_support_retrieved": support_set.issubset(selected_set) if support_set else False,
        "support_found": len(found),
    }


def format_documents(example: dict[str, Any], selected_indexes: list[int]) -> str:
    sections = []
    for document_number, article_index in enumerate(selected_indexes, start=1):
        article_text = get_article_text(example["articles"][article_index])
        date = example.get("dates", [""] * len(example["articles"]))[article_index] if "dates" in example else ""
        url = example.get("urls", [""] * len(example["articles"]))[article_index] if "urls" in example else ""
        sections.append(
            f"DOCUMENT {document_number}\n"
            f"Original article number: {article_index + 1}\n"
            f"Publication date: {date}\n"
            f"Source URL: {url}\n"
            f"Article text:\n{article_text}"
        )
    return "\n\n".join(sections)


def build_user_prompt(question: str, documents_text: str) -> str:
    return f"RETRIEVED DOCUMENTS:\n\n{documents_text}\n\nQUESTION:\n{question}"


def get_response_text(response: Any) -> str:
    return (response.output_text or "").strip()


def get_stop_reason(response: Any) -> str:
    if response.status == "completed":
        return "end_turn"
    if response.status == "incomplete":
        details = getattr(response, "incomplete_details", None)
        return getattr(details, "reason", None) or "incomplete"
    return response.status or "unknown"


def save_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_completed(path: Path) -> set[tuple[int, int, str]]:
    completed = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            completed.add((int(record["example_number"]), int(record["top_k"]), str(record["mode"])))
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multilingual RAG pipeline v1.")
    parser.add_argument("--sample-file", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-examples", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--generate", action="store_true", help="Call the OpenAI API. Without this flag, only preview retrieval and prompts.")
    args = parser.parse_args()

    mode = "generate" if args.generate else "preview"
    examples = load_jsonl(args.sample_file)[: args.max_examples]
    completed = load_completed(args.output_file)
    client = None
    if args.generate:
        load_dotenv(PROJECT_ROOT / ".env")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY was not found in .env.")
        client = OpenAI(api_key=api_key)

    print("MULTILINGUAL RAG PIPELINE V1")
    print("=" * 70)
    print(f"Sample file: {args.sample_file}")
    print(f"Output file: {args.output_file}")
    print(f"Mode: {mode}")
    print(f"Top-K: {args.top_k}")
    print(f"Examples: {len(examples)}")

    for example_number, example in enumerate(examples, start=1):
        if (example_number, args.top_k, mode) in completed:
            print(f"Skipping completed example {example_number}")
            continue
        question = example["question"]
        ranked = rank_with_bm25(question, example["articles"])
        selected = [i for i, _ in ranked[: args.top_k]]
        ranked_indexes = [i for i, _ in ranked]
        ranked_scores = [s for _, s in ranked]
        support = get_support_indexes(example)
        metrics = retrieval_metrics(selected, support)
        documents_text = format_documents(example, selected)
        user_prompt = build_user_prompt(question, documents_text)

        print("\n" + "#" * 70)
        print(f"Example {example_number}")
        print(f"XRAG ID: {example.get('id')}")
        print(f"Language: {example.get('language')}")
        print(f"Task: {example.get('task')}")
        print(f"Question: {question}")
        print(f"BM25 ranking: {[i + 1 for i in ranked_indexes]}")
        print(f"Selected Top-{args.top_k}: {[i + 1 for i in selected]}")
        print(f"True support: {[i + 1 for i in support]}")
        print(f"Retrieval precision/recall/complete: {metrics['retrieval_precision']:.3f} / {metrics['retrieval_recall']:.3f} / {metrics['complete_support_retrieved']}")

        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "example_number": example_number,
            "xrag_id": example.get("id"),
            "language": example.get("language"),
            "task": example.get("task"),
            "question_type": example.get("q_type"),
            "answer_type": example.get("a_type"),
            "crossdoc_type": example.get("crossdoc_type"),
            "question": question,
            "gold_answer": example.get("answer"),
            "retriever": "BM25Okapi",
            "top_k": args.top_k,
            "selected_article_numbers_one_based": [i + 1 for i in selected],
            "support_article_numbers_one_based": [i + 1 for i in support],
            "bm25_full_ranking_one_based": [i + 1 for i in ranked_indexes],
            "bm25_scores_by_rank": [round(s, 6) for s in ranked_scores],
            **metrics,
            "prompt_version": PROMPT_VERSION,
        }

        if args.generate:
            assert client is not None
            response = client.responses.create(
                model=MODEL_NAME,
                instructions=SYSTEM_PROMPT,
                input=user_prompt,
                reasoning={"effort": "none"},
                temperature=0,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                store=False,
            )
            answer = get_response_text(response)
            record.update({
                "model": MODEL_NAME,
                "model_answer": answer,
                "word_count": len(answer.split()),
                "stop_reason": get_stop_reason(response),
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            })
            print("\nMODEL ANSWER:")
            print(answer)
            time.sleep(PAUSE_BETWEEN_CALLS_SECONDS)
        else:
            record.update({
                "model": None,
                "model_answer": None,
                "word_count": None,
                "stop_reason": None,
                "input_tokens": None,
                "output_tokens": None,
                "prompt_preview_first_1500_chars": user_prompt[:1500],
            })
            print("\nPROMPT PREVIEW FIRST 1500 CHARS:")
            print(user_prompt[:1500])

        save_jsonl(args.output_file, record)

    print("\n" + "=" * 70)
    print("PIPELINE V1 FINISHED")
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
