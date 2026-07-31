import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLE = PROJECT_ROOT / "data" / "xrag_eval_sample_50.jsonl"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "outputs" / "bm25_topk_eval_50.csv"
DEFAULT_OUTPUT_TXT = PROJECT_ROOT / "outputs" / "bm25_topk_eval_50_summary.txt"


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


def get_support_indexes(example: dict[str, Any]) -> list[int]:
    support_labels = example.get("is_support", [])
    article_order = example.get("article_order") or list(range(len(support_labels)))
    support_indexes = []
    for article_index in article_order:
        try:
            if float(support_labels[article_index]) > 0:
                support_indexes.append(int(article_index))
        except (IndexError, TypeError, ValueError):
            continue
    return support_indexes


def rank_with_bm25(question: str, articles: list[Any]) -> list[tuple[int, float]]:
    tokenized_articles = [tokenize(get_article_text(article)) for article in articles]
    tokenized_question = tokenize(question)
    if not tokenized_question:
        raise ValueError("Question produced no tokens.")
    bm25 = BM25Okapi(tokenized_articles)
    scores = bm25.get_scores(tokenized_question)
    return sorted([(i, float(s)) for i, s in enumerate(scores)], key=lambda item: item[1], reverse=True)


def metrics_at_k(ranked_indexes: list[int], support_indexes: list[int], k: int) -> dict[str, Any]:
    selected = ranked_indexes[:k]
    support_set = set(support_indexes)
    found = set(selected).intersection(support_set)
    precision = len(found) / k if k else 0.0
    recall = len(found) / len(support_set) if support_set else 0.0
    complete = support_set.issubset(set(selected)) if support_set else False
    return {
        f"precision_at_{k}": precision,
        f"recall_at_{k}": recall,
        f"complete_support_at_{k}": complete,
        f"support_found_at_{k}": len(found),
    }


def reciprocal_rank(ranked_indexes: list[int], support_indexes: list[int]) -> float:
    support_set = set(support_indexes)
    for rank, article_index in enumerate(ranked_indexes, start=1):
        if article_index in support_set:
            return 1.0 / rank
    return 0.0


def summarize(rows: list[dict[str, Any]], top_ks: list[int]) -> str:
    lines = ["BM25 TOP-K RETRIEVAL SUMMARY", "=" * 70, f"Examples: {len(rows)}", "", "Overall metrics:"]
    for k in top_ks:
        avg_p = sum(float(row[f"precision_at_{k}"]) for row in rows) / len(rows)
        avg_r = sum(float(row[f"recall_at_{k}"]) for row in rows) / len(rows)
        comp = sum(str(row[f"complete_support_at_{k}"]).lower() == "true" for row in rows)
        lines.append(f"Top-{k}: Precision={avg_p:.3f}, Recall={avg_r:.3f}, Complete support={comp}/{len(rows)}")
    avg_mrr = sum(float(row["mrr"]) for row in rows) / len(rows)
    lines.append(f"MRR: {avg_mrr:.3f}")
    for group_name, field in [("By language", "language"), ("By task", "task"), ("By question type", "question_type")]:
        lines.extend(["", group_name + ":", "-" * 70])
        grouped = defaultdict(list)
        for row in rows:
            grouped[row[field]].append(row)
        for value, group_rows in sorted(grouped.items()):
            lines.append(f"{value} ({len(group_rows)} examples)")
            for k in top_ks:
                avg_r = sum(float(row[f"recall_at_{k}"]) for row in group_rows) / len(group_rows)
                comp = sum(str(row[f"complete_support_at_{k}"]).lower() == "true" for row in group_rows)
                lines.append(f"  Top-{k}: Recall={avg_r:.3f}, Complete={comp}/{len(group_rows)}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BM25 retrieval at multiple Top-K settings.")
    parser.add_argument("--sample-file", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_OUTPUT_TXT)
    parser.add_argument("--top-ks", type=int, nargs="+", default=[1, 2, 3, 4])
    args = parser.parse_args()
    examples = load_jsonl(args.sample_file)
    rows = []
    for example_number, example in enumerate(examples, start=1):
        ranked = rank_with_bm25(example["question"], example["articles"])
        ranked_indexes = [i for i, _ in ranked]
        ranked_scores = [s for _, s in ranked]
        support_indexes = get_support_indexes(example)
        row = {
            "example_number": example_number,
            "xrag_id": example.get("id", ""),
            "language": example.get("language", ""),
            "task": example.get("task", ""),
            "question_type": example.get("q_type", ""),
            "answer_type": example.get("a_type", ""),
            "crossdoc_type": example.get("crossdoc_type", ""),
            "question": example["question"],
            "gold_answer": example.get("answer", ""),
            "support_articles_one_based": ",".join(str(i + 1) for i in support_indexes),
            "bm25_ranking_one_based": ",".join(str(i + 1) for i in ranked_indexes),
            "bm25_scores_by_rank": ",".join(f"{score:.6f}" for score in ranked_scores),
            "mrr": reciprocal_rank(ranked_indexes, support_indexes),
        }
        for k in args.top_ks:
            row.update(metrics_at_k(ranked_indexes, support_indexes, k))
        rows.append(row)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary_text = summarize(rows, args.top_ks)
    args.output_summary.write_text(summary_text, encoding="utf-8")
    print(summary_text)
    print("\nFiles created:")
    print(args.output_csv)
    print(args.output_summary)


if __name__ == "__main__":
    main()
