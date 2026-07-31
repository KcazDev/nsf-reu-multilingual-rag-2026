import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = PROJECT_ROOT / "outputs" / "bm25_topk_eval_50.csv"
DEFAULT_OUTPUT_TXT = PROJECT_ROOT / "outputs" / "bm25_topk_failure_analysis_50.txt"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "outputs" / "bm25_topk_failure_cases_50.csv"


def load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Results file was not found:\n{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "yes", "1"}


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_list(value: Any) -> list[int]:
    text = str(value or "").strip()
    if not text:
        return []
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            pass
    return out


def support_ranks(row: dict[str, Any]) -> list[int]:
    support = set(parse_list(row.get("support_articles_one_based")))
    ranking = parse_list(row.get("bm25_ranking_one_based"))
    ranks = []
    for article in support:
        try:
            ranks.append(ranking.index(article) + 1)
        except ValueError:
            ranks.append(999)
    return sorted(ranks)


def available_top_ks(rows: list[dict[str, Any]]) -> list[int]:
    top_ks = []
    for key in rows[0].keys():
        if key.startswith("complete_support_at_"):
            try:
                top_ks.append(int(key.replace("complete_support_at_", "")))
            except ValueError:
                pass
    return sorted(top_ks)


def recommend_k(rows: list[dict[str, Any]], top_ks: list[int]) -> tuple[int, list[str]]:
    best_k = top_ks[0]
    best_score = -999.0
    lines = []
    for k in top_ks:
        complete = sum(truthy(r[f"complete_support_at_{k}"]) for r in rows)
        avg_recall = sum(to_float(r[f"recall_at_{k}"]) for r in rows) / len(rows)
        score = complete + avg_recall - (0.05 * k)
        lines.append(f"Top-{k}: complete={complete}/{len(rows)}, avg_recall={avg_recall:.3f}, selection_score={score:.3f}")
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze BM25 Top-K retrieval failures.")
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_OUTPUT_TXT)
    parser.add_argument("--output-failures", type=Path, default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args()
    rows = load_csv(args.results_file)
    top_ks = available_top_ks(rows)
    if not top_ks:
        raise ValueError("No complete_support_at_K columns were found.")
    recommended_k, rec_lines = recommend_k(rows, top_ks)
    lines = [
        "BM25 TOP-K FAILURE ANALYSIS",
        "=" * 70,
        f"Results file: {args.results_file}",
        f"Examples: {len(rows)}",
        "",
        "Top-K recommendation:",
        "-" * 70,
    ]
    lines.extend(rec_lines)
    lines.extend([
        "",
        f"Recommended starting Top-K for system v1: Top-{recommended_k}",
        "",
        "Interpretation:",
        "Use the smallest K that gives a strong complete-support gain. A larger K can retrieve more support, but it also sends more distractor evidence to the generator.",
        "",
    ])
    for k in top_ks:
        failed = [r for r in rows if not truthy(r[f"complete_support_at_{k}"])]
        lines.extend([f"Failure summary for Top-{k}", "-" * 70])
        lines.append(f"Missing complete support: {len(failed)}/{len(rows)}")
        lines.append(f"Failures by language: {dict(Counter(r['language'] for r in failed))}")
        lines.append(f"Failures by task: {dict(Counter(r['task'] for r in failed))}")
        lines.append(f"Failures by question type: {dict(Counter(r['question_type'] for r in failed))}")
        patterns = Counter(tuple(support_ranks(r)) for r in failed)
        lines.append("Support-rank patterns among failures:")
        for ranks, count in patterns.most_common(10):
            lines.append(f"  {list(ranks)}: {count}")
        lines.append("")
    failure_rows = []
    for r in rows:
        out = {
            "example_number": r.get("example_number"),
            "xrag_id": r.get("xrag_id"),
            "language": r.get("language"),
            "task": r.get("task"),
            "question_type": r.get("question_type"),
            "question": r.get("question"),
            "support_articles_one_based": r.get("support_articles_one_based"),
            "bm25_ranking_one_based": r.get("bm25_ranking_one_based"),
            "support_ranks": ",".join(str(x) for x in support_ranks(r)),
        }
        for k in top_ks:
            out[f"complete_support_at_{k}"] = r.get(f"complete_support_at_{k}")
            out[f"recall_at_{k}"] = r.get(f"recall_at_{k}")
        failure_rows.append(out)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_failures.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text("\n".join(lines), encoding="utf-8")
    with args.output_failures.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(failure_rows[0].keys()))
        writer.writeheader()
        writer.writerows(failure_rows)
    print("\n".join(lines))
    print("Files created:")
    print(args.output_summary)
    print(args.output_failures)


if __name__ == "__main__":
    main()
