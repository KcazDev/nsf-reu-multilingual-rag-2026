import csv
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"
SUMMARY_MD = OUTPUTS / "xrag_current_research_summary.md"
SUMMARY_TXT = OUTPUTS / "xrag_current_research_summary.txt"
CONDITION_SUMMARY = OUTPUTS / "openai_retrieval_condition_summary.csv"
BM25_TOP2_SCORED = OUTPUTS / "openai_bm25_top2_review_scored.csv"
BM25_RETRIEVAL = OUTPUTS / "bm25_retrieval_results.csv"
E5_RETRIEVAL = OUTPUTS / "chunked_e5_retrieval_results.csv"
HYBRID_RETRIEVAL = OUTPUTS / "bm25_e5_hybrid_comparison.csv"


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "yes", "1"}


def condition_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return [f"Condition summary not found yet: {CONDITION_SUMMARY}"]
    lines = [
        "| Condition | Score | Accuracy | Fully correct | Partial | Wrong | Unsupported |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        pct = to_float(r.get("correctness_percentage"))
        if pct <= 1:
            pct *= 100
        lines.append(
            f"| {r.get('condition','')} | {r.get('total_score','')}/{r.get('possible_score','')} | "
            f"{pct:.1f}% | {r.get('fully_correct','')} | {r.get('partially_correct','')} | "
            f"{r.get('incorrect','')} | {r.get('unsupported_information','')} |"
        )
    return lines


def bm25_generation_summary(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return [f"BM25 Top-2 scored file not found yet: {BM25_TOP2_SCORED}"]
    scores = [to_int(r.get("correctness_score")) for r in rows]
    complete = [truthy(r.get("complete_support_at_2")) for r in rows]
    total = sum(scores)
    possible = len(rows) * 2
    complete_scores = [s for s, c in zip(scores, complete) if c]
    incomplete_scores = [s for s, c in zip(scores, complete) if not c]
    complete_acc = sum(complete_scores) / (len(complete_scores) * 2) * 100 if complete_scores else 0
    incomplete_acc = sum(incomplete_scores) / (len(incomplete_scores) * 2) * 100 if incomplete_scores else 0
    unsupported = sum(truthy(r.get("unsupported_information")) for r in rows)
    over_35 = sum(str(r.get("over_35_words", "")).strip().lower() == "yes" for r in rows)
    errors = Counter(r.get("primary_error_category", "").strip() for r in rows if r.get("primary_error_category", "").strip())
    lines = [
        f"- BM25 Top-2 total score: {total}/{possible} = {total/possible*100:.1f}%",
        f"- Complete support retrieved: {sum(complete)}/{len(rows)}",
        f"- Accuracy when complete support was retrieved: {complete_acc:.1f}%",
        f"- Accuracy when support was incomplete: {incomplete_acc:.1f}%",
        f"- Unsupported-information flags: {unsupported}/{len(rows)}",
        f"- Responses over 35 words: {over_35}/{len(rows)}",
        "",
        "BM25 Top-2 error categories:",
    ]
    for category, count in errors.most_common():
        lines.append(f"- {category}: {count}")
    return lines


def retrieval_summary(path: Path, name: str) -> list[str]:
    rows = read_csv(path)
    if not rows:
        return [f"- {name}: file not found yet: {path}"]
    complete_keys = ["complete_support_at_2", "complete_support_top2", "complete_at_2", "bm25_complete_at_2", "e5_complete_at_2", "hybrid_complete_at_2"]
    recall_keys = ["recall_at_2", "retrieval_recall_at_2", "bm25_recall_at_2", "e5_recall_at_2", "hybrid_recall_at_2"]
    precision_keys = ["precision_at_2", "retrieval_precision_at_2", "bm25_precision_at_2", "e5_precision_at_2", "hybrid_precision_at_2"]
    def get(row: dict[str, Any], keys: list[str]) -> Any:
        for key in keys:
            if key in row:
                return row[key]
        return ""
    comp = sum(truthy(get(r, complete_keys)) for r in rows)
    avg_p = sum(to_float(get(r, precision_keys)) for r in rows) / len(rows)
    avg_r = sum(to_float(get(r, recall_keys)) for r in rows) / len(rows)
    return [
        f"- {name}: {len(rows)} examples",
        f"  - Complete support @2: {comp}/{len(rows)}",
        f"  - Average Precision@2: {avg_p:.3f}",
        f"  - Average Recall@2: {avg_r:.3f}",
    ]


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    lines = [
        "# XRAG Project Current Research Summary",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Project framing",
        "",
        "The project has two phases. First, test multilingual and cross-lingual RAG benchmarks such as XRAG to understand where current systems succeed and fail. Second, use those benchmark findings to build a multilingual RAG system with stronger retrieval, evidence filtering, and answer generation.",
        "",
        "## Current research question",
        "",
        "How does retrieved evidence quality affect multilingual RAG answer accuracy, language correctness, and unsupported-information errors?",
        "",
        "## Completed benchmark work",
        "",
        "- Built a 20-question Arabic/Spanish XRAG pilot.",
        "- Tested GPT-5.4 under no-documents, support-only, and all-documents conditions.",
        "- Tested BM25 retrieval, chunked multilingual E5 retrieval, and BM25+E5 hybrid retrieval.",
        "- Ran an end-to-end BM25 Top-2 → GPT-5.4 generation condition.",
        "- Manually scored answers with 0/1/2 correctness, refusal, language, unsupported information, and error category.",
        "",
        "## GPT-5.4 condition comparison",
        "",
    ]
    lines.extend(condition_table(read_csv(CONDITION_SUMMARY)))
    lines.extend(["", "## BM25 Top-2 generation summary", ""])
    lines.extend(bm25_generation_summary(read_csv(BM25_TOP2_SCORED)))
    lines.extend(["", "## Retrieval baseline summaries", ""])
    lines.extend(retrieval_summary(BM25_RETRIEVAL, "BM25"))
    lines.extend(retrieval_summary(E5_RETRIEVAL, "Chunked multilingual E5"))
    lines.extend(retrieval_summary(HYBRID_RETRIEVAL, "BM25+E5 hybrid"))
    lines.extend([
        "", "## Main findings so far", "",
        "1. Evidence selection strongly affects final answer quality.",
        "2. Providing all documents can hurt performance because distractors interfere with generation.",
        "3. BM25 was the strongest retrieval baseline in the 20-question pilot.",
        "4. Dense semantic retrieval did not automatically outperform keyword retrieval.",
        "5. Complete-support coverage is more informative than MRR for cross-document QA.",
        "", "## Next technical steps", "",
        "1. Expand retrieval testing to a 50-question XRAG sample.",
        "2. Evaluate BM25 at Top-1, Top-2, Top-3, and Top-4.",
        "3. Analyze retrieval failure patterns by language, task, and question type.",
        "4. Use those results to decide how many documents the first system should pass to the generator.",
        "5. Build the first version of a multilingual RAG pipeline.",
    ])
    text = "\n".join(lines)
    SUMMARY_MD.write_text(text, encoding="utf-8")
    SUMMARY_TXT.write_text(text, encoding="utf-8")
    print("XRAG CURRENT RESEARCH SUMMARY CREATED")
    print("=" * 70)
    print(SUMMARY_MD)
    print(SUMMARY_TXT)


if __name__ == "__main__":
    main()
