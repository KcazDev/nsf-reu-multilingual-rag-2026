import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BM25_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "bm25_retrieval_results.csv"
)

E5_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "chunked_e5_retrieval_results.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "bm25_e5_hybrid_comparison.csv"
)

RRF_CONSTANT = 60


def load_csv(file_path: Path) -> list[dict]:
    """Load rows from a CSV file."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required results file was not found:\n{file_path}"
        )

    with file_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def parse_integer_list(value: str) -> list[int]:
    """Convert a comma-separated value into a list of integers."""
    if not value or not value.strip():
        return []

    return [
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    ]


def reciprocal_rank_fusion(
    bm25_ranking: list[int],
    e5_ranking: list[int],
) -> tuple[list[int], dict[int, float]]:
    """
    Combine BM25 and E5 rankings using Reciprocal Rank Fusion.

    Higher RRF scores represent better combined rankings.
    """
    scores = {}

    for rank_position, article_number in enumerate(
        bm25_ranking,
        start=1,
    ):
        scores.setdefault(article_number, 0.0)

        scores[article_number] += (
            1.0 / (RRF_CONSTANT + rank_position)
        )

    for rank_position, article_number in enumerate(
        e5_ranking,
        start=1,
    ):
        scores.setdefault(article_number, 0.0)

        scores[article_number] += (
            1.0 / (RRF_CONSTANT + rank_position)
        )

    combined_ranking = sorted(
        scores,
        key=lambda article_number: (
            -scores[article_number],
            (
                bm25_ranking.index(article_number)
                if article_number in bm25_ranking
                else len(bm25_ranking)
            ),
            article_number,
        ),
    )

    return combined_ranking, scores


def calculate_metrics(
    ranking: list[int],
    true_support_articles: list[int],
    k: int,
) -> tuple[float, float, bool]:
    """Calculate Precision@K, Recall@K, and complete support."""
    top_k = ranking[:k]

    support_found = [
        article_number
        for article_number in top_k
        if article_number in true_support_articles
    ]

    precision = len(support_found) / k

    recall = (
        len(support_found)
        / len(true_support_articles)
        if true_support_articles
        else 0.0
    )

    complete_support = set(
        true_support_articles
    ).issubset(
        set(top_k)
    )

    return precision, recall, complete_support


def calculate_reciprocal_rank(
    ranking: list[int],
    true_support_articles: list[int],
) -> float:
    """Calculate reciprocal rank of the first support article."""
    for rank_position, article_number in enumerate(
        ranking,
        start=1,
    ):
        if article_number in true_support_articles:
            return 1.0 / rank_position

    return 0.0


def calculate_method_summary(
    rows: list[dict],
    ranking_field: str,
) -> dict:
    """Calculate aggregate retrieval metrics for one method."""
    totals = {
        "precision_at_1": 0.0,
        "recall_at_1": 0.0,
        "precision_at_2": 0.0,
        "recall_at_2": 0.0,
        "precision_at_3": 0.0,
        "recall_at_3": 0.0,
        "reciprocal_rank": 0.0,
        "complete_at_2": 0,
        "complete_at_3": 0,
    }

    for row in rows:
        ranking = parse_integer_list(
            row[ranking_field]
        )

        true_support = parse_integer_list(
            row["true_support_articles"]
        )

        precision_1, recall_1, _ = calculate_metrics(
            ranking,
            true_support,
            k=1,
        )

        precision_2, recall_2, complete_2 = (
            calculate_metrics(
                ranking,
                true_support,
                k=2,
            )
        )

        precision_3, recall_3, complete_3 = (
            calculate_metrics(
                ranking,
                true_support,
                k=3,
            )
        )

        reciprocal_rank = calculate_reciprocal_rank(
            ranking,
            true_support,
        )

        totals["precision_at_1"] += precision_1
        totals["recall_at_1"] += recall_1
        totals["precision_at_2"] += precision_2
        totals["recall_at_2"] += recall_2
        totals["precision_at_3"] += precision_3
        totals["recall_at_3"] += recall_3
        totals["reciprocal_rank"] += reciprocal_rank
        totals["complete_at_2"] += int(complete_2)
        totals["complete_at_3"] += int(complete_3)

    number_of_rows = len(rows)

    return {
        "precision_at_1": (
            totals["precision_at_1"] / number_of_rows
        ),
        "recall_at_1": (
            totals["recall_at_1"] / number_of_rows
        ),
        "precision_at_2": (
            totals["precision_at_2"] / number_of_rows
        ),
        "recall_at_2": (
            totals["recall_at_2"] / number_of_rows
        ),
        "precision_at_3": (
            totals["precision_at_3"] / number_of_rows
        ),
        "recall_at_3": (
            totals["recall_at_3"] / number_of_rows
        ),
        "mrr": (
            totals["reciprocal_rank"] / number_of_rows
        ),
        "complete_at_2": totals["complete_at_2"],
        "complete_at_3": totals["complete_at_3"],
    }


def print_summary(
    method_name: str,
    summary: dict,
    number_of_examples: int,
) -> None:
    """Print one retrieval method's summary."""
    print(f"\n{method_name}")
    print("-" * len(method_name))

    print(
        f"Precision@1: "
        f"{summary['precision_at_1']:.3f}"
    )

    print(
        f"Recall@1: "
        f"{summary['recall_at_1']:.3f}"
    )

    print(
        f"Precision@2: "
        f"{summary['precision_at_2']:.3f}"
    )

    print(
        f"Recall@2: "
        f"{summary['recall_at_2']:.3f}"
    )

    print(
        f"Precision@3: "
        f"{summary['precision_at_3']:.3f}"
    )

    print(
        f"Recall@3: "
        f"{summary['recall_at_3']:.3f}"
    )

    print(
        f"Mean Reciprocal Rank: "
        f"{summary['mrr']:.3f}"
    )

    print(
        f"Complete support in top 2: "
        f"{summary['complete_at_2']}/"
        f"{number_of_examples}"
    )

    print(
        f"Complete support in top 3: "
        f"{summary['complete_at_3']}/"
        f"{number_of_examples}"
    )


def main() -> None:
    bm25_rows = load_csv(BM25_FILE)
    e5_rows = load_csv(E5_FILE)

    bm25_by_id = {
        str(row["xrag_id"]): row
        for row in bm25_rows
    }

    e5_by_id = {
        str(row["xrag_id"]): row
        for row in e5_rows
    }

    common_ids = [
        str(row["xrag_id"])
        for row in bm25_rows
        if str(row["xrag_id"]) in e5_by_id
    ]

    if len(common_ids) != len(bm25_rows):
        raise ValueError(
            "BM25 and E5 files do not contain the same examples."
        )

    comparison_rows = []

    bm25_only_complete = 0
    e5_only_complete = 0
    both_complete = 0
    neither_complete = 0
    hybrid_only_complete = 0

    for xrag_id in common_ids:
        bm25_row = bm25_by_id[xrag_id]
        e5_row = e5_by_id[xrag_id]

        true_support = parse_integer_list(
            bm25_row["true_support_articles"]
        )

        bm25_ranking = parse_integer_list(
            bm25_row["bm25_full_ranking"]
        )

        e5_ranking = parse_integer_list(
            e5_row["e5_full_ranking"]
        )

        hybrid_ranking, hybrid_scores = (
            reciprocal_rank_fusion(
                bm25_ranking,
                e5_ranking,
            )
        )

        _, bm25_recall_2, bm25_complete_2 = (
            calculate_metrics(
                bm25_ranking,
                true_support,
                k=2,
            )
        )

        _, e5_recall_2, e5_complete_2 = (
            calculate_metrics(
                e5_ranking,
                true_support,
                k=2,
            )
        )

        _, hybrid_recall_2, hybrid_complete_2 = (
            calculate_metrics(
                hybrid_ranking,
                true_support,
                k=2,
            )
        )

        _, hybrid_recall_3, hybrid_complete_3 = (
            calculate_metrics(
                hybrid_ranking,
                true_support,
                k=3,
            )
        )

        if bm25_complete_2 and e5_complete_2:
            both_complete += 1
        elif bm25_complete_2:
            bm25_only_complete += 1
        elif e5_complete_2:
            e5_only_complete += 1
        else:
            neither_complete += 1

        if (
            hybrid_complete_2
            and not bm25_complete_2
            and not e5_complete_2
        ):
            hybrid_only_complete += 1

        ordered_hybrid_scores = [
            round(hybrid_scores[article], 6)
            for article in hybrid_ranking
        ]

        comparison_rows.append(
            {
                "example_number": (
                    bm25_row["example_number"]
                ),
                "xrag_id": xrag_id,
                "language": bm25_row["language"],
                "task": bm25_row["task"],
                "question_type": (
                    bm25_row["question_type"]
                ),
                "question": bm25_row["question"],
                "true_support_articles": ",".join(
                    map(str, true_support)
                ),
                "bm25_ranking": ",".join(
                    map(str, bm25_ranking)
                ),
                "e5_ranking": ",".join(
                    map(str, e5_ranking)
                ),
                "hybrid_ranking": ",".join(
                    map(str, hybrid_ranking)
                ),
                "hybrid_scores": ",".join(
                    map(str, ordered_hybrid_scores)
                ),
                "bm25_recall_at_2": bm25_recall_2,
                "e5_recall_at_2": e5_recall_2,
                "hybrid_recall_at_2": hybrid_recall_2,
                "hybrid_recall_at_3": hybrid_recall_3,
                "bm25_complete_at_2": bm25_complete_2,
                "e5_complete_at_2": e5_complete_2,
                "hybrid_complete_at_2": (
                    hybrid_complete_2
                ),
                "hybrid_complete_at_3": (
                    hybrid_complete_3
                ),
            }
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=comparison_rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(comparison_rows)

    bm25_summary = calculate_method_summary(
        comparison_rows,
        ranking_field="bm25_ranking",
    )

    e5_summary = calculate_method_summary(
        comparison_rows,
        ranking_field="e5_ranking",
    )

    hybrid_summary = calculate_method_summary(
        comparison_rows,
        ranking_field="hybrid_ranking",
    )

    number_of_examples = len(comparison_rows)

    print("Retriever comparison completed.")
    print(
        f"Examples compared: "
        f"{number_of_examples}"
    )

    print(
        f"Results saved to: "
        f"{OUTPUT_FILE}"
    )

    print_summary(
        "BM25",
        bm25_summary,
        number_of_examples,
    )

    print_summary(
        "Chunked Multilingual E5",
        e5_summary,
        number_of_examples,
    )

    print_summary(
        "BM25 + E5 Hybrid RRF",
        hybrid_summary,
        number_of_examples,
    )

    print("\nTop-two complementarity")
    print("-----------------------")

    print(
        f"Both complete: "
        f"{both_complete}"
    )

    print(
        f"BM25 only complete: "
        f"{bm25_only_complete}"
    )

    print(
        f"E5 only complete: "
        f"{e5_only_complete}"
    )

    print(
        f"Neither complete: "
        f"{neither_complete}"
    )

    print(
        f"Hybrid complete when neither individual "
        f"retriever was complete: "
        f"{hybrid_only_complete}"
    )


if __name__ == "__main__":
    main()