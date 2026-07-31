import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "xrag.dev.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "xrag_eval_sample_50.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file was not found:\n{path}")
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


def save_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def select_balanced_by_language(examples: list[dict[str, Any]], languages: list[str], sample_size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_language = defaultdict(list)
    for ex in examples:
        lang = str(ex.get("language", "")).strip()
        if lang in languages:
            by_language[lang].append(ex)
    missing = [lang for lang in languages if not by_language[lang]]
    if missing:
        raise ValueError("No examples found for requested languages: " + ", ".join(missing))
    for lang in languages:
        rng.shuffle(by_language[lang])
    base = sample_size // len(languages)
    remainder = sample_size % len(languages)
    selected = []
    for idx, lang in enumerate(languages):
        target = base + (1 if idx < remainder else 0)
        if len(by_language[lang]) < target:
            raise ValueError(f"Not enough examples for {lang}: needed {target}, found {len(by_language[lang])}")
        selected.extend(by_language[lang][:target])
    rng.shuffle(selected)
    for i, ex in enumerate(selected, start=1):
        ex["_sample_example_number"] = i
    return selected


def print_distribution(examples: list[dict[str, Any]]) -> None:
    print("SAMPLE DISTRIBUTION")
    print("=" * 70)
    print(f"Total examples: {len(examples)}")
    print(f"Languages: {dict(Counter(ex.get('language') for ex in examples))}")
    print(f"Tasks: {dict(Counter(ex.get('task') for ex in examples))}")
    print(f"Question types: {dict(Counter(ex.get('q_type') for ex in examples))}")
    print(f"Answer types: {dict(Counter(ex.get('a_type') for ex in examples))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a larger XRAG evaluation sample.")
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--languages", nargs="+", default=["ar", "es"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    examples = load_jsonl(args.input_file)
    selected = select_balanced_by_language(examples, args.languages, args.sample_size, args.seed)
    save_jsonl(args.output_file, selected)
    print("XRAG EVALUATION SAMPLE CREATED")
    print("=" * 70)
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Seed: {args.seed}")
    print_distribution(selected)


if __name__ == "__main__":
    main()
