from pathlib import Path
from dotenv import load_dotenv
import os
import json


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
ENV_FILE = ROOT_DIR / ".env"


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0

    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1

    return count


def preview_first_record(path: Path):
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return json.loads(line)

    return None


def main():
    print("MULTILINGUAL RAG SYSTEM — SETUP CHECK")
    print("=" * 70)

    print(f"Root directory: {ROOT_DIR}")
    print(f"Data directory exists: {DATA_DIR.exists()}")
    print(f"Outputs directory exists: {OUTPUTS_DIR.exists()}")
    print(f".env file exists: {ENV_FILE.exists()}")

    load_dotenv(ENV_FILE)

    openai_key = os.getenv("OPENAI_API_KEY")
    print(f"OPENAI_API_KEY found: {bool(openai_key)}")

    dev_file = DATA_DIR / "xrag.dev.jsonl"
    sample_file = DATA_DIR / "xrag_eval_sample_50.jsonl"

    print()
    print("Data files")
    print("-" * 70)
    print(f"xrag.dev.jsonl exists: {dev_file.exists()}")
    print(f"xrag.dev.jsonl examples: {count_jsonl_lines(dev_file)}")
    print(f"xrag_eval_sample_50.jsonl exists: {sample_file.exists()}")
    print(f"xrag_eval_sample_50.jsonl examples: {count_jsonl_lines(sample_file)}")

    first_record = preview_first_record(sample_file)

    print()
    print("First sample preview")
    print("-" * 70)

    if first_record:
        print(f"XRAG ID: {first_record.get('id') or first_record.get('xrag_id')}")
        print(f"Language: {first_record.get('language')}")
        print(f"Task: {first_record.get('task')}")
        print(f"Question type: {first_record.get('question_type')}")
        print(f"Question: {first_record.get('question')}")
    else:
        print("No sample record found.")

    print()
    print("SETUP CHECK FINISHED")


if __name__ == "__main__":
    main()