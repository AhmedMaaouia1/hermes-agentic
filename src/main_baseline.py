import json
from pathlib import Path

from src.parsing.file_parser import parse_directory
from src.baseline.mono_agent import run_baseline

INPUT_DIR = Path("data/downloads_raw")
OUTPUT_FILE = Path("logs/baseline_result.json")

def main():
    parsed_files = parse_directory(INPUT_DIR)
    result = run_baseline(parsed_files)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)

    print(f"[BASELINE] Result written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
