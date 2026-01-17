import sys
import json
from pathlib import Path

from orchestration.pipeline import run_pipeline


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/main.py <path_to_folder>")
        sys.exit(1)

    folder_path = sys.argv[1]
    print(f"Running HERMES Agentic on folder: {folder_path}")

    profiles, categorizations = run_pipeline(folder_path)

    print("Pipeline completed.")

    # --------------------------------------------------
    # 1) Affichage lisible
    # --------------------------------------------------
    for fp, cat in zip(profiles, categorizations):
        print("-" * 60)
        print(f"File       : {fp.filename}")
        print(f"Type       : {fp.file_type}")
        print(f"Topic      : {fp.topic}")
        print(f"Keywords   : {', '.join(fp.keywords)}")
        print(f"Summary    : {fp.summary}")
        print(f"Category   : {cat.category}")
        print(f"Subcategory: {cat.subcategory}")
        print(f"Confidence : {cat.confidence}")
        print(f"Source     : {cat.decision_source}")
        print(f"Rationale  : {cat.rationale}")

    # --------------------------------------------------
    # 2) Sauvegarde JSON (Analyst + Categorizer)
    # --------------------------------------------------
    output_path = Path("logs/output_categorizer.json")
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "profile": fp.model_dump(),
                    "categorization": cat.model_dump()
                }
                for fp, cat in zip(profiles, categorizations)
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nCategorization results saved to {output_path}")


if __name__ == "__main__":
    main()
