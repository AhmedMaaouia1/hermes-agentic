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
    result = run_pipeline(folder_path)

    print("Pipeline completed.")

    # --------------------------------------------------
    # 1) Affichage lisible
    # --------------------------------------------------
    for fp in result:
        print("-" * 60)
        print(f"File     : {fp.filename}")
        print(f"Type     : {fp.file_type}")
        print(f"Topic    : {fp.topic}")
        print(f"Keywords : {', '.join(fp.keywords)}")
        print(f"Summary  : {fp.summary}")
        print(f"Signals  : {fp.signals}")

    # --------------------------------------------------
    # 2) Sauvegarde JSON 
    # --------------------------------------------------
    output_path = Path("logs/output_analyst.json")
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            [fp.model_dump() for fp in result],
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nAnalyst results saved to {output_path}")


if __name__ == "__main__":
    main()
