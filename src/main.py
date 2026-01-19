import sys
import json
from pathlib import Path

from orchestration.pipeline import run_pipeline


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python src/main.py <path_to_folder>")
        sys.exit(1)

    folder_path = sys.argv[1]
    print(f"Running HERMES Agentic on folder: {folder_path}")

    results_pipe = run_pipeline(folder_path)

    profiles = results_pipe.fileprofiles
    categorizations = results_pipe.categorizationRes
    hierarchy_plan = results_pipe.initial_structure
    review_result = results_pipe.review

    print("Pipeline completed.")

    # --------------------------------------------------
    # 1) Affichage lisible (Analyst + Categorizer)
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
    # 2) Affichage lisible (Planner)
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("PROPOSED FOLDER HIERARCHY")
    print("=" * 60)

    for folder, files in hierarchy_plan.folder_structure.items():
        print(f"\n[{folder}]")
        for filename in files:
            print(f"  - {filename}")

    if hierarchy_plan.warnings:
        print("\nWARNINGS:")
        for warning in hierarchy_plan.warnings:
            print(f" - {warning}")

    print("\nRATIONALE:")
    print(hierarchy_plan.rationale)

    # --------------------------------------------------
    # 3) Affichage lisible (Reviewer)
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("REVIEWER REPORT")
    print("=" * 60)

    if review_result.issues:
        print("\nISSUES DETECTED:")
        for issue in review_result.issues:
            print(f"- [{issue.severity.upper()}] {issue.issue_type}")
            print(f"  {issue.description}")
            print(f"  Files: {issue.affected_files}")
    else:
        print("\nNo issues detected.")

    if review_result.suggestions:
        print("\nSUGGESTIONS:")
        for suggestion in review_result.suggestions:
            print(f"- Action : {suggestion.action}")
            if suggestion.source:
                print(f"  Source : {suggestion.source}")
            if suggestion.target:
                print(f"  Target : {suggestion.target}")
            print(f"  Reason : {suggestion.reason}")
    else:
        print("\nNo suggestions proposed.")

    # --------------------------------------------------
    # 4) Sauvegarde JSON (Analyst + Categorizer)
    # --------------------------------------------------
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    categorizer_output_path = logs_dir / "output_categorizer.json"
    with open(categorizer_output_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "profile": fp.model_dump(),
                    "categorization": cat.model_dump(),
                }
                for fp, cat in zip(profiles, categorizations)
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------
    # 5) Sauvegarde JSON (Planner)
    # --------------------------------------------------
    planner_output_path = logs_dir / "output_hierarchy_plan.json"
    with open(planner_output_path, "w", encoding="utf-8") as f:
        json.dump(
            hierarchy_plan.model_dump(),
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------
    # 6) Sauvegarde JSON (Reviewer)
    # --------------------------------------------------
    reviewer_output_path = logs_dir / "output_reviewer.json"
    with open(reviewer_output_path, "w", encoding="utf-8") as f:
        json.dump(
            review_result.model_dump(),
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nCategorization results saved to {categorizer_output_path}")
    print(f"Hierarchy proposal saved to {planner_output_path}")
    print(f"Reviewer report saved to {reviewer_output_path}")


if __name__ == "__main__":
    main()