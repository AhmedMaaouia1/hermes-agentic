"""
main.py

Point d'entrée CLI de HERMES Agentic.

Responsabilité UNIQUE :
- Lancer le pipeline
- Afficher les résultats de manière lisible pour un humain

IMPORTANT :
- AUCUNE logique métier
- AUCUNE écriture de fichiers (fait dans pipeline.py)
"""

import sys
from orchestration.pipeline import run_pipeline


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python src/main.py <path_to_folder>")
        sys.exit(1)

    folder_path = sys.argv[1]
    print(f"Running HERMES Agentic on folder: {folder_path}")

    # --------------------------------------------------
    # Exécution du pipeline
    # --------------------------------------------------
    results_pipe = run_pipeline(folder_path)

    profiles = results_pipe.file_profiles
    categorizations = results_pipe.categorizations
    hierarchy_plan = results_pipe.hierarchy
    review_result = results_pipe.review

    print("\nPipeline completed successfully.")

    # --------------------------------------------------
    # 1) Résultats Analyst + Categorizer (fichier par fichier)
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("FILE ANALYSIS & CATEGORIZATION")
    print("=" * 60)

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
    # 2) Résultat Planner (hiérarchie globale)
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
    # 3) Résultat Reviewer (critique & suggestions)
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("REVIEWER REPORT")
    print("=" * 60)

    if review_result.issues:
        print("\nISSUES DETECTED:")
        for issue in review_result.issues:
            print(f"- [{issue.severity.upper()}] {issue.issue_type}")
            print(f"  {issue.description}")
            if issue.affected_files:
                print(f"  Files: {issue.affected_files}")
    else:
        print("\nNo issues detected.")

    if review_result.suggestions:
        print("\nSUGGESTIONS:")
        for suggestion in review_result.suggestions:
            print(f"- Action     : {suggestion.action}")
            print(f"  Target     : {suggestion.target}")
            print(f"  Suggestion : {suggestion.suggestion}")
    else:
        print("\nNo suggestions proposed.")

    print("\nAll detailed outputs are available in the 'logs/' directory.")


if __name__ == "__main__":
    main()
