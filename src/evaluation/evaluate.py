import json
import argparse
from pathlib import Path
from typing import Dict, List

from metrics import (
    exact_match_accuracy,
    category_accuracy,
    subcategory_accuracy,
    confusion_matrix,
    confidence_analysis,
)


# ======================================================
# Loading helpers
# ======================================================
def load_ground_truth(path: Path) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_predictions(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    predictions = []
    for item in raw:
        # On récupère directement les champs
        category = item["category"]
        subcategory = item.get("subcategory")
        if subcategory:
            label = f"{category}/{subcategory}"
        else:
            label = category

        predictions.append({
            "filename": item["filename"],
            "label": label,
            "confidence": item.get("confidence", 0.0),
        })

    return predictions

def load_hierarchy_predictions(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    file_to_folder = raw["file_to_folder"]
    predictions = []
    for filename, folder in file_to_folder.items():
        predictions.append({
            "filename": filename,
            "label": folder,
            "confidence": 1.0
        })
    return predictions

def load_reviewer_predictions(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # On marque tous les fichiers présents dans au moins une issue comme "Review"
    files_to_review = set()
    for issue in raw.get("issues", []):
        files_to_review.update(issue.get("affected_files", []))
    predictions = []
    for filename in files_to_review:
        predictions.append({
            "filename": filename,
            "label": "Review",
            "confidence": 1.0
        })
    return predictions

def build_predicted_map(predictions: List[dict]) -> Dict[str, str]:
    return {p["filename"]: p["label"] for p in predictions}


# ======================================================
# CLI
# ======================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HERMES Agentic predictions")
    parser.add_argument(
        "--ground-truth",
        required=True,
        type=Path,
        help="Path to ground_truth.json",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help="Path to logs/output_categorizer.json",
    )
    parser.add_argument(
        "--output",
        default=Path("evaluation_report.json"),
        type=Path,
        help="Output JSON report",
    )

    args = parser.parse_args()

    # Load data
    y_true = load_ground_truth(args.ground_truth)
    # Détection automatique du format de prédiction
    with open(args.predictions, "r", encoding="utf-8") as f:
        first = json.load(f)
    if isinstance(first, dict) and "file_to_folder" in first:
        predictions = load_hierarchy_predictions(args.predictions)
    elif isinstance(first, dict) and "issues" in first:
        predictions = load_reviewer_predictions(args.predictions)
    else:
        predictions = load_predictions(args.predictions)
    y_pred = build_predicted_map(predictions)

    # Metrics
    exact_acc = exact_match_accuracy(y_true, y_pred)
    cat_acc = category_accuracy(y_true, y_pred)
    sub_acc = subcategory_accuracy(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    conf_analysis = confidence_analysis(predictions, y_true)

    evaluated_files = len(
        [f for f in y_true.keys() if f in y_pred]
    )

    report = {
        "files_evaluated": evaluated_files,
        "exact_match_accuracy": exact_acc,
        "category_accuracy": cat_acc,
        "subcategory_accuracy": sub_acc,
        "confidence_analysis": conf_analysis,
        "confusion_matrix": conf_matrix,
    }

    # Save JSON report
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Console report
    print("\n=== HERMES AGENTIC — EVALUATION REPORT ===\n")
    print(f"Files evaluated       : {evaluated_files}")
    print(f"Exact match accuracy  : {exact_acc:.3f}")
    print(f"Category accuracy     : {cat_acc:.3f}")
    print(f"Subcategory accuracy  : {sub_acc:.3f}\n")

    print("Confidence analysis:")
    print(f"  ≥ 0.7 accuracy      : {conf_analysis['high_confidence_accuracy']:.3f}")
    print(f"  < 0.7 accuracy      : {conf_analysis['low_confidence_accuracy']:.3f}")

    print("\nEvaluation report saved to:", args.output)

if __name__ == "__main__":
    main()
