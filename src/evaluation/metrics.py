from typing import Dict, List, Tuple
from collections import defaultdict


# ======================================================
# Helpers
# ======================================================
def _split_label(label: str) -> Tuple[str, str | None]:
    """
    "Category/Subcategory" -> ("Category", "Subcategory")
    "Category"             -> ("Category", None)
    """
    if "/" in label:
        cat, sub = label.split("/", 1)
        return cat, sub
    return label, None


# ======================================================
# Core metrics
# ======================================================
def exact_match_accuracy(
    y_true: Dict[str, str],
    y_pred: Dict[str, str],
) -> float:
    """
    Strict accuracy:
    predicted label must exactly match ground truth label.
    """
    if not y_true:
        return 0.0

    correct = 0
    total = 0

    for fname, true_label in y_true.items():
        if fname not in y_pred:
            continue
        total += 1
        if y_pred[fname] == true_label:
            correct += 1

    return correct / total if total > 0 else 0.0


def category_accuracy(
    y_true: Dict[str, str],
    y_pred: Dict[str, str],
) -> float:
    """
    Category-level accuracy (ignores subcategories).
    """
    if not y_true:
        return 0.0

    correct = 0
    total = 0

    for fname, true_label in y_true.items():
        if fname not in y_pred:
            continue

        true_cat, _ = _split_label(true_label)
        pred_cat, _ = _split_label(y_pred[fname])

        total += 1
        if true_cat == pred_cat:
            correct += 1

    return correct / total if total > 0 else 0.0


def subcategory_accuracy(
    y_true: Dict[str, str],
    y_pred: Dict[str, str],
) -> float:
    """
    Subcategory accuracy.
    Evaluated ONLY when ground truth contains a subcategory.
    """
    correct = 0
    total = 0

    for fname, true_label in y_true.items():
        if fname not in y_pred:
            continue

        true_cat, true_sub = _split_label(true_label)
        if true_sub is None:
            continue  # no subcategory in GT → ignored

        pred_cat, pred_sub = _split_label(y_pred[fname])

        total += 1
        if true_cat == pred_cat and true_sub == pred_sub:
            correct += 1

    return correct / total if total > 0 else 0.0


def confusion_matrix(
    y_true: Dict[str, str],
    y_pred: Dict[str, str],
) -> Dict[str, Dict[str, int]]:
    """
    Confusion matrix at CATEGORY level.
    """
    matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for fname, true_label in y_true.items():
        if fname not in y_pred:
            continue

        true_cat, _ = _split_label(true_label)
        pred_cat, _ = _split_label(y_pred[fname])

        matrix[true_cat][pred_cat] += 1

    return {k: dict(v) for k, v in matrix.items()}


def confidence_analysis(
    categorizations: List[dict],
    y_true: Dict[str, str],
    threshold: float = 0.7,
) -> Dict[str, float]:
    """
    Compare accuracy for high-confidence vs low-confidence predictions.
    """
    high_correct = high_total = 0
    low_correct = low_total = 0

    for item in categorizations:
        fname = item.get("filename")
        if fname not in y_true:
            continue

        confidence = float(item.get("confidence", 0.0))
        pred_label = item.get("label")
        true_label = y_true[fname]

        if confidence >= threshold:
            high_total += 1
            if pred_label == true_label:
                high_correct += 1
        else:
            low_total += 1
            if pred_label == true_label:
                low_correct += 1

    return {
        "high_confidence_accuracy": high_correct / high_total if high_total else 0.0,
        "low_confidence_accuracy": low_correct / low_total if low_total else 0.0,
    }
