import numpy as np


# ============================================================
# CONFUSION COUNTS
# ============================================================

def calculate_confusion_counts(y_true, y_pred):
    """
    Calculate TP, FP, FN, and support for every class.

    Returns:
        Dictionary containing classification counts
        for every intent.
    """

    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)

    labels = np.unique(
        np.concatenate([y_true, y_pred])
    )

    metrics = {}

    for label in labels:

        true_positive = np.sum(
            (y_true == label) &
            (y_pred == label)
        )

        false_positive = np.sum(
            (y_true != label) &
            (y_pred == label)
        )

        false_negative = np.sum(
            (y_true == label) &
            (y_pred != label)
        )

        support = np.sum(
            y_true == label
        )

        metrics[label] = {
            "tp": int(true_positive),
            "fp": int(false_positive),
            "fn": int(false_negative),
            "support": int(support)
        }

    return metrics


# ============================================================
# MOST COMMON CLASSIFICATION ERRORS
# ============================================================

def find_most_common_confusions(
    y_true,
    y_pred,
    limit=10
):
    """
    Find the most common incorrect predictions.

    Returns:
        List of:

        (actual_intent, predicted_intent, count)
    """

    y_true = np.asarray(
        y_true,
        dtype=object
    )

    y_pred = np.asarray(
        y_pred,
        dtype=object
    )

    confusion_pairs = {}

    for actual, predicted in zip(
        y_true,
        y_pred
    ):

        if actual == predicted:
            continue

        key = (
            actual,
            predicted
        )

        confusion_pairs[key] = (
            confusion_pairs.get(
                key,
                0
            ) + 1
        )

    sorted_errors = sorted(
        confusion_pairs.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return [
        (
            actual,
            predicted,
            count
        )
        for (
            (actual, predicted),
            count
        )
        in sorted_errors[:limit]
    ]


# ============================================================
# PER-INTENT ERROR SUMMARY
# ============================================================

def calculate_intent_error_summary(
    y_true,
    y_pred
):
    """
    Calculate correct and incorrect predictions
    for every intent.

    Returns:

        {
            intent: {
                "correct": ...,
                "incorrect": ...,
                "total": ...,
                "accuracy": ...
            }
        }
    """

    y_true = np.asarray(
        y_true,
        dtype=object
    )

    y_pred = np.asarray(
        y_pred,
        dtype=object
    )

    labels = np.unique(y_true)

    summary = {}

    for label in labels:

        actual_mask = (
            y_true == label
        )

        correct = np.sum(
            actual_mask &
            (y_pred == label)
        )

        total = np.sum(
            actual_mask
        )

        incorrect = (
            total -
            correct
        )

        intent_accuracy = (
            correct / total
            if total > 0
            else 0.0
        )

        summary[label] = {
            "correct": int(correct),
            "incorrect": int(incorrect),
            "total": int(total),
            "accuracy": float(
                intent_accuracy
            )
        }

    return summary