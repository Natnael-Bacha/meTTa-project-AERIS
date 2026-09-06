import numpy as np

try:
    from .confusion import calculate_confusion_counts
    from .metrics import (
        calculate_precision,
        calculate_recall,
        calculate_f1
    )
except ImportError:
    from confusion import calculate_confusion_counts
    from metrics import (
        calculate_precision,
        calculate_recall,
        calculate_f1
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

def calculate_classification_report(
    y_true,
    y_pred
):
    """
    Calculate precision, recall, F1-score,
    and support for every intent.

    Also calculates macro averages.
    """

    confusion_counts = (
        calculate_confusion_counts(
            y_true,
            y_pred
        )
    )

    report = {}

    precision_values = []
    recall_values = []
    f1_values = []

    for label, counts in (
        confusion_counts.items()
    ):

        precision = calculate_precision(
            counts["tp"],
            counts["fp"]
        )

        recall = calculate_recall(
            counts["tp"],
            counts["fn"]
        )

        f1 = calculate_f1(
            precision,
            recall
        )

        report[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": counts["support"]
        }

        precision_values.append(
            precision
        )

        recall_values.append(
            recall
        )

        f1_values.append(
            f1
        )

    number_of_classes = len(report)

    if number_of_classes == 0:
        return report, {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }

    macro_precision = float(
        np.mean(
            precision_values
        )
    )

    macro_recall = float(
        np.mean(
            recall_values
        )
    )

    macro_f1_score = float(
        np.mean(
            f1_values
        )
    )

    return report, {
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1_score
    }


# ============================================================
# MACRO F1
# ============================================================

def macro_f1(y_true, y_pred):
    """
    Calculate the macro-averaged F1-score directly
    from true and predicted labels.

    Macro F1 = mean of the per-class F1-scores.
    """

    _, macro_metrics = (
        calculate_classification_report(
            y_true,
            y_pred
        )
    )

    return macro_metrics["f1"]