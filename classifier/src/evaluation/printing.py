try:
    from .accuracy import accuracy
    from .confusion import (
        find_most_common_confusions,
        calculate_intent_error_summary
    )
except ImportError:
    from accuracy import accuracy
    from confusion import (
        find_most_common_confusions,
        calculate_intent_error_summary
    )


# ============================================================
# PRINT CLASSIFICATION REPORT
# ============================================================

def print_classification_report(
    report,
    macro_metrics
):
    """
    Print a readable classification report.
    """

    print(
        "\nClassification Report"
    )

    print(
        "=" * 80
    )

    print(
        f"{'Intent':<35}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'Support':>10}"
    )

    print(
        "-" * 81
    )

    for label in sorted(report):

        metrics = report[label]

        print(
            f"{label:<35}"
            f"{metrics['precision']:>12.4f}"
            f"{metrics['recall']:>12.4f}"
            f"{metrics['f1']:>12.4f}"
            f"{metrics['support']:>10}"
        )

    print(
        "-" * 81
    )

    print(
        f"{'Macro Average':<35}"
        f"{macro_metrics['precision']:>12.4f}"
        f"{macro_metrics['recall']:>12.4f}"
        f"{macro_metrics['f1']:>12.4f}"
    )


# ============================================================
# PRINT MOST COMMON ERRORS
# ============================================================

def print_most_common_errors(
    y_true,
    y_pred,
    limit=10
):
    """
    Print the most common classification mistakes.
    """

    errors = find_most_common_confusions(
        y_true,
        y_pred,
        limit
    )

    print(
        "\nMost Common Classification Errors"
    )

    print(
        "=" * 80
    )

    if not errors:

        print(
            "No classification errors."
        )

        return

    for index, (
        actual,
        predicted,
        count
    ) in enumerate(
        errors,
        start=1
    ):

        print(
            f"{index:>2}. "
            f"{actual} "
            f"-> "
            f"{predicted} "
            f"({count} errors)"
        )


# ============================================================
# PRINT PER-INTENT ERROR SUMMARY
# ============================================================

def print_intent_error_summary(
    y_true,
    y_pred
):
    """
    Print correct and incorrect predictions
    for each intent.
    """

    summary = (
        calculate_intent_error_summary(
            y_true,
            y_pred
        )
    )

    print(
        "\nPer-Intent Error Summary"
    )

    print(
        "=" * 80
    )

    for label in sorted(summary):

        metrics = summary[label]

        print(
            f"\n{label}"
        )

        print(
            f"  Correct:   "
            f"{metrics['correct']}"
        )

        print(
            f"  Incorrect: "
            f"{metrics['incorrect']}"
        )

        print(
            f"  Total:     "
            f"{metrics['total']}"
        )

        print(
            f"  Accuracy:  "
            f"{metrics['accuracy'] * 100:.2f}%"
        )


# ============================================================
# PRINT ALL EVALUATION RESULTS
# ============================================================

def print_evaluation_results(
    y_true,
    y_pred,
    report,
    macro_metrics
):
    """
    Print the complete model evaluation.

    The output is intentionally kept readable
    for terminal use.
    """

    test_accuracy = accuracy(
        y_true,
        y_pred
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "MODEL EVALUATION"
    )

    print(
        "=" * 80
    )

    print(
        f"\nTest records: "
        f"{len(y_true)}"
    )

    print(
        f"Accuracy:     "
        f"{test_accuracy:.4f} "
        f"({test_accuracy * 100:.2f}%)"
    )

    print(
        f"Macro F1:     "
        f"{macro_metrics['f1']:.4f}"
    )

    print_classification_report(
        report,
        macro_metrics
    )

    print_most_common_errors(
        y_true,
        y_pred
    )

    print_intent_error_summary(
        y_true,
        y_pred
    )