# ============================================================
# PRECISION
# ============================================================

def calculate_precision(
    true_positive,
    false_positive
):
    """
    Calculate precision.

    Precision = TP / (TP + FP)
    """

    denominator = (
        true_positive +
        false_positive
    )

    if denominator == 0:
        return 0.0

    return (
        true_positive /
        denominator
    )


# ============================================================
# RECALL
# ============================================================

def calculate_recall(
    true_positive,
    false_negative
):
    """
    Calculate recall.

    Recall = TP / (TP + FN)
    """

    denominator = (
        true_positive +
        false_negative
    )

    if denominator == 0:
        return 0.0

    return (
        true_positive /
        denominator
    )


# ============================================================
# F1 SCORE
# ============================================================

def calculate_f1(
    precision,
    recall
):
    """
    Calculate F1-score.

    F1 = 2 * Precision * Recall
         ------------------------
         Precision + Recall
    """

    denominator = (
        precision +
        recall
    )

    if denominator == 0:
        return 0.0

    return (
        2 *
        precision *
        recall /
        denominator
    )