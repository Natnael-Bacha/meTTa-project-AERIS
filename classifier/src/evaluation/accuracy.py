import numpy as np


# ============================================================
# ACCURACY
# ============================================================

def accuracy(y_true, y_pred):
    """
    Calculate overall classification accuracy.

    Accuracy = correct predictions / total predictions
    """

    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same length."
        )

    if len(y_true) == 0:
        return 0.0

    return float(
        np.mean(y_true == y_pred)
    )