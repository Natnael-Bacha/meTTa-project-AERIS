import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from .report import calculate_classification_report
    from .printing import print_evaluation_results
except ImportError:
    from report import calculate_classification_report
    from printing import print_evaluation_results


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Import project modules
    # --------------------------------------------------------

    from data_preparation import (
        load_dataset,
        remove_duplicates,
        stratified_split
    )

    from naive_bayes.naive_bayes import (
        MultinomialNaiveBayes
    )

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    data = load_dataset()

    # --------------------------------------------------------
    # 2. Remove duplicates
    # --------------------------------------------------------

    data = remove_duplicates(
        data
    )

    print(
        "\nDataset"
    )

    print(
        "=" * 80
    )

    print(
        f"Total records after duplicate removal: "
        f"{len(data)}"
    )

    # --------------------------------------------------------
    # 3. Split dataset
    # --------------------------------------------------------

    train_data, test_data = (
        stratified_split(data)
    )

    print(
        f"Training records: "
        f"{len(train_data)}"
    )

    print(
        f"Test records:     "
        f"{len(test_data)}"
    )

    # --------------------------------------------------------
    # 4. Create classifier
    # --------------------------------------------------------

    classifier = (
        MultinomialNaiveBayes()
    )

    # --------------------------------------------------------
    # 5. Train classifier
    # --------------------------------------------------------

    classifier.fit(
        train_data["instruction"].tolist(),
        train_data["intent"].tolist()
    )

    # --------------------------------------------------------
    # 6. Predict test data
    # --------------------------------------------------------

    y_true = (
        test_data["intent"].tolist()
    )

    y_pred = classifier.predict(
        test_data["instruction"].tolist()
    )

    # --------------------------------------------------------
    # 7. Calculate classification report
    # --------------------------------------------------------

    report, macro_metrics = (
        calculate_classification_report(
            y_true,
            y_pred
        )
    )

    # --------------------------------------------------------
    # 8. Print results
    # --------------------------------------------------------

    print_evaluation_results(
        y_true,
        y_pred,
        report,
        macro_metrics
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()