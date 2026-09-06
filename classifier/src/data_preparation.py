import pandas as pd


from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)

TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_dataset():
    """
    Load the dataset and keep only the columns required
    for customer-support intent classification.
    """

    data = pd.read_csv(DATASET_PATH)

    data = data[["instruction", "intent"]].copy()

    return data


def remove_duplicates(data):
    """
    Remove exact duplicate customer instructions.

    The dataset analysis confirmed that no identical instruction
    is associated with multiple intents.
    """

    original_size = len(data)

    data = (
        data
        .drop_duplicates(
            subset="instruction",
            keep="first"
        )
        .reset_index(drop=True)
    )

    removed = original_size - len(data)

    print("Duplicate Removal")
    print("=" * 60)
    print(f"Original records:   {original_size}")
    print(f"Duplicates removed: {removed}")
    print(f"Remaining records:  {len(data)}")

    return data


def stratified_split(data):
    """
    Perform a reproducible stratified train/test split.

    Each intent contributes approximately the same percentage
    of its examples to the training and testing sets.

    Parameters:
        data: DataFrame containing instruction and intent columns.

    Returns:
        train_data: Training DataFrame.
        test_data: Testing DataFrame.
    """

    train_parts = []
    test_parts = []

    # Process each intent independently.
    for intent in sorted(data["intent"].unique()):

        intent_data = data[
            data["intent"] == intent
        ].copy()

        # Shuffle examples belonging to this intent.
        intent_data = intent_data.sample(
            frac=1,
            random_state=RANDOM_STATE
        ).reset_index(drop=True)

        # Calculate the number of test examples.
        test_count = int(len(intent_data) * TEST_SIZE)

        # Make sure every class contributes at least one
        # example to the test set.
        test_count = max(1, test_count)

        # Select test and training examples.
        test_part = intent_data.iloc[:test_count]
        train_part = intent_data.iloc[test_count:]

        train_parts.append(train_part)
        test_parts.append(test_part)

    # Combine all classes.
    train_data = pd.concat(
        train_parts,
        ignore_index=True
    )

    test_data = pd.concat(
        test_parts,
        ignore_index=True
    )

    # Shuffle the final datasets so that intents are not grouped.
    train_data = train_data.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    test_data = test_data.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    return train_data, test_data


def display_class_distribution(data, title):
    """
    Display the number of examples belonging to each intent.
    """

    print(f"\n{title}")
    print("=" * 60)

    intent_counts = (
        data["intent"]
        .value_counts()
        .sort_index()
    )

    for intent, count in intent_counts.items():
        print(f"{intent:<30} {count}")


def verify_split(train_data, test_data):
    """
    Verify that the training and testing datasets contain
    all expected intents and that no records overlap.
    """

    train_intents = set(train_data["intent"])
    test_intents = set(test_data["intent"])

    if train_intents != test_intents:
        raise ValueError(
            "Training and testing sets do not contain the same intents."
        )

    train_instructions = set(train_data["instruction"])
    test_instructions = set(test_data["instruction"])

    overlap = train_instructions.intersection(
        test_instructions
    )

    if overlap:
        raise ValueError(
            f"Data leakage detected: {len(overlap)} "
            "instructions appear in both training and testing sets."
        )

    print("\nSplit Verification")
    print("=" * 60)
    print("All 27 intents present in both sets: YES")
    print("Training/testing instruction overlap: 0")


def main():
    # ---------------------------------------------------------
    # 1. Load Dataset
    # ---------------------------------------------------------
    data = load_dataset()

    print("Data Preparation")
    print("=" * 60)
    print(f"Loaded records: {len(data)}")

    # ---------------------------------------------------------
    # 2. Remove Exact Duplicates
    # ---------------------------------------------------------
    data = remove_duplicates(data)

    # ---------------------------------------------------------
    # 3. Stratified Train/Test Split
    # ---------------------------------------------------------
    train_data, test_data = stratified_split(data)

    print("\nTrain/Test Split")
    print("=" * 60)
    print(f"Training records: {len(train_data)}")
    print(f"Testing records:  {len(test_data)}")
    print(f"Test percentage:  {TEST_SIZE * 100:.0f}%")

    # ---------------------------------------------------------
    # 4. Display Class Distribution
    # ---------------------------------------------------------
    display_class_distribution(
        train_data,
        "Training Set Intent Distribution"
    )

    display_class_distribution(
        test_data,
        "Testing Set Intent Distribution"
    )

    # ---------------------------------------------------------
    # 5. Verify Split
    # ---------------------------------------------------------
    verify_split(
        train_data,
        test_data
    )


if __name__ == "__main__":
    main()