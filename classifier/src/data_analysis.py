import re

import pandas as pd


from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)


def tokenize(text):
    """
    Convert a customer message into lowercase word tokens.

    This function is only used for dataset analysis.
    The final preprocessing implementation will be created separately.
    """
    return re.findall(r"\b\w+\b", text.lower())


def main():
    # ---------------------------------------------------------
    # Load Dataset
    # ---------------------------------------------------------
    data = pd.read_csv(DATASET_PATH)

    print("Dataset Analysis")
    print("=" * 60)

    print(f"Total records: {len(data)}")
    print(f"Number of columns: {len(data.columns)}")
    print(f"Number of intents: {data['intent'].nunique()}")

    print("\nColumns:")
    for column in data.columns:
        print(f"- {column}")

    # ---------------------------------------------------------
    # Intent Distribution
    # ---------------------------------------------------------
    print("\nIntent Distribution")
    print("=" * 60)

    intent_counts = data["intent"].value_counts().sort_index()

    for intent, count in intent_counts.items():
        print(f"{intent:<30} {count}")

    # ---------------------------------------------------------
    # Missing Values
    # ---------------------------------------------------------
    print("\nMissing Values")
    print("=" * 60)

    missing_values = data[["instruction", "intent"]].isnull().sum()
    print(missing_values)

    # ---------------------------------------------------------
    # Sample Customer Requests
    # ---------------------------------------------------------
    print("\nSample Customer Requests")
    print("=" * 60)

    samples = data[["instruction", "intent"]].sample(
        n=10,
        random_state=42
    )

    for _, row in samples.iterrows():
        print(f"\nCustomer: {row['instruction']}")
        print(f"Intent:   {row['intent']}")

    # ---------------------------------------------------------
    # Text Length Analysis
    # ---------------------------------------------------------
    data["word_count"] = data["instruction"].apply(
        lambda text: len(tokenize(text))
    )

    data["character_count"] = data["instruction"].str.len()

    print("\nText Length Analysis")
    print("=" * 60)

    print(f"Average words per message: {data['word_count'].mean():.2f}")
    print(f"Minimum words in a message: {data['word_count'].min()}")
    print(f"Maximum words in a message: {data['word_count'].max()}")

    print(
        f"\nAverage characters per message: "
        f"{data['character_count'].mean():.2f}"
    )

    print(
        f"Minimum characters in a message: "
        f"{data['character_count'].min()}"
    )

    print(
        f"Maximum characters in a message: "
        f"{data['character_count'].max()}"
    )

    # ---------------------------------------------------------
    # Vocabulary Analysis
    # ---------------------------------------------------------
    vocabulary = set()

    for text in data["instruction"]:
        vocabulary.update(tokenize(text))

    print("\nVocabulary Analysis")
    print("=" * 60)

    print(f"Unique words: {len(vocabulary)}")

    # ---------------------------------------------------------
    # Placeholder Analysis
    # ---------------------------------------------------------
    placeholder_pattern = r"\{\{.*?\}\}"

    placeholder_count = data["instruction"].str.count(
        placeholder_pattern
    ).sum()

    messages_with_placeholders = data["instruction"].str.contains(
        placeholder_pattern,
        regex=True
    ).sum()

    print("\nPlaceholder Analysis")
    print("=" * 60)

    print(
        f"Messages containing placeholders: "
        f"{messages_with_placeholders}"
    )

    print(
        f"Total placeholder occurrences: "
        f"{placeholder_count}"
    )

    # ---------------------------------------------------------
    # Duplicate Analysis
    # ---------------------------------------------------------
    duplicate_count = data["instruction"].duplicated().sum()

    print("\nDuplicate Analysis")
    print("=" * 60)

    print(f"Duplicate customer messages: {duplicate_count}")

    # ---------------------------------------------------------
    # Duplicate Instructions With Multiple Intents
    # ---------------------------------------------------------
    intent_counts_per_instruction = (
        data.groupby("instruction")["intent"]
        .nunique()
    )

    ambiguous_duplicates = (
        intent_counts_per_instruction[
            intent_counts_per_instruction > 1
        ]
    )

    print("\nAmbiguous Duplicate Analysis")
    print("=" * 60)

    print(
        f"Instructions assigned to multiple intents: "
        f"{len(ambiguous_duplicates)}"
    )

    if len(ambiguous_duplicates) > 0:
        print("\nExamples:")

        for instruction in ambiguous_duplicates.index[:10]:
            matching_rows = data[
                data["instruction"] == instruction
            ][["instruction", "intent"]]

            print()
            for _, row in matching_rows.iterrows():
                print(f"Customer: {row['instruction']}")
                print(f"Intent:   {row['intent']}")

    # ---------------------------------------------------------
    # Most Common Words
    # ---------------------------------------------------------
    word_series = pd.Series(
        word
        for text in data["instruction"]
        for word in tokenize(text)
    )

    word_counts = word_series.value_counts()

    print("\nMost Common Words")
    print("=" * 60)

    for word, count in word_counts.head(20).items():
        print(f"{word:<20} {count}")


if __name__ == "__main__":
    main()