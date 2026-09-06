from collections import Counter

import pandas as pd

from preprocessing import preprocess_text


class BagOfWordsVectorizer:
    """
    Convert text messages into Bag-of-Words word-count vectors.

    The vocabulary is built only from the training data.
    """

    def __init__(self):
        self.vocabulary = {}
        self.index_to_word = {}

    def fit(self, texts):
        """
        Build the vocabulary from training texts.

        Each unique word receives an integer index.
        """

        vocabulary_words = set()

        for text in texts:
            tokens = preprocess_text(text)

            for token in tokens:
                vocabulary_words.add(token)

        # Sort words so vocabulary creation is deterministic.
        sorted_words = sorted(vocabulary_words)

        self.vocabulary = {
            word: index
            for index, word in enumerate(sorted_words)
        }

        self.index_to_word = {
            index: word
            for word, index in self.vocabulary.items()
        }

        return self

    def transform(self, texts):
        """
        Convert texts into Bag-of-Words count vectors.

        The vocabulary is NOT changed during transformation.
        """

        if not self.vocabulary:
            raise ValueError(
                "Vocabulary is empty. Call fit() before transform()."
            )

        vectors = []

        for text in texts:
            tokens = preprocess_text(text)

            word_counts = Counter(tokens)

            vector = [0] * len(self.vocabulary)

            for word, count in word_counts.items():

                if word in self.vocabulary:
                    index = self.vocabulary[word]
                    vector[index] = count

            vectors.append(vector)

        return vectors

    def fit_transform(self, texts):
        """
        Build the vocabulary and transform the same training texts.
        """

        self.fit(texts)

        return self.transform(texts)

    def get_vocabulary_size(self):
        """
        Return the number of unique words in the vocabulary.
        """

        return len(self.vocabulary)


def main():
    # ---------------------------------------------------------
    # Load the prepared dataset
    # ---------------------------------------------------------
    from data_preparation import (
        load_dataset,
        remove_duplicates,
        stratified_split
    )

    data = load_dataset()

    data = remove_duplicates(data)

    train_data, test_data = stratified_split(data)

    # ---------------------------------------------------------
    # Build vocabulary using TRAINING DATA ONLY
    # ---------------------------------------------------------
    vectorizer = BagOfWordsVectorizer()

    X_train = vectorizer.fit_transform(
        train_data["instruction"]
    )

    # ---------------------------------------------------------
    # Transform testing data using the SAME vocabulary
    # ---------------------------------------------------------
    X_test = vectorizer.transform(
        test_data["instruction"]
    )

    # ---------------------------------------------------------
    # Display Results
    # ---------------------------------------------------------
    print("\nBag-of-Words Vectorization")
    print("=" * 60)

    print(f"Training records: {len(X_train)}")
    print(f"Testing records:  {len(X_test)}")
    print(f"Vocabulary size:  {vectorizer.get_vocabulary_size()}")

    # ---------------------------------------------------------
    # Display a sample vocabulary
    # ---------------------------------------------------------
    print("\nFirst 30 Vocabulary Words")
    print("=" * 60)

    for word, index in list(
        vectorizer.vocabulary.items()
    )[:30]:
        print(f"{index:<5} {word}")

    # ---------------------------------------------------------
    # Display one example vector
    # ---------------------------------------------------------
    print("\nExample")
    print("=" * 60)

    example_text = train_data.iloc[0]["instruction"]

    example_vector = X_train[0]

    print(f"Text: {example_text}")

    print("\nNon-zero features:")

    for index, count in enumerate(example_vector):

        if count > 0:

            word = vectorizer.index_to_word[index]

            print(
                f"{word:<20} {count}"
            )


if __name__ == "__main__":
    main()