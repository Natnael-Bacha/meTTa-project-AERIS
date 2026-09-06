from preprocessing import preprocess_text


class FeatureMixin:
    """
    Provides feature generation and document frequency
    utilities for MultinomialNaiveBayes.
    """

    # =========================================================
    # FEATURE GENERATION
    # =========================================================

    def _generate_features(self, text):
        """
        Convert text into unigram and bigram features.

        Example:

            "change shipping address"

        becomes:

            change
            shipping
            address
            change_shipping
            shipping_address
        """

        tokens = preprocess_text(text)

        tokens = [
            token
            for token in tokens
            if token not in self.stopwords
        ]

        min_n, max_n = self.ngram_range

        features = []

        for n in range(min_n, max_n + 1):

            if n == 1:

                features.extend(tokens)

            else:

                for i in range(
                    len(tokens) - n + 1
                ):

                    features.append(
                        "_".join(
                            tokens[i:i + n]
                        )
                    )

        return features

    # =========================================================
    # DOCUMENT FREQUENCY
    # =========================================================

    def _document_frequency(self, documents):
        """
        Count how many documents contain each feature.
        """

        frequencies = {}

        for features in documents:

            for feature in set(features):

                frequencies[feature] = (
                    frequencies.get(feature, 0) + 1
                )

        return frequencies