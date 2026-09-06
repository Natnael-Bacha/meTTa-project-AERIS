import numpy as np


class TrainingMixin:
    """
    Provides the fit() method for MultinomialNaiveBayes.
    """

    # =========================================================
    # TRAIN
    # =========================================================

    def fit(self, texts, labels):
        """
        Train the Multinomial Naive Bayes classifier.

        Calculates:

            P(class)

        and:

            P(feature | class)

        using Laplace smoothing.
        """

        if len(texts) != len(labels):

            raise ValueError(
                "texts and labels must contain "
                "the same number of examples."
            )

        if len(texts) == 0:

            raise ValueError(
                "Training data cannot be empty."
            )

        # -----------------------------------------------------
        # Reset model
        # -----------------------------------------------------

        self.is_fitted = False

        self.total_documents = len(texts)

        # -----------------------------------------------------
        # Find classes
        # -----------------------------------------------------

        self.classes = sorted(
            set(labels)
        )

        self.class_to_index = {
            label: index
            for index, label in enumerate(self.classes)
        }

        number_of_classes = len(self.classes)

        # -----------------------------------------------------
        # Convert documents into features
        # -----------------------------------------------------

        documents = [
            self._generate_features(text)
            for text in texts
        ]

        # -----------------------------------------------------
        # Build vocabulary
        # -----------------------------------------------------

        document_frequency = (
            self._document_frequency(documents)
        )

        max_df_count = (
            self.max_df_ratio
            * self.total_documents
        )

        vocabulary = [
            feature
            for feature, frequency
            in document_frequency.items()
            if (
                frequency >= self.min_df
                and
                frequency <= max_df_count
            )
        ]

        vocabulary.sort()

        self.vocabulary = {
            feature: index
            for index, feature
            in enumerate(vocabulary)
        }

        self.index_to_feature = {
            index: feature
            for feature, index
            in self.vocabulary.items()
        }

        self.vocabulary_size = len(
            self.vocabulary
        )

        # -----------------------------------------------------
        # Initialize NumPy arrays
        # -----------------------------------------------------

        self.class_counts = np.zeros(
            number_of_classes,
            dtype=np.int64
        )

        self.feature_counts = np.zeros(
            (
                number_of_classes,
                self.vocabulary_size
            ),
            dtype=np.int64
        )

        # -----------------------------------------------------
        # Count classes and features
        # -----------------------------------------------------

        for features, label in zip(
            documents,
            labels
        ):

            class_index = (
                self.class_to_index[label]
            )

            self.class_counts[class_index] += 1

            for feature in features:

                if feature not in self.vocabulary:
                    continue

                feature_index = (
                    self.vocabulary[feature]
                )

                self.feature_counts[
                    class_index,
                    feature_index
                ] += 1

        # -----------------------------------------------------
        # Total words per class
        # -----------------------------------------------------

        self.total_words = np.sum(
            self.feature_counts,
            axis=1
        )

        # =====================================================
        # PRIOR PROBABILITY
        # =====================================================

        class_probabilities = (
            self.class_counts
            /
            self.total_documents
        )

        self.class_log_prior = np.log(
            class_probabilities
        )

        # =====================================================
        # LIKELIHOOD WITH LAPLACE SMOOTHING
        # =====================================================

        # P(feature | class)
        #
        # = (feature_count + alpha)
        #   /
        #   (total_words + alpha * vocabulary_size)

        numerator = (
            self.feature_counts
            + self.alpha
        )

        denominator = (
            self.total_words[:, np.newaxis]
            +
            self.alpha * self.vocabulary_size
        )

        feature_probabilities = (
            numerator
            /
            denominator
        )

        # -----------------------------------------------------
        # Store logarithmic probabilities
        # -----------------------------------------------------

        self.feature_log_probability = np.log(
            feature_probabilities
        )

        self.is_fitted = True

        return self