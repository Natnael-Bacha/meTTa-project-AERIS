import numpy as np


class PredictionMixin:
    """
    Provides prediction-related methods for MultinomialNaiveBayes.
    """

    # =========================================================
    # TEXT → COUNT VECTOR
    # =========================================================

    def _vectorize(self, text):
        """
        Convert one text into a Bag-of-Words count vector.
        """

        if not self.is_fitted:

            raise ValueError(
                "The classifier must be fitted before prediction."
            )

        vector = np.zeros(
            self.vocabulary_size,
            dtype=np.int64
        )

        features = self._generate_features(text)

        for feature in features:

            if feature not in self.vocabulary:
                continue

            index = self.vocabulary[feature]

            vector[index] += 1

        return vector

    # =========================================================
    # PREDICT ONE
    # =========================================================

    def predict_one(self, text):
        """
        Predict the intent of one text.
        """

        log_scores = self._calculate_log_scores(
            text
        )

        best_index = np.argmax(
            log_scores
        )

        return self.classes[best_index]

    # =========================================================
    # PREDICT MANY
    # =========================================================

    def predict(self, texts):
        """
        Predict the intents of multiple texts.
        """

        return [
            self.predict_one(text)
            for text in texts
        ]

    # =========================================================
    # LOG SCORES
    # =========================================================

    def _calculate_log_scores(self, text):
        """
        Calculate the Naive Bayes log score for every class.

        Formula:

            log P(class)
            +
            Σ count(feature)
              × log P(feature | class)
        """

        vector = self._vectorize(text)

        scores = (
            self.class_log_prior
            +
            np.sum(
                self.feature_log_probability
                *
                vector[np.newaxis, :],
                axis=1
            )
        )

        return scores

    # =========================================================
    # PREDICT PROBABILITIES
    # =========================================================

    def predict_proba(self, texts):
        """
        Return probability distributions for each text.
        """

        results = []

        for text in texts:

            log_scores = (
                self._calculate_log_scores(text)
            )

            # Log-sum-exp stabilization
            maximum = np.max(
                log_scores
            )

            exponentials = np.exp(
                log_scores - maximum
            )

            probabilities = (
                exponentials
                /
                np.sum(exponentials)
            )

            results.append(
                {
                    self.classes[index]: float(
                        probabilities[index]
                    )
                    for index in range(
                        len(self.classes)
                    )
                }
            )

        return results