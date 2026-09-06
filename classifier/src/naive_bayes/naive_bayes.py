import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

try:
    from .constants import DEFAULT_STOPWORDS
    from .features import FeatureMixin
    from .training import TrainingMixin
    from .prediction import PredictionMixin
except ImportError:
    from constants import DEFAULT_STOPWORDS
    from features import FeatureMixin
    from training import TrainingMixin
    from prediction import PredictionMixin


class MultinomialNaiveBayes(FeatureMixin, TrainingMixin, PredictionMixin):
    """
    Multinomial Naive Bayes classifier implemented from scratch
    using NumPy.

    The classifier uses:

    - Bag-of-Words features
    - Unigrams and bigrams
    - Class prior probabilities
    - Laplace smoothing
    - Log probabilities
    - NumPy vectorized calculations
    """

    def __init__(
        self,
        alpha=1.0,
        min_df=1,
        max_df_ratio=1.0,
        ngram_range=(1, 2),
        stopwords=None
    ):
        """
        Parameters
        ----------
        alpha : float
            Laplace smoothing parameter.

        min_df : int
            Minimum number of documents a feature must appear in.

        max_df_ratio : float
            Maximum fraction of documents a feature can appear in.

        ngram_range : tuple
            Range of n-grams to generate.

        stopwords : set or None
            Optional set of words to remove.
        """

        if alpha <= 0:
            raise ValueError(
                "alpha must be greater than zero."
            )

        self.alpha = alpha
        self.min_df = min_df
        self.max_df_ratio = max_df_ratio
        self.ngram_range = ngram_range

        self.stopwords = (
            set(stopwords)
            if stopwords is not None
            else set(DEFAULT_STOPWORDS)
        )

        # -----------------------------------------------------
        # Vocabulary
        # -----------------------------------------------------

        self.vocabulary = {}

        self.index_to_feature = {}

        self.vocabulary_size = 0

        # -----------------------------------------------------
        # Classes
        # -----------------------------------------------------

        self.classes = []

        self.class_to_index = {}

        # -----------------------------------------------------
        # Counts
        # -----------------------------------------------------

        self.class_counts = None

        self.feature_counts = None

        self.total_words = None

        # -----------------------------------------------------
        # Probabilities
        # -----------------------------------------------------

        self.class_log_prior = None

        self.feature_log_probability = None

        # -----------------------------------------------------
        # Training state
        # -----------------------------------------------------

        self.total_documents = 0

        self.is_fitted = False