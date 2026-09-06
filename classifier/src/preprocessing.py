import re


# ============================================================
# REGULAR EXPRESSIONS
# ============================================================

# Matches dataset placeholders such as:
#
# {{Order Number}}
# {{Person Name}}
# {{Website URL}}
PLACEHOLDER_PATTERN = re.compile(
    r"\{\{(.*?)\}\}"
)


# Matches punctuation characters.
PUNCTUATION_PATTERN = re.compile(
    r"[^\w\s]"
)


# Matches multiple whitespace characters.
WHITESPACE_PATTERN = re.compile(
    r"\s+"
)


# Matches numeric prefixes in alphanumeric tokens.
#
# Examples:
#
# 350dollars          -> dollars
# 1499dollars         -> dollars
# 370795561790current -> current
NUMERIC_PREFIX_PATTERN = re.compile(
    r"^\d+([a-z_]+)$"
)


# ============================================================
# CONTRACTIONS
# ============================================================

CONTRACTIONS = {

    # Do
    "don't": "do_not",
    "doesn't": "does_not",
    "didn't": "did_not",

    # Can / could
    "can't": "can_not",
    "cannot": "can_not",
    "couldn't": "could_not",

    # Will / would
    "won't": "will_not",
    "wouldn't": "would_not",

    # Should
    "shouldn't": "should_not",

    # Be
    "isn't": "is_not",
    "aren't": "are_not",
    "wasn't": "was_not",
    "weren't": "were_not",

    # Have
    "haven't": "have_not",
    "hasn't": "has_not",
    "hadn't": "had_not",

    # Must
    "mustn't": "must_not",

    # I / you / we / they
    "i'm": "i_am",
    "you're": "you_are",
    "we're": "we_are",
    "they're": "they_are",

    "i've": "i_have",
    "you've": "you_have",
    "we've": "we_have",
    "they've": "they_have",

    "i'll": "i_will",
    "you'll": "you_will",
    "we'll": "we_will",
    "they'll": "they_will",

    "i'd": "i_would",
    "you'd": "you_would",
    "we'd": "we_would",
    "they'd": "they_would",
}


# ============================================================
# PLACEHOLDER NORMALIZATION
# ============================================================

def normalize_placeholders(text):
    """
    Convert dataset placeholders into meaningful tokens.

    Examples:

        {{Order Number}}
            ->
        order_number

        {{Person Name}}
            ->
        person_name

        {{Website URL}}
            ->
        website_url
    """

    def replace_placeholder(match):

        content = match.group(1)

        content = content.strip()
        content = content.lower()

        # Convert spaces inside placeholders to underscores.
        content = re.sub(
            r"\s+",
            "_",
            content
        )

        return content

    return PLACEHOLDER_PATTERN.sub(
        replace_placeholder,
        text
    )


# ============================================================
# CONTRACTION + NEGATION NORMALIZATION
# ============================================================

def normalize_contractions(text):
    """
    Normalize English contractions and explicit negation.

    Examples:

        don't       -> do_not
        do not      -> do_not

        can't       -> can_not
        cannot      -> can_not
        can not     -> can_not

        won't       -> will_not
        will not    -> will_not

        haven't     -> have_not
        have not    -> have_not

    Keeping negation together is important for intent
    classification.
    """

    text = text.lower()

    # --------------------------------------------------------
    # 1. Normalize contractions
    # --------------------------------------------------------

    for contraction, replacement in CONTRACTIONS.items():

        pattern = rf"\b{re.escape(contraction)}\b"

        text = re.sub(
            pattern,
            replacement,
            text
        )

    # --------------------------------------------------------
    # 2. Normalize explicit negation
    # --------------------------------------------------------

    negation_patterns = {

        r"\bdo\s+not\b": "do_not",

        r"\bdoes\s+not\b": "does_not",

        r"\bdid\s+not\b": "did_not",

        r"\bcan\s+not\b": "can_not",

        r"\bcould\s+not\b": "could_not",

        r"\bwill\s+not\b": "will_not",

        r"\bwould\s+not\b": "would_not",

        r"\bshould\s+not\b": "should_not",

        r"\bis\s+not\b": "is_not",

        r"\bare\s+not\b": "are_not",

        r"\bwas\s+not\b": "was_not",

        r"\bwere\s+not\b": "were_not",

        r"\bhave\s+not\b": "have_not",

        r"\bhas\s+not\b": "has_not",

        r"\bhad\s+not\b": "had_not",

        r"\bmust\s+not\b": "must_not",
    }

    for pattern, replacement in negation_patterns.items():

        text = re.sub(
            pattern,
            replacement,
            text
        )

    return text


# ============================================================
# NUMERIC TOKEN NORMALIZATION
# ============================================================

def normalize_numeric_token(token):
    """
    Remove meaningless numeric information.

    Examples:

        350dollars
            ->
        dollars

        1499dollars
            ->
        dollars

        370795561790current
            ->
        current

        12588
            ->
        None

    Pure numbers are normally customer-specific identifiers,
    such as:

        order numbers
        invoice numbers
        account numbers
    """

    # Remove pure numbers.
    if token.isdigit():
        return None

    # Remove numeric prefixes from alphanumeric tokens.
    match = NUMERIC_PREFIX_PATTERN.match(token)

    if match:
        return match.group(1)

    return token


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize raw customer text.

    Processing order:

        1. Convert input to string.
        2. Normalize placeholders.
        3. Normalize contractions.
        4. Convert to lowercase.
        5. Remove punctuation.
        6. Normalize whitespace.
    """

    text = str(text)

    # --------------------------------------------------------
    # 1. Normalize placeholders
    # --------------------------------------------------------

    # This must happen before punctuation removal because
    # placeholders contain { } characters.
    text = normalize_placeholders(text)

    # --------------------------------------------------------
    # 2. Normalize contractions and negation
    # --------------------------------------------------------

    text = normalize_contractions(text)

    # --------------------------------------------------------
    # 3. Lowercase
    # --------------------------------------------------------

    text = text.lower()

    # --------------------------------------------------------
    # 4. Remove punctuation
    # --------------------------------------------------------

    # Replace punctuation with spaces instead of simply
    # deleting it. This prevents words from being joined.
    text = PUNCTUATION_PATTERN.sub(
        " ",
        text
    )

    # --------------------------------------------------------
    # 5. Normalize whitespace
    # --------------------------------------------------------

    text = WHITESPACE_PATTERN.sub(
        " ",
        text
    )

    return text.strip()


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text):
    """
    Convert normalized text into individual tokens.

    Tokens may contain:

        letters
        numbers
        underscores
    """

    normalized_text = normalize_text(text)

    return re.findall(
        r"\b[a-z0-9_]+\b",
        normalized_text
    )


# ============================================================
# COMPLETE PREPROCESSING PIPELINE
# ============================================================

def preprocess_text(text):
    """
    Complete preprocessing pipeline.

    Steps:

        1. Normalize text.
        2. Tokenize.
        3. Remove pure numeric identifiers.
        4. Remove numeric prefixes from alphanumeric tokens.

    Returns:

        list[str]
    """

    tokens = tokenize(text)

    cleaned_tokens = []

    for token in tokens:

        normalized_token = normalize_numeric_token(
            token
        )

        if normalized_token is not None:

            cleaned_tokens.append(
                normalized_token
            )

    return cleaned_tokens


# ============================================================
# PREPROCESSING TESTS
# ============================================================

if __name__ == "__main__":

    examples = [

        # ----------------------------------------------------
        # Placeholders
        # ----------------------------------------------------

        "I need help cancelling my purchase {{Order Number}}!",

        "Can you help me with {{Person Name}}?",


        # ----------------------------------------------------
        # Normal messages
        # ----------------------------------------------------

        "I forgot my password.",

        "I want to check invoice #12588.",

        "Can you help me with order #85632?",


        # ----------------------------------------------------
        # Numeric normalization
        # ----------------------------------------------------

        "The item costs 350dollars.",

        "The item costs 1499dollars.",

        "My account number is 370795561790current.",


        # ----------------------------------------------------
        # Normal sentence
        # ----------------------------------------------------

        "I need assistance changing to the premium account.",


        # ----------------------------------------------------
        # Negation / contractions
        # ----------------------------------------------------

        "I don't want to cancel my order.",

        "I do not want to cancel my order.",

        "I can't make a payment.",

        "I cannot make a payment.",

        "I can not make a payment.",

        "I won't place the order.",

        "I will not place the order.",

        "I haven't received my order.",

        "I have not received my order.",


        # ----------------------------------------------------
        # Order examples
        # ----------------------------------------------------

        "Please stop my order!",

        "I no longer need this order.",

        "I changed my mind about the order.",

        "I want to cancel my order.",

        "I want to place an order.",

        "I want to change my existing order.",


        # ----------------------------------------------------
        # Short query
        # ----------------------------------------------------

        "Where is my order?!",


        # ----------------------------------------------------
        # Whitespace
        # ----------------------------------------------------

        "I    want     to    cancel    my    order.",
    ]


    print("\nPREPROCESSING TESTS")
    print("=" * 70)

    for example in examples:

        print("\nOriginal:")
        print(example)

        print("\nNormalized:")
        print(normalize_text(example))

        print("\nTokens:")
        print(preprocess_text(example))

        print("-" * 70)