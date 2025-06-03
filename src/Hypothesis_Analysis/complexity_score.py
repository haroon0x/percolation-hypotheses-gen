import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import textstat 
import numpy as np 

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def calculate_complexity_score(hypothesis_text: str) -> float:
    """
    Calculate the complexity score of a given text based on a weighted combination of normalized linguistic features.

    Parameters:
    hypothesis_text (str): The text to analyze.

    Returns:
    float: The calculated complexity score scaled to be roughly between 0 and 100.
    """
    if not hypothesis_text or not hypothesis_text.strip():
        return 0.0

    # 1. Basic tokenization
    try:
        words = word_tokenize(hypothesis_text.lower()) # Lowercase for consistency in lexical measures
        num_words = len(words)
        sentences = sent_tokenize(hypothesis_text)
        num_sentences = len(sentences)
    except Exception as e:
        # Handle potential errors during tokenization if text is very unusual
        print(f"Tokenization error: {e}")
        return 0.0 # Or a default low complexity score

    if num_words == 0 or num_sentences == 0:
        return 0.0

    # --- Feature Extraction ---

    # Feature 1: Average Sentence Length (ASL)
    avg_sent_length = num_words / num_sentences
    # Define expected min/max for normalization (these are heuristics, can be tuned)
    MIN_ASL = 5.0   # Very simple sentences
    MAX_ASL = 36.0  # Very long, complex sentences for a hypothesis

    # Feature 2: Flesch-Kincaid Grade Level (FK_GRADE)
    # Higher grade level indicates more complexity.
    fk_grade = textstat.flesch_kincaid_grade(hypothesis_text)
    MIN_FK_GRADE = 0.0  # Early elementary
    MAX_FK_GRADE = 20.0 # Post-graduate level

    # Feature 3: Lexical Diversity (MTLD or MATTR)
    # MTLD (Measure of Textual Lexical Diversity) is robust.
    # For very short texts, MATTR might return None or raise error, handle this.
    try:
        # window_size can be adjusted. Smaller windows are more sensitive to local variations.
        lex_diversity_score = textstat.mattr(hypothesis_text, window_size=25)
        if lex_diversity_score is None:
            lex_diversity_score = 0.3 # Assign a low-ish diversity
    except: 
        lex_diversity_score = 0.3

    MIN_LEX_DIV = 0.2  # Low diversity
    MAX_LEX_DIV = 0.9  # High diversity 

    # Feature 4: Average Syllables per Word (ASW) - Word Complexity
    total_syllables = textstat.syllable_count(hypothesis_text)
    avg_syllables_per_word = total_syllables / num_words if num_words > 0 else 0
    MIN_ASW = 1.0  # e.g., "a", "is", "to"
    MAX_ASW = 2.5  # Average for highly technical/polysyllabic text

    # --- Normalization (Min-Max to 0-1 scale) ---
    def normalize(value, min_val, max_val):
        if max_val == min_val: # Avoid division by zero
            return 0.5 # Or 0 or 1 depending on desired behavior for flat range
        # Clip to ensure value is within [min_val, max_val] before normalization
        # then clip result to [0, 1] to handle values outside expected range gracefully.
        clamped_value = np.clip(value, min_val, max_val)
        return (clamped_value - min_val) / (max_val - min_val)

    norm_asl = normalize(avg_sent_length, MIN_ASL, MAX_ASL)
    norm_fk_grade = normalize(fk_grade, MIN_FK_GRADE, MAX_FK_GRADE)
    norm_lex_div = normalize(lex_diversity_score, MIN_LEX_DIV, MAX_LEX_DIV)
    norm_asw = normalize(avg_syllables_per_word, MIN_ASW, MAX_ASW)

    # --- Weighted Combination ---
    # These weights are crucial and represent the perceived importance of each feature.
    # They should ideally sum to 1 if the desired output before final scaling is 0-1.
    # Tune these based on empirical results and desired emphasis.
    weights = {
        'asl': 0.25,         # Sentence length
        'fk_grade': 0.30,    # Readability/Grade level
        'lex_div': 0.25,     # Vocabulary richness
        'asw': 0.20          # Word complexity
    }

    complexity_score_0_1 = (
        norm_asl * weights['asl'] +
        norm_fk_grade * weights['fk_grade'] +
        norm_lex_div * weights['lex_div'] +
        norm_asw * weights['asw']
    )


    final_complexity_score = complexity_score_0_1 * 100
    return round(final_complexity_score, 2)