import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import textstat 
import numpy as np 

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

def calculate_complexity_score(hypothesis_text: str) -> float:
    """
    Calculate the complexity score of a given text based on a weighted combination of normalized linguistic features.

    Parameters:
    hypothesis_text (str): The text to analyze.

    Returns:
    float: The calculated complexity score scaled to be roughly between 0 and 100.
    """
    if not hypothesis_text:
        return False

    try:
        words = word_tokenize(hypothesis_text.lower())
        num_words = len(words)
        sentences = sent_tokenize(hypothesis_text)
        num_sentences = len(sentences)
    except Exception as e:
        print(f"Tokenization error: {e}")
        return 0.0 # Or a default low complexity score
    if num_sentences == 0:
        num_sentences = 1  
    
    if num_words == 0 or num_sentences == 0:
        return 0.0

    avg_sent_length = num_words / num_sentences
    MIN_ASL = 5.0   
    MAX_ASL = 36.0  

    try:
        fk_grade = textstat.flesch_kincaid_grade(hypothesis_text)
    except:
        fk_grade = 5.0
    MIN_FK_GRADE = 0.0  
    MAX_FK_GRADE = 20.0 


    try:
        lex_diversity_score = textstat.mattr(hypothesis_text, window_size=25)
        if lex_diversity_score is None:
            lex_diversity_score = 0.3
    except: 
        lex_diversity_score = 0.3

    MIN_LEX_DIV = 0.2  # Low diversity
    MAX_LEX_DIV = 0.9  # High diversity 

   
    try:
        total_syllables = textstat.syllable_count(hypothesis_text)
    except:
        total_syllables = estimate_syllables_accurate(hypothesis_text, num_words)

    avg_syllables_per_word = total_syllables / num_words if num_words > 0 else 0
    MIN_ASW = 1.0  
    MAX_ASW = 2.5  

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


def estimate_syllables_accurate(text, num_words):
    """More accurate syllable estimation based on vowel patterns"""
    vowels = 'aeiouyAEIOUY'
    syllable_count = 0
    
    for word in word_tokenize(text.lower()):
        word_syllables = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                word_syllables += 1
            prev_was_vowel = is_vowel
        
        # Handle silent 'e' and minimum syllable rules
        if word.endswith('e') and word_syllables > 1:
            word_syllables -= 1
        if word_syllables == 0:
            word_syllables = 1
            
        syllable_count += word_syllables
    
    return syllable_count