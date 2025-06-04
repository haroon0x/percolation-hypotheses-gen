import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

@dataclass
class HypothesisEvaluation:
    """Container for multi-dimensional hypothesis evaluation"""
    #hypothesis: str
    specificity_score: float
    falsifiability_score: float
    conceptual_density: float
    empirical_grounding: float
    predictive_content: float
    overall_quality: float
    detailed_metrics: Dict

class ScientificHypothesisEvaluator:
    """
    Multi-dimensional evaluator for scientific hypotheses
    Based on philosophy of science principles
    """
    
    def __init__(self):
        self.domain_terms = set() 
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
    def load_domain_knowledge(self, literature_texts: List[str]):
        """Extract domain-specific knowledge from literature"""

        for text in literature_texts:
            technical_terms = re.findall(r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b', text)
            technical_terms.extend(re.findall(r'\b\w*(?:tion|sion|ment|ity|ness|ism)\b', text.lower()))
            self.domain_terms.update(term.lower() for term in technical_terms)
        if literature_texts:
            self.tfidf_vectorizer.fit(literature_texts)
    
    def evaluate_specificity(self, hypothesis: str) -> Tuple[float, Dict]:
        """
        Evaluate how specific vs. general the hypothesis is
        More specific = higher information density
        """
        metrics = {}
        quantitative_terms = [
            'increase', 'decrease', 'correlation', 'association', 'percentage',
            'significantly', 'proportional', 'linear', 'exponential', 'ratio'
        ]
        metrics['quantitative_terms'] = sum(1 for term in quantitative_terms 
                                          if term in hypothesis.lower())
        measurement_patterns = [
            r'\d+\s*(?:%|percent|fold|times|standard deviation)',
            r'(?:p\s*<|p\s*=|r\s*=|α\s*=)',
            r'(?:compared to|relative to|versus|vs\.)',
        ]
        metrics['measurement_references'] = sum(len(re.findall(pattern, hypothesis, re.IGNORECASE)) 
                                             for pattern in measurement_patterns)
        
        hypothesis_words = set(hypothesis.lower().split())
        metrics['domain_terms'] = len(hypothesis_words.intersection(self.domain_terms))

        word_count = len(hypothesis.split())
        specificity = (
            (metrics['quantitative_terms'] / word_count) * 0.4 +
            (metrics['measurement_references'] / word_count) * 0.4 +
            (metrics['domain_terms'] / word_count) * 0.2
        ) * 10  
        
        return min(specificity, 1), metrics
    
    def evaluate_falsifiability(self, hypothesis: str) -> Tuple[float, Dict]:
        """
        Evaluate how testable/falsifiable the hypothesis is
        Karl Popper's criterion for scientific hypotheses
        """
        metrics = {}
        
        prediction_terms = [
            'will', 'should', 'predicts', 'expects', 'anticipates',
            'results in', 'leads to', 'causes', 'produces'
        ]
        metrics['prediction_terms'] = sum(1 for term in prediction_terms 
                                        if term in hypothesis.lower())
        
        observable_terms = [
            'measured', 'observed', 'detected', 'recorded', 'monitored',
            'behavior', 'response', 'change', 'difference', 'effect'
        ]
        metrics['observable_terms'] = sum(1 for term in observable_terms 
                                        if term in hypothesis.lower())

        conditional_patterns = [
            r'if\s+.+\s+then',
            r'when\s+.+\s+(?:will|should|would)',
            r'given\s+.+\s+(?:expect|predict)',
        ]
        metrics['conditional_statements'] = sum(len(re.findall(pattern, hypothesis, re.IGNORECASE)) 
                                              for pattern in conditional_patterns)
        
        unfalsifiable_terms = ['always', 'never', 'all', 'none', 'impossible', 'certain']
        metrics['unfalsifiable_terms'] = sum(1 for term in unfalsifiable_terms 
                                           if term in hypothesis.lower())
    
        word_count = len(hypothesis.split())
        falsifiability = (
            (metrics['prediction_terms'] / word_count) * 0.3 +
            (metrics['observable_terms'] / word_count) * 0.3 +
            (metrics['conditional_statements'] / word_count) * 0.3 +
            (1 - metrics['unfalsifiable_terms'] / word_count) * 0.1
        ) * 5  
        
        return min(falsifiability, 1.0), metrics
    
    def evaluate_conceptual_density(self, hypothesis: str) -> Tuple[float, Dict]:
        """
        Evaluate the density of scientific concepts
        High concept density = more information per unit text
        """
        metrics = {}
        
        hypothesis_words = hypothesis.lower().split()
        metrics['unique_domain_terms'] = len(set(hypothesis_words).intersection(self.domain_terms))
        metrics['total_words'] = len(hypothesis_words)
        metrics['unique_words'] = len(set(hypothesis_words))
        
        relationship_terms = [
            'relationship', 'association', 'correlation', 'interaction',
            'modulates', 'regulates', 'influences', 'affects', 'mediates'
        ]
        metrics['relationship_terms'] = sum(1 for term in relationship_terms 
                                          if term in hypothesis.lower())
        
        if metrics['total_words'] > 0:
            density = (
                (metrics['unique_domain_terms'] / metrics['total_words']) * 0.5 +
                (metrics['unique_words'] / metrics['total_words']) * 0.3 +
                (metrics['relationship_terms'] / metrics['total_words']) * 0.2
            )
        else:
            density = 0.0
            
        return density, metrics
    
    def evaluate_empirical_grounding(self, hypothesis: str, literature_texts: List[str] = None) -> Tuple[float, Dict]:
        """
        Evaluate how well grounded the hypothesis is in existing empirical work
        """
        metrics = {}
        
        if not literature_texts:
            return 0.0, {'error': 'No literature provided'}
        try:
            literature_combined = ' '.join(literature_texts)
            tfidf_matrix = self.tfidf_vectorizer.transform([hypothesis, literature_combined])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            metrics['semantic_similarity'] = similarity
        except:
            metrics['semantic_similarity'] = 0.0
        
        evidence_terms = [
            'study', 'research', 'experiment', 'data', 'evidence',
            'findings', 'results', 'analysis', 'investigation'
        ]
        metrics['evidence_terms'] = sum(1 for term in evidence_terms 
                                      if term in hypothesis.lower())
    
        citation_patterns = [
            r'\([^)]*\d{4}[^)]*\)',  # (Author, 2023)
            r'according to',
            r'previous research',
            r'studies show'
        ]
        metrics['citation_indicators'] = sum(len(re.findall(pattern, hypothesis, re.IGNORECASE)) 
                                           for pattern in citation_patterns)
        
        word_count = len(hypothesis.split())
        grounding = (
            metrics['semantic_similarity'] * 0.6 +
            (metrics['evidence_terms'] / word_count) * 0.3 +
            (metrics['citation_indicators'] / word_count) * 0.1
        )
        
        return min(grounding, 1.0), metrics
    
    def evaluate_predictive_content(self, hypothesis: str) -> Tuple[float, Dict]:
        """
        Evaluate the predictive content of the hypothesis
        """
        metrics = {}
        
        predictive_terms = [
            'will', 'would', 'should', 'expect', 'predict', 'anticipate',
            'likely', 'probable', 'potential', 'may', 'might'
        ]
        metrics['predictive_terms'] = sum(1 for term in predictive_terms 
                                        if term in hypothesis.lower())
        
      
        causal_terms = [
            'because', 'due to', 'caused by', 'results from', 'leads to',
            'produces', 'generates', 'triggers', 'induces'
        ]
        metrics['causal_terms'] = sum(1 for term in causal_terms 
                                    if term in hypothesis.lower())
        
 
        mechanism_terms = [
            'mechanism', 'process', 'pathway', 'mediates', 'through',
            'via', 'by means of', 'operates', 'functions'
        ]
        metrics['mechanism_terms'] = sum(1 for term in mechanism_terms 
                                       if term in hypothesis.lower())
   
        word_count = len(hypothesis.split())
        predictive_content = (
            (metrics['predictive_terms'] / word_count) * 0.4 +
            (metrics['causal_terms'] / word_count) * 0.4 +
            (metrics['mechanism_terms'] / word_count) * 0.2
        ) * 5  
        
        return min(predictive_content, 1.0), metrics
    
    def evaluate_hypothesis(self, hypothesis: str, literature_texts: List[str] = None) -> HypothesisEvaluation:
        """
        Comprehensive evaluation of scientific hypothesis
        """
        if literature_texts:
            self.load_domain_knowledge(literature_texts)
        
        
        specificity, spec_metrics = self.evaluate_specificity(hypothesis)
        falsifiability, fals_metrics = self.evaluate_falsifiability(hypothesis)
        conceptual_density, conc_metrics = self.evaluate_conceptual_density(hypothesis)
        empirical_grounding, emp_metrics = self.evaluate_empirical_grounding(hypothesis, literature_texts)
        predictive_content, pred_metrics = self.evaluate_predictive_content(hypothesis)
        
        overall_quality = (
            specificity * 0.2 +
            falsifiability * 0.25 +
            conceptual_density * 0.2 +
            empirical_grounding * 0.2 +
            predictive_content * 0.15
        )
        
        detailed_metrics = {
            'specificity': spec_metrics,
            'falsifiability': fals_metrics,
            'conceptual_density': conc_metrics,
            'empirical_grounding': emp_metrics,
            'predictive_content': pred_metrics
        }
        
        return HypothesisEvaluation(
            specificity_score=specificity,
            falsifiability_score=falsifiability,
            conceptual_density=conceptual_density,
            empirical_grounding=empirical_grounding,
            predictive_content=predictive_content,
            overall_quality=overall_quality,
            detailed_metrics=detailed_metrics
        )


class InformationTheoreticAnalyzer:
    """
    Measures information content using information theory principles
    """
    
    def __init__(self, corpus_texts: List[str] = None):
        """
        Initialize with background corpus for probability estimation
        """
        self.corpus_vocab = Counter()
        self.total_words = 0
        
        if corpus_texts:
            self.build_corpus_model(corpus_texts)
    
    def build_corpus_model(self, texts: List[str]):
        """Build word frequency model from corpus"""
        for text in texts:
            words = [w.lower() for w in word_tokenize(text) if w.isalpha()]
            self.corpus_vocab.update(words)
            self.total_words += len(words)
    
    def calculate_surprisal(self, word: str) -> float:
        """
        Calculate surprisal (self-information) of a word
        Surprisal = -log2(P(word))
        """
        if self.total_words == 0:
            return 0.0
            
        word = word.lower()
        word_freq = self.corpus_vocab.get(word, 1) 
        probability = word_freq / self.total_words
        
        return -math.log2(probability)
    
    def calculate_entropy(self, text: str) -> float:
        """
        Calculate Shannon entropy of text
        H(X) = -Σ P(x) * log2(P(x))
        """
        words = [w.lower() for w in word_tokenize(text) if w.isalpha()]
        if not words:
            return 0.0
            
        word_counts = Counter(words)
        total_words = len(words)
        
        entropy = 0.0
        for count in word_counts.values():
            probability = count / total_words
            entropy -= probability * math.log2(probability)
            
        return entropy
    
    def calculate_perplexity(self, text: str) -> float:
        """
        Calculate perplexity based on corpus model
        Perplexity = 2^(average surprisal)
        """
        words = [w.lower() for w in word_tokenize(text) if w.isalpha()]
        if not words:
            return 1.0
            
        total_surprisal = sum(self.calculate_surprisal(word) for word in words)
        avg_surprisal = total_surprisal / len(words)
        
        return 2 ** avg_surprisal
    
    def calculate_information_density_metrics(self, hypothesis: str) -> Dict:
        """
        Calculate multiple information-theoretic measures
        """
        words = [w.lower() for w in word_tokenize(hypothesis) if w.isalpha()]
        
        if not words:
            return {
                'entropy': 0.0,
                'perplexity': 1.0,
                'avg_surprisal': 0.0,
                'unique_word_ratio': 0.0,
                'information_density': 0.0
            }
        
        # Basic metrics
        entropy = self.calculate_entropy(hypothesis)
        perplexity = self.calculate_perplexity(hypothesis)
        avg_surprisal = sum(self.calculate_surprisal(w) for w in words) / len(words)
        unique_word_ratio = len(set(words)) / len(words)
        
        # Combined information density score
        # Higher entropy + higher surprisal + higher uniqueness = higher info density
        information_density = (entropy * 0.4 + 
                             (avg_surprisal / 20) * 0.4 +  # Normalized surprisal
                             unique_word_ratio * 0.2)
        
        return {
            'entropy': entropy,
            'perplexity': perplexity,
            'avg_surprisal': avg_surprisal,
            'unique_word_ratio': unique_word_ratio,
            'information_density': information_density,
            'word_count': len(words)
        }

# Example usage
if __name__ == "__main__":
    evaluator = ScientificHypothesisEvaluator()
    
    # Sample literature
    literature = [
        "Attention mechanisms in neural networks allow models to focus on relevant parts of input sequences.",
        "Transformer architectures use self-attention to process sequences in parallel.",
        "Research shows that attention improves performance on long-range dependency tasks."
    ]
    
    hypothesis = """
    If neural networks implement multi-head attention mechanisms with at least 8 heads,
    then they will demonstrate significantly improved performance on sequence-to-sequence
    tasks compared to models without attention, as measured by BLEU scores exceeding
    baseline performance by 15% or more.
    """
    
    result = evaluator.evaluate_hypothesis(hypothesis, literature)
    
    print("Multi-Dimensional Hypothesis Evaluation:")
    print(f"Overall Quality Score: {result.overall_quality:.3f}")
    print(f"Specificity: {result.specificity_score:.3f}")
    print(f"Falsifiability: {result.falsifiability_score:.3f}")
    print(f"Conceptual Density: {result.conceptual_density:.3f}")
    print(f"Empirical Grounding: {result.empirical_grounding:.3f}")
    print(f"Predictive Content: {result.predictive_content:.3f}")