import numpy as np
import math
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy as scipy_entropy
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

@dataclass
class HypothesisInformationProfile:
    hypothesis: str
    overall_information_density: float
    scientific_quality_score: float
    information_theoretic_score: float
    specificity_score: float
    falsifiability_score: float
    conceptual_complexity: float
    empirical_grounding: float
    predictive_power: float
    semantic_richness: float
    structural_information: float
    novelty_score: float
    coherence_score: float
    detailed_metrics: Dict
    recommendations: List[str]

class AdvancedHypothesisAnalyzer:
    def __init__(self, background_corpus: List[str] = None):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.domain_knowledge = defaultdict(float)
        self.concept_graph = defaultdict(set)
        self.corpus_vocab = Counter()
        self.bigram_vocab = Counter()
        self.trigram_vocab = Counter()
        self.total_words = 0
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 3))
        
        self.scientific_patterns = {
            'quantitative': r'\b\d+\.?\d*\s*(?:%|percent|fold|times|μg|mg|kg|ml|μl|nm|μm|mm|cm|m|km|Hz|kHz|MHz|GHz|°C|°F|K|Pa|kPa|MPa|V|mV|A|mA|Ω|kΩ|MΩ)\b',
            'statistical': r'\b(?:p\s*[<>=]\s*\d+\.?\d*|α\s*[=]\s*\d+\.?\d*|r\s*[=]\s*[+-]?\d+\.?\d*|β\s*[=]\s*[+-]?\d+\.?\d*|t\s*[=]\s*[+-]?\d+\.?\d*|F\s*[=]\s*\d+\.?\d*|χ²\s*[=]\s*\d+\.?\d*)\b',
            'experimental': r'\b(?:control|treatment|intervention|manipulation|randomized|placebo|blinded|crossover|factorial)\b',
            'causal': r'\b(?:causes?|leads?\s+to|results?\s+in|produces?|generates?|triggers?|induces?|mediates?|moderates?|influences?)\b',
            'conditional': r'\b(?:if\s+.*\s+then|when\s+.*\s+will|given\s+.*\s+expect|under\s+conditions?)\b',
            'temporal': r'\b(?:before|after|during|following|preceding|concurrent|simultaneous|sequential)\b',
            'comparative': r'\b(?:compared\s+to|relative\s+to|versus|vs\.?|higher\s+than|lower\s+than|greater\s+than|less\s+than)\b',
            'mechanism': r'\b(?:mechanism|pathway|process|mediates?|through|via|by\s+means\s+of|operates?|functions?)\b'
        }
        
        self.complexity_indicators = {
            'technical_terms': r'\b[A-Z][a-z]*(?:[A-Z][a-z]*)+\b',
            'scientific_notation': r'\b\d+\.?\d*\s*[×x]\s*10\^?[+-]?\d+\b',
            'greek_letters': r'\b(?:alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|omicron|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega)\b',
            'domain_specific': r'\b(?:algorithm|methodology|paradigm|framework|protocol|procedure|technique|approach)\b'
        }
        
        if background_corpus:
            self.build_knowledge_base(background_corpus)
    
    def build_knowledge_base(self, texts: List[str]):
        all_words = []
        all_bigrams = []
        all_trigrams = []
        
        for text in texts:
            words = self._preprocess_text(text)
            all_words.extend(words)
            
            if len(words) >= 2:
                bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
                all_bigrams.extend(bigrams)
            
            if len(words) >= 3:
                trigrams = [f"{words[i]}_{words[i+1]}_{words[i+2]}" for i in range(len(words)-2)]
                all_trigrams.extend(trigrams)
            
            self._extract_domain_concepts(text)
        
        self.corpus_vocab.update(all_words)
        self.bigram_vocab.update(all_bigrams)
        self.trigram_vocab.update(all_trigrams)
        self.total_words = len(all_words)
        
        if texts:
            self.tfidf_vectorizer.fit(texts)
    
    def _preprocess_text(self, text: str) -> List[str]:
        words = word_tokenize(text.lower())
        words = [self.lemmatizer.lemmatize(word) for word in words 
                if word.isalpha() and word not in self.stop_words and len(word) > 2]
        return words
    
    def _extract_domain_concepts(self, text: str):
        technical_terms = re.findall(r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b', text)
        suffix_terms = re.findall(r'\b\w*(?:tion|sion|ment|ity|ness|ism|ology|ography|metry)\b', text.lower())
        
        for term in technical_terms + suffix_terms:
            normalized = term.lower().strip()
            if len(normalized) > 3:
                self.domain_knowledge[normalized] += 1.0
    
    def calculate_information_theoretic_metrics(self, hypothesis: str) -> Dict:
        words = self._preprocess_text(hypothesis)
        
        if not words:
            return {'entropy': 0.0, 'perplexity': 1.0, 'surprisal': 0.0, 'information_density': 0.0}
        
        word_counts = Counter(words)
        total_words = len(words)
        
        local_entropy = 0.0
        for count in word_counts.values():
            prob = count / total_words
            local_entropy -= prob * math.log2(prob)
        
        surprisal_scores = []
        for word in words:
            corpus_freq = self.corpus_vocab.get(word, 1)
            prob = corpus_freq / max(self.total_words, 1)
            surprisal = -math.log2(prob)
            surprisal_scores.append(surprisal)
        
        avg_surprisal = np.mean(surprisal_scores) if surprisal_scores else 0.0
        perplexity = 2 ** avg_surprisal
        
        unique_ratio = len(set(words)) / len(words)
        rare_word_ratio = sum(1 for word in words if self.corpus_vocab.get(word, 0) < 5) / len(words)
        
        bigram_entropy = self._calculate_ngram_entropy(words, 2)
        trigram_entropy = self._calculate_ngram_entropy(words, 3)
        
        information_density = (
            local_entropy * 0.25 +
            (avg_surprisal / 20) * 0.25 +
            unique_ratio * 0.15 +
            rare_word_ratio * 0.15 +
            (bigram_entropy / 10) * 0.1 +
            (trigram_entropy / 15) * 0.1
        )
        
        return {
            'entropy': local_entropy,
            'perplexity': perplexity,
            'avg_surprisal': avg_surprisal,
            'unique_ratio': unique_ratio,
            'rare_word_ratio': rare_word_ratio,
            'bigram_entropy': bigram_entropy,
            'trigram_entropy': trigram_entropy,
            'information_density': information_density
        }
    
    def _calculate_ngram_entropy(self, words: List[str], n: int) -> float:
        if len(words) < n:
            return 0.0
        
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        ngram_counts = Counter(ngrams)
        total_ngrams = len(ngrams)
        
        entropy = 0.0
        for count in ngram_counts.values():
            prob = count / total_ngrams
            entropy -= prob * math.log2(prob)
        
        return entropy
    
    def evaluate_scientific_quality(self, hypothesis: str) -> Dict:
        metrics = {}
        word_count = len(hypothesis.split())
        
        for pattern_type, pattern in self.scientific_patterns.items():
            matches = len(re.findall(pattern, hypothesis, re.IGNORECASE))
            metrics[f'{pattern_type}_density'] = matches / word_count if word_count > 0 else 0
        
        falsifiability_score = (
            metrics['quantitative_density'] * 0.3 +
            metrics['statistical_density'] * 0.25 +
            metrics['experimental_density'] * 0.2 +
            metrics['conditional_density'] * 0.15 +
            metrics['comparative_density'] * 0.1
        )
        
        predictive_power = (
            metrics['causal_density'] * 0.4 +
            metrics['conditional_density'] * 0.3 +
            metrics['temporal_density'] * 0.2 +
            metrics['mechanism_density'] * 0.1
        )
        
        specificity_score = (
            metrics['quantitative_density'] * 0.4 +
            metrics['statistical_density'] * 0.3 +
            metrics['comparative_density'] * 0.2 +
            metrics['experimental_density'] * 0.1
        )
        
        return {
            'falsifiability': min(falsifiability_score * 5, 1.0),
            'predictive_power': min(predictive_power * 5, 1.0),
            'specificity': min(specificity_score * 5, 1.0),
            'pattern_metrics': metrics
        }
    
    def evaluate_conceptual_complexity(self, hypothesis: str) -> Dict:
        words = self._preprocess_text(hypothesis)
        word_count = len(words)
        
        if word_count == 0:
            return {'complexity': 0.0, 'domain_coverage': 0.0, 'technical_density': 0.0}
        
        domain_terms = sum(1 for word in words if word in self.domain_knowledge)
        domain_coverage = domain_terms / word_count
        
        complexity_scores = []
        for pattern_type, pattern in self.complexity_indicators.items():
            matches = len(re.findall(pattern, hypothesis, re.IGNORECASE))
            complexity_scores.append(matches / word_count)
        
        technical_density = np.mean(complexity_scores)
        
        sentence_count = len(sent_tokenize(hypothesis))
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        sentence_complexity = min(avg_sentence_length / 20, 1.0)
        
        conceptual_complexity = (
            domain_coverage * 0.4 +
            technical_density * 0.3 +
            sentence_complexity * 0.3
        )
        
        return {
            'complexity': conceptual_complexity,
            'domain_coverage': domain_coverage,
            'technical_density': technical_density,
            'sentence_complexity': sentence_complexity
        }
    
    def evaluate_semantic_richness(self, hypothesis: str, literature_texts: List[str] = None) -> Dict:
        if not literature_texts or not hasattr(self.tfidf_vectorizer, 'vocabulary_'):
            return {'semantic_similarity': 0.0, 'concept_diversity': 0.0, 'semantic_richness': 0.0}
        
        try:
            combined_literature = ' '.join(literature_texts)
            texts = [hypothesis, combined_literature]
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
            semantic_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            semantic_similarity = 0.0
        
        words = self._preprocess_text(hypothesis)
        unique_concepts = len(set(words))
        total_words = len(words)
        concept_diversity = unique_concepts / total_words if total_words > 0 else 0
        
        rare_concept_ratio = sum(1 for word in words 
                               if self.domain_knowledge.get(word, 0) > 0 and self.corpus_vocab.get(word, 0) < 10) / max(total_words, 1)
        
        semantic_richness = (
            concept_diversity * 0.4 +
            rare_concept_ratio * 0.3 +
            min(semantic_similarity, 0.8) * 0.3
        )
        
        return {
            'semantic_similarity': semantic_similarity,
            'concept_diversity': concept_diversity,
            'rare_concept_ratio': rare_concept_ratio,
            'semantic_richness': semantic_richness
        }
    
    def evaluate_structural_information(self, hypothesis: str) -> Dict:
        sentences = sent_tokenize(hypothesis)
        sentence_count = len(sentences)
        
        if sentence_count == 0:
            return {'structural_complexity': 0.0, 'logical_structure': 0.0}
        
        sentence_lengths = [len(word_tokenize(sent)) for sent in sentences]
        length_variance = np.var(sentence_lengths) if len(sentence_lengths) > 1 else 0
        avg_sentence_length = np.mean(sentence_lengths)
        
        logical_connectors = [
            'therefore', 'however', 'moreover', 'furthermore', 'consequently',
            'thus', 'hence', 'because', 'since', 'although', 'whereas', 'while'
        ]
        connector_count = sum(1 for connector in logical_connectors 
                            if connector in hypothesis.lower())
        
        punctuation_complexity = len([c for c in hypothesis if c in ',;:()[]{}']) / len(hypothesis)
        
        structural_complexity = (
            min(length_variance / 100, 1.0) * 0.3 +
            min(avg_sentence_length / 30, 1.0) * 0.3 +
            min(connector_count / sentence_count, 1.0) * 0.25 +
            punctuation_complexity * 0.15
        )
        
        logical_structure = min(connector_count / sentence_count, 1.0) if sentence_count > 0 else 0
        
        return {
            'structural_complexity': structural_complexity,
            'logical_structure': logical_structure,
            'sentence_variance': length_variance,
            'avg_sentence_length': avg_sentence_length
        }
    
    def calculate_novelty_score(self, hypothesis: str, literature_texts: List[str] = None) -> float:
        if not literature_texts:
            return 0.5
        
        words = set(self._preprocess_text(hypothesis))
        literature_words = set()
        
        for text in literature_texts:
            literature_words.update(self._preprocess_text(text))
        
        if not words:
            return 0.0
        
        novel_concepts = words - literature_words
        novelty_ratio = len(novel_concepts) / len(words)
        
        uncommon_concepts = sum(1 for word in words 
                              if self.corpus_vocab.get(word, 0) < 3) / len(words)
        
        novelty_score = novelty_ratio * 0.6 + uncommon_concepts * 0.4
        return min(novelty_score, 1.0)
    
    def calculate_coherence_score(self, hypothesis: str) -> float:
        sentences = sent_tokenize(hypothesis)
        if len(sentences) < 2:
            return 1.0
        
        sentence_similarities = []
        for i in range(len(sentences) - 1):
            words1 = set(self._preprocess_text(sentences[i]))
            words2 = set(self._preprocess_text(sentences[i + 1]))
            
            if not words1 or not words2:
                similarity = 0.0
            else:
                intersection = len(words1 & words2)
                union = len(words1 | words2)
                similarity = intersection / union if union > 0 else 0.0
            
            sentence_similarities.append(similarity)
        
        return np.mean(sentence_similarities) if sentence_similarities else 1.0
    
    def generate_recommendations(self, metrics: Dict) -> List[str]:
        recommendations = []
        
        if metrics['scientific_quality']['specificity'] < 0.3:
            recommendations.append("Increase specificity by adding quantitative measures or statistical parameters")
        
        if metrics['scientific_quality']['falsifiability'] < 0.4:
            recommendations.append("Enhance falsifiability by including testable predictions and observable outcomes")
        
        if metrics['information_theoretic']['information_density'] < 0.3:
            recommendations.append("Improve information density by using more technical terminology and reducing redundancy")
        
        if metrics['conceptual_complexity']['complexity'] < 0.2:
            recommendations.append("Increase conceptual complexity by incorporating domain-specific concepts")
        
        if metrics['novelty_score'] < 0.2:
            recommendations.append("Enhance novelty by introducing less common concepts or novel combinations")
        
        if metrics['coherence_score'] < 0.5:
            recommendations.append("Improve coherence by strengthening logical connections between sentences")
        
        return recommendations
    
    def analyze_hypothesis(self, hypothesis: str, literature_texts: List[str] = None) -> HypothesisInformationProfile:
        if literature_texts:
            self.build_knowledge_base(literature_texts)
        
        info_theoretic = self.calculate_information_theoretic_metrics(hypothesis)
        scientific_quality = self.evaluate_scientific_quality(hypothesis)
        conceptual_complexity = self.evaluate_conceptual_complexity(hypothesis)
        semantic_richness = self.evaluate_semantic_richness(hypothesis, literature_texts)
        structural_info = self.evaluate_structural_information(hypothesis)
        novelty_score = self.calculate_novelty_score(hypothesis, literature_texts)
        coherence_score = self.calculate_coherence_score(hypothesis)
        
        empirical_grounding = semantic_richness['semantic_similarity']
        
        information_theoretic_score = (
            info_theoretic['information_density'] * 0.4 +
            info_theoretic['entropy'] / 10 * 0.3 +
            info_theoretic['unique_ratio'] * 0.2 +
            info_theoretic['rare_word_ratio'] * 0.1
        )
        
        scientific_quality_score = (
            scientific_quality['specificity'] * 0.3 +
            scientific_quality['falsifiability'] * 0.35 +
            scientific_quality['predictive_power'] * 0.35
        )
        
        overall_information_density = (
            information_theoretic_score * 0.25 +
            scientific_quality_score * 0.25 +
            conceptual_complexity['complexity'] * 0.2 +
            semantic_richness['semantic_richness'] * 0.15 +
            structural_info['structural_complexity'] * 0.1 +
            novelty_score * 0.05
        )
        
        detailed_metrics = {
            'information_theoretic': info_theoretic,
            'scientific_quality': scientific_quality,
            'conceptual_complexity': conceptual_complexity,
            'semantic_richness': semantic_richness,
            'structural_information': structural_info,
            'novelty_score': novelty_score,
            'coherence_score': coherence_score
        }
        
        recommendations = self.generate_recommendations(detailed_metrics)
        
        return HypothesisInformationProfile(
            hypothesis=hypothesis,
            overall_information_density=overall_information_density,
            scientific_quality_score=scientific_quality_score,
            information_theoretic_score=information_theoretic_score,
            specificity_score=scientific_quality['specificity'],
            falsifiability_score=scientific_quality['falsifiability'],
            conceptual_complexity=conceptual_complexity['complexity'],
            empirical_grounding=empirical_grounding,
            predictive_power=scientific_quality['predictive_power'],
            semantic_richness=semantic_richness['semantic_richness'],
            structural_information=structural_info['structural_complexity'],
            novelty_score=novelty_score,
            coherence_score=coherence_score,
            detailed_metrics=detailed_metrics,
            recommendations=recommendations
        )

if __name__ == "__main__":
    analyzer = AdvancedHypothesisAnalyzer()
    
    literature = [
        "Neural attention mechanisms enable models to selectively focus on relevant input features.",
        "Transformer architectures utilize multi-head self-attention for parallel sequence processing.",
        "Research demonstrates that attention mechanisms significantly improve performance on long-range dependencies.",
        "Empirical studies show that attention visualization reveals interpretable patterns in model behavior."
    ]
    
    complex_hypothesis = """
    The emergence of intelligence from non-biological matter requires the spontaneous symmetry breaking of thermodynamic equilibrium within a confined, information-rich, non-ergodic system, leading to the formation of self-referential, error-correcting, and anticipatory dissipative structures capable of recursively manipulating their internal state space to maximize predictive accuracy and minimize entropic dissipation through the active construction of hierarchical, context-dependent models of their external environment, thereby achieving adaptive autonomy and exhibiting emergent sentience.
    """
    
    result = analyzer.analyze_hypothesis(complex_hypothesis, literature)
    
    print("Advanced Hypothesis Information Density Analysis")
    print("=" * 60)
    print(f"Overall Information Density: {result.overall_information_density:.4f}")
    print(f"Scientific Quality Score: {result.scientific_quality_score:.4f}")
    print(f"Information Theoretic Score: {result.information_theoretic_score:.4f}")
    print(f"Specificity: {result.specificity_score:.4f}")
    print(f"Falsifiability: {result.falsifiability_score:.4f}")
    print(f"Conceptual Complexity: {result.conceptual_complexity:.4f}")
    print(f"Empirical Grounding: {result.empirical_grounding:.4f}")
    print(f"Predictive Power: {result.predictive_power:.4f}")
    print(f"Semantic Richness: {result.semantic_richness:.4f}")
    print(f"Structural Information: {result.structural_information:.4f}")
    print(f"Novelty Score: {result.novelty_score:.4f}")
    print(f"Coherence Score: {result.coherence_score:.4f}")
    
    print("\nRecommendations:")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"{i}. {rec}")