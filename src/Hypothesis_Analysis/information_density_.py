import os
import re
import json
import math
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict, Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import PyPDF2
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download required NLTK data
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

class LiteratureProcessor:
    """Processes scientific literature from PDF and text files"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        # Load spaCy model for NER and linguistic analysis
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return ""
    
    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from PDF or text file"""
        if file_path.lower().endswith('.pdf'):
            return self.extract_text_from_pdf(file_path)
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                return ""
    
    def extract_citations(self, text: str) -> List[str]:
        """Extract citations from text using regex patterns"""
        citation_patterns = [
            r'\([^)]*\d{4}[^)]*\)',  # (Author, 2023) or (Author et al., 2023)
            r'\[[^\]]*\d+[^\]]*\]',  # [1], [1,2,3], [Author 2023]
            r'(?:doi:|DOI:)\s*[\w\.\-\/]+',  # DOI references
            r'(?:et al\.|et al)',  # et al. references
        ]
        
        citations = []
        for pattern in citation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            citations.extend(matches)
        
        return list(set(citations))  # Remove duplicates
    
    def extract_key_concepts(self, text: str) -> Set[str]:
        """Extract key scientific concepts using NLP"""
        concepts = set()
        
        if self.nlp:
            doc = self.nlp(text)
            # Extract named entities (scientific terms, organizations, etc.)
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PERSON', 'GPE', 'PRODUCT', 'EVENT']:
                    concepts.add(ent.text.lower())
            
            # Extract noun phrases as potential concepts
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3:  # Keep short phrases
                    concepts.add(chunk.text.lower())
        
        # Fallback: extract capitalized words and phrases
        capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        concepts.update([word.lower() for word in capitalized_words])
        
        # Filter out common words
        concepts = {concept for concept in concepts 
                   if concept not in self.stop_words and len(concept) > 2}
        
        return concepts

class SemanticAnalyzer:
    """Analyzes semantic relationships between hypotheses and literature"""
    
    def __init__(self):
        # Load sentence transformer for semantic similarity
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Warning: Could not load sentence transformer: {e}")
            self.model = None
    
    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts"""
        if not self.model:
            return 0.0
        
        try:
            embeddings = self.model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Error computing similarity: {e}")
            return 0.0
    
    def find_supporting_sentences(self, hypothesis: str, literature_sentences: List[str], 
                                threshold: float = 0.3) -> List[Tuple[str, float]]:
        """Find sentences in literature that support the hypothesis"""
        if not self.model:
            return []
        
        supporting_sentences = []
        
        try:
            # Get hypothesis embedding
            hyp_embedding = self.model.encode([hypothesis])
            
            # Get literature embeddings
            lit_embeddings = self.model.encode(literature_sentences)
            
            # Compute similarities
            similarities = cosine_similarity(hyp_embedding, lit_embeddings)[0]
            
            # Find sentences above threshold
            for i, similarity in enumerate(similarities):
                if similarity >= threshold:
                    supporting_sentences.append((literature_sentences[i], float(similarity)))
            
            # Sort by similarity score
            supporting_sentences.sort(key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            print(f"Error finding supporting sentences: {e}")
        
        return supporting_sentences

class VerifiabilityScorer:
    """Scores how well a hypothesis can be verified against literature"""
    
    def __init__(self, literature_processor: LiteratureProcessor, 
                 semantic_analyzer: SemanticAnalyzer):
        self.lit_processor = literature_processor
        self.semantic_analyzer = semantic_analyzer
    
    def extract_claims(self, hypothesis: str) -> List[str]:
        """Extract individual claims from hypothesis"""
        # Split hypothesis into sentences as individual claims
        sentences = sent_tokenize(hypothesis)
        
        # Filter out very short sentences
        claims = [sent.strip() for sent in sentences if len(sent.strip()) > 10]
        
        return claims
    
    def score_claim_verifiability(self, claim: str, literature_concepts: Set[str], 
                                literature_sentences: List[str]) -> Dict:
        """Score how well a claim can be verified"""
        score_data = {
            'claim': claim,
            'concept_overlap_score': 0.0,
            'semantic_support_score': 0.0,
            'citation_mentions': 0,
            'supporting_sentences': [],
            'total_score': 0.0
        }
        
        # 1. Concept overlap score
        claim_words = set(word.lower() for word in word_tokenize(claim) 
                         if word.isalpha() and word.lower() not in self.lit_processor.stop_words)
        
        if literature_concepts:
            overlap = len(claim_words.intersection(literature_concepts))
            score_data['concept_overlap_score'] = overlap / len(claim_words) if claim_words else 0.0
        
        # 2. Semantic support score
        supporting_sentences = self.semantic_analyzer.find_supporting_sentences(
            claim, literature_sentences, threshold=0.3)
        
        if supporting_sentences:
            # Average similarity of top 3 supporting sentences
            top_similarities = [sim for _, sim in supporting_sentences[:3]]
            score_data['semantic_support_score'] = sum(top_similarities) / len(top_similarities)
            score_data['supporting_sentences'] = supporting_sentences[:5]  # Store top 5
        
        # 3. Citation mentions (simple heuristic)
        citation_indicators = ['study', 'research', 'found', 'showed', 'demonstrated', 
                              'evidence', 'according to', 'reported']
        score_data['citation_mentions'] = sum(1 for indicator in citation_indicators 
                                            if indicator in claim.lower())
        
        # Calculate total score (weighted combination)
        score_data['total_score'] = (
            0.4 * score_data['concept_overlap_score'] +
            0.5 * score_data['semantic_support_score'] +
            0.1 * min(score_data['citation_mentions'] / 3.0, 1.0)  # Normalized citation score
        )
        
        return score_data

class InformationDensityCalculator:
    """Main class for calculating information density of hypotheses"""
    
    def __init__(self):
        self.lit_processor = LiteratureProcessor()
        self.semantic_analyzer = SemanticAnalyzer()
        self.verifiability_scorer = VerifiabilityScorer(self.lit_processor, self.semantic_analyzer)
        
        # Literature database
        self.literature_texts = []
        self.literature_sentences = []
        self.literature_concepts = set()
        self.literature_citations = []
    
    def load_literature(self, literature_paths: List[str]):
        """Load and process literature from file paths"""
        print("Loading literature...")
        
        for path in literature_paths:
            if os.path.exists(path):
                text = self.lit_processor.extract_text_from_file(path)
                if text:
                    self.literature_texts.append(text)
                    
                    # Extract sentences
                    sentences = sent_tokenize(text)
                    self.literature_sentences.extend(sentences)
                    
                    # Extract concepts
                    concepts = self.lit_processor.extract_key_concepts(text)
                    self.literature_concepts.update(concepts)
                    
                    # Extract citations
                    citations = self.lit_processor.extract_citations(text)
                    self.literature_citations.extend(citations)
                    
                    print(f"Processed: {path}")
            else:
                print(f"File not found: {path}")
        
        print(f"Loaded {len(self.literature_texts)} documents")
        print(f"Total sentences: {len(self.literature_sentences)}")
        print(f"Total concepts: {len(self.literature_concepts)}")
        print(f"Total citations: {len(self.literature_citations)}")
    
    def calculate_information_density(self, hypothesis: str) -> Dict:
        """Calculate comprehensive information density score"""
        
        if not self.literature_texts:
            raise ValueError("No literature loaded. Call load_literature() first.")
        
        # Basic metrics
        word_count = len(word_tokenize(hypothesis))
        sentence_count = len(sent_tokenize(hypothesis))
        
        # Extract claims for verification
        claims = self.verifiability_scorer.extract_claims(hypothesis)
        
        # Score each claim
        claim_scores = []
        for claim in claims:
            score_data = self.verifiability_scorer.score_claim_verifiability(
                claim, self.literature_concepts, self.literature_sentences)
            claim_scores.append(score_data)
        
        # Calculate aggregate scores
        verifiable_claims = [score for score in claim_scores if score['total_score'] > 0.2]
        avg_verifiability = (sum(score['total_score'] for score in claim_scores) / 
                           len(claim_scores) if claim_scores else 0.0)
        
        # Calculate citation density
        hypothesis_citations = self.lit_processor.extract_citations(hypothesis)
        citation_density = len(hypothesis_citations) / word_count if word_count > 0 else 0.0
        
        # Calculate verifiable claims ratio
        verifiable_claims_ratio = len(verifiable_claims) / len(claims) if claims else 0.0
        
        # Calculate final information density
        information_density = (
            0.3 * citation_density * 100 +  # Citations per unit text (scaled)
            0.4 * verifiable_claims_ratio +  # Ratio of verifiable claims
            0.3 * avg_verifiability  # Average verifiability score
        )
        
        return {
            'hypothesis': hypothesis,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'claims_count': len(claims),
            'verifiable_claims_count': len(verifiable_claims),
            'citation_count': len(hypothesis_citations),
            'citation_density': citation_density,
            'verifiable_claims_ratio': verifiable_claims_ratio,
            'avg_verifiability_score': avg_verifiability,
            'information_density': information_density,
            'claim_details': claim_scores,
            'supporting_citations': hypothesis_citations
        }
    
    def batch_calculate_density(self, hypotheses: List[str]) -> List[Dict]:
        """Calculate information density for multiple hypotheses"""
        results = []
        
        for i, hypothesis in enumerate(hypotheses):
            print(f"Processing hypothesis {i+1}/{len(hypotheses)}")
            try:
                result = self.calculate_information_density(hypothesis)
                results.append(result)
            except Exception as e:
                print(f"Error processing hypothesis {i+1}: {e}")
                results.append({
                    'hypothesis': hypothesis,
                    'information_density': 0.0,
                    'error': str(e)
                })
        
        return results

class PercolationDetector:
    """Detects the percolation point in complexity vs information density"""
    
    @staticmethod
    def detect_percolation_point(complexity_scores: List[float], 
                               density_scores: List[float]) -> Dict:
        """Detect the percolation point where density drops significantly"""
        
        if len(complexity_scores) != len(density_scores) or len(complexity_scores) < 3:
            return {'percolation_index': -1, 'percolation_point': None}
        
        # Calculate rate of change in density
        density_changes = []
        for i in range(1, len(density_scores)):
            change = density_scores[i] - density_scores[i-1]
            density_changes.append(change)
        
        # Find the point of steepest decline
        min_change = min(density_changes)
        min_change_index = density_changes.index(min_change)
        
        # The percolation point is where the steepest decline occurs
        percolation_index = min_change_index + 1  # +1 because we started from index 1
        
        return {
            'percolation_index': percolation_index,
            'percolation_point': {
                'complexity': complexity_scores[percolation_index],
                'density': density_scores[percolation_index]
            },
            'density_change_rate': min_change,
            'all_changes': density_changes
        }

# Example usage and testing
if __name__ == "__main__":
    import os

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pdf_path = os.path.join(BASE_DIR, "sample_pdfs", "2505.11309v1.pdf")

    # Initialize the calculator
    calculator = InformationDensityCalculator()
    
    # Example literature loading (you would provide actual file paths)
    calculator.load_literature([pdf_path])
    
    # Example hypothesis
    example_hypothesis = """
    Recent studies have shown that machine learning algorithms can predict protein folding 
    with high accuracy. This breakthrough, as demonstrated by AlphaFold, represents a 
    significant advancement in computational biology. The implications for drug discovery 
    are substantial, potentially reducing the time required for pharmaceutical development.
    """
    
    # Calculate information density (would work after loading literature)
    result = calculator.calculate_information_density(example_hypothesis)
    print(json.dumps(result, indent=2))
    
    print("Information Density Calculator initialized successfully!")
    print("Load literature using calculator.load_literature([file_paths]) before calculating density.")