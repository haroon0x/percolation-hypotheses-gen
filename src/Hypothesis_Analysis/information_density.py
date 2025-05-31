import spacy
import os
import spacy.cli
from spacy.tokens import Doc
from typing import Set
import textstat
from spacy.lang.en.stop_words import STOP_WORDS
from entity_labels import *

NLP_PROCESSOR = None
SCISPACY_MODEL_NAME = "en_core_sci_md"


MIN_LEMMA_LENGTH_FOR_NOUN_CONCEPTS = 3

MIN_MAX_SCALING_PARAMS = {
    'concept_richness_norm': {'min': 0.02, 'max': 0.35},
    'proposition_density_norm': {'min': 0.3, 'max': 2.5},
}

FEATURE_WEIGHTS = {
    'concept_richness': 0.45,
    'proposition_density': 0.35,
    'lexical_density': 0.20
}

def load_nlp_model(model_name: str = SCISPACY_MODEL_NAME):
    global NLP_PROCESSOR
    
    if NLP_PROCESSOR is not None:
        current_model_base_name = NLP_PROCESSOR.meta.get('name', '')
        requested_model_base_name = model_name.split('/')[-1].split('@')[0]
        if current_model_base_name == requested_model_base_name:
            return NLP_PROCESSOR
        else:
            NLP_PROCESSOR = None

    primary_model_loaded = False
    try:
        NLP_PROCESSOR = spacy.load(model_name)
        primary_model_loaded = True
    except OSError:
        try:
            spacy.cli.download(model_name)
            NLP_PROCESSOR = spacy.load(model_name)
            primary_model_loaded = True
        except SystemExit:
             pass
        except Exception:
            pass

    if not primary_model_loaded:
        fallback_model = "en_core_web_md"
        try:
            NLP_PROCESSOR = spacy.load(fallback_model)
        except OSError:
            try:
                spacy.cli.download(fallback_model)
                NLP_PROCESSOR = spacy.load(fallback_model)
            except SystemExit:
                NLP_PROCESSOR = None
            except Exception:
                NLP_PROCESSOR = None
        except Exception:
             NLP_PROCESSOR = None
            
    if NLP_PROCESSOR is None:
        print(f"CRITICAL ERROR: No spaCy model could be loaded after trying {model_name} and fallback.")
    else:
        print(f"spaCy model loaded: {NLP_PROCESSOR.meta.get('name', 'N/A')} (Version: {NLP_PROCESSOR.meta.get('version', 'N/A')})")
    return NLP_PROCESSOR

def get_total_word_and_sentence_count(doc: Doc) -> tuple[int, int]:
    if not doc:
        return (0,0)
    total_words = 0
    for token in doc:
        if not token.is_punct and not token.is_space:
            total_words += 1

    sents_list = list(doc.sents)
    num_sents = len(sents_list)
    
    if num_sents == 0 and total_words > 0:
        num_sents = 1 

    return (total_words, num_sents)

def get_specific_concepts(doc: Doc) -> Set[str]:
    if not doc: return set()
    concepts: Set[str] = set()
    entity_token_indices: Set[int] = set()

    for ent in doc.ents:
        if ent.label_ in INCLUDED_SCIENTIFIC_ENTITY_LABELS and ent.label_ not in EXCLUDED_GENERAL_NER_LABELS:
            if ent.lemma_ and len(ent.lemma_.strip()) > 0:
                concepts.add(ent.lemma_.lower())
                for token_in_ent in ent:
                    entity_token_indices.add(token_in_ent.i)
    
    for noun_chunk in doc.noun_chunks:
        is_part_of_processed_entity = any(token.i in entity_token_indices for token in noun_chunk)
        if is_part_of_processed_entity:
            continue
        root_token = noun_chunk.root
        if (root_token.pos_ == "NOUN" or root_token.pos_ == "PROPN") and \
           not root_token.is_stop and \
           not root_token.is_punct and \
           root_token.lemma_ and \
           len(root_token.lemma_) >= MIN_LEMMA_LENGTH_FOR_NOUN_CONCEPTS:
            concepts.add(root_token.lemma_.lower())

    for token in doc:
        if token.i in entity_token_indices:
            continue
        is_in_noun_chunk_already = False
        for nc in doc.noun_chunks:
            if token.i >= nc.start and token.i < nc.end:
                is_in_noun_chunk_already = True
                break
        if is_in_noun_chunk_already:
            continue

        if (token.pos_ == "NOUN" or token.pos_ == "PROPN") and \
           not token.is_stop and \
           not token.is_punct and \
           token.lemma_ and \
           len(token.lemma_) >= MIN_LEMMA_LENGTH_FOR_NOUN_CONCEPTS:
            concepts.add(token.lemma_.lower())
            
    return concepts

def get_propositions_count(doc: Doc) -> int:
    if not doc: return 0
    proposition_count = 0
    counted_verb_indices: Set[int] = set()

    for sent in doc.sents:
        sentence_has_proposition = False
        for token in sent:
            if token.pos_ == "VERB" and not token.is_aux:
                if token.dep_ == "ROOT" or \
                   (token.dep_ in {"ccomp", "xcomp", "advcl", "acl"} and token.head.pos_ == "VERB") or \
                   (token.dep_ == "conj" and token.head.pos_ == "VERB" and token.head.i in counted_verb_indices):
                    current_verb_head = token
                    while current_verb_head.head != current_verb_head and current_verb_head.head.pos_ == "VERB" and current_verb_head.dep_ != "ROOT":
                        current_verb_head = current_verb_head.head
                    
                    if current_verb_head.i not in counted_verb_indices:
                        proposition_count += 1
                        counted_verb_indices.add(current_verb_head.i)
                        sentence_has_proposition = True
        
        if not sentence_has_proposition and len(sent.text.strip()) > 2 :
            meaningful_tokens_in_sent = [t for t in sent if not t.is_punct and not t.is_space]
            if len(meaningful_tokens_in_sent) > 1 :
                proposition_count += 1
    
    if proposition_count == 0 and len([t for t in doc if not t.is_space]) > 0:
        return 1
    return proposition_count

def get_lexical_density(hypothesis_text: str) -> float:
    if not hypothesis_text or not hypothesis_text.strip():
        return 0.0
    try:
        score = textstat.lexical_density(hypothesis_text)
        return score if score is not None else 30.0
    except Exception:
        return 30.0

def calculate_intrinsic_information_density(hypothesis_text: str) -> float:
    nlp = load_nlp_model()
    if not nlp or not hypothesis_text or not hypothesis_text.strip():
        return 0.0

    doc = nlp(hypothesis_text)
    total_words, num_sentences = get_total_word_and_sentence_count(doc)
    if total_words == 0:
        return 0.0
    if num_sentences == 0:
        num_sentences = 1

    unique_concepts = get_specific_concepts(doc)
    num_unique_specific_concepts = len(unique_concepts)
    concept_richness_norm = num_unique_specific_concepts / total_words if total_words > 0 else 0

    num_propositions = get_propositions_count(doc)
    proposition_density_norm = num_propositions / num_sentences if num_sentences > 0 else 0
    if num_sentences == 0 and num_propositions > 0:
        proposition_density_norm = num_propositions

    lex_density_val = get_lexical_density(hypothesis_text)

    min_cr = MIN_MAX_SCALING_PARAMS['concept_richness_norm']['min']
    max_cr = MIN_MAX_SCALING_PARAMS['concept_richness_norm']['max']
    cr_0_1 = min(1.0, max(0.0, (concept_richness_norm - min_cr) / (max_cr - min_cr + 1e-6)))

    min_pd = MIN_MAX_SCALING_PARAMS['proposition_density_norm']['min']
    max_pd = MIN_MAX_SCALING_PARAMS['proposition_density_norm']['max']
    pd_0_1 = min(1.0, max(0.0, (proposition_density_norm - min_pd) / (max_pd - min_pd + 1e-6)))
    
    ld_0_1 = lex_density_val / 100.0

    final_density_score_0_1 = (
        FEATURE_WEIGHTS['concept_richness'] * cr_0_1 +
        FEATURE_WEIGHTS['proposition_density'] * pd_0_1 +
        FEATURE_WEIGHTS['lexical_density'] * ld_0_1
    )
    
    final_density_score_0_100 = final_density_score_0_1 * 100.0
    
    return round(max(0.0, min(100.0, final_density_score_0_100)), 2)

if __name__ == '__main__':
    load_nlp_model() 

    test_hypotheses_for_density = [
        ("E=mc².", "Iconic, simple structure, one core concept/proposition in physics."),
        ("The novel compound XZ-7 selectively inhibits kinase Y, leading to apoptosis in cancer cell line A, but not in normal cell line B, via the downregulation of protein Z.", "Highly specific entities, multiple propositions."),
        ("Some things might affect other things in cells under certain circumstances sometimes.", "Vague, low specificity, few concrete propositions."),
        ("This statement is short and about statements.", "Simple, self-referential, low specific concepts."),
        ("Advanced neuro-computational models leveraging deep graph convolutional networks can predict synaptic plasticity changes from high-resolution connectomic and transcriptomic data integration.", "Technical terms, multiple concepts and implied relations."),
        ("The Earth revolves around the Sun, which is a star at the center of our solar system, influencing terrestrial seasons.", "Multiple concepts, clear propositions."),
        ("A catalytic converter reduces harmful emissions from an internal combustion engine through redox reactions involving precious metals like platinum, palladium, and rhodium.", "Specific technical terms, clear process described."),
        ("Metabolic pathways are complex networks of biochemical reactions.", "Definitional, moderate concepts."),
        ("The theory posits that by manipulating quantum entanglement, faster-than-light communication could be achieved, challenging current interpretations of special relativity and information causality within closed systems, potentially revolutionizing interstellar travel and computational paradigms.", "Highly abstract, many complex concepts and far-reaching implications.")
    ]

    if NLP_PROCESSOR:
        print(f"\n--- Testing Intrinsic Information Density (using {NLP_PROCESSOR.meta['name']}) ---")
        for i, (hyp_text, desc) in enumerate(test_hypotheses_for_density):
            intrinsic_density_score = calculate_intrinsic_information_density(hyp_text)
            print(f"\nHypothesis {i+1}: \"{hyp_text}\"")
            print(f"Description: {desc}")
            print(f"Intrinsic Information Density Score: {intrinsic_density_score:.2f}/100.0")
            
            if i < 2 or "XZ-7" in hyp_text:
                doc_detail = NLP_PROCESSOR(hyp_text)
                total_w, num_s = get_total_word_and_sentence_count(doc_detail)
                concepts_found = get_specific_concepts(doc_detail)
                props_found = get_propositions_count(doc_detail)
                lex_d = get_lexical_density(hyp_text)
                print(f"  Details: Words={total_w}, Sents={num_s}, Concepts={len(concepts_found)} ({list(concepts_found)[:10] if concepts_found else 'None'}), Props={props_found}, LexDensity={lex_d:.2f}")
    else:
        print("Skipping information_density tests as spaCy model could not be loaded.")