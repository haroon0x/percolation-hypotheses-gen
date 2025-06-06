from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
import nltk
import os
from src.main import *
import chromadb
import time
import re
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

CHROMA_DB_PERSIST_PATH = "./chroma_data_store"
try:
    if not os.path.exists(CHROMA_DB_PERSIST_PATH):
        os.makedirs(CHROMA_DB_PERSIST_PATH)
    CHROMA_CLIENT = chromadb.PersistentClient(path=CHROMA_DB_PERSIST_PATH) 
    print(f"ChromaDB persistent client initialized. Data will be stored in: {CHROMA_DB_PERSIST_PATH}")
except Exception as e:
    print(f"Fatal Error: Could not initialize persistent ChromaDB client at '{CHROMA_DB_PERSIST_PATH}': {e}.")
    print("Attempting to use in-memory ChromaDB client as fallback (data will be lost on exit).")
    try:
        CHROMA_CLIENT = chromadb.Client()
        print("ChromaDB in-memory client initialized.")
    except Exception as e_mem:
        print(f"Fatal Error: Could not initialize in-memory ChromaDB client either: {e_mem}.")
        CHROMA_CLIENT = None

def extract_text_from_pdf(pdf_path):
    pdf_text = ""
    try:
        reader = PdfReader(str(pdf_path))
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    cleaned_text = clean_extracted_text(page_text)
                    if cleaned_text:
                        pdf_text += cleaned_text + '\n'
            except Exception as page_error:
                print(f"Error extracting text from page {page_num + 1}: {str(page_error)}")
                continue
    except Exception as e:
        print(f"Error occurred during PDF text extraction: {str(e)}")
        return ""
    except FileNotFoundError:
        print(f"Error: PDF file not found at path: {pdf_path}")
        return ""
    return pdf_text.strip()

def clean_extracted_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\/\@\#\$\%\^\&\*\+\=\<\>\|\\]', '', text)
    text = re.sub(r'\.{3,}', '...', text)
    text = re.sub(r'\n+', '\n', text)
    text = text.strip()
    return text

def create_text_chunks(
    text: str,
    chunk_size: int = 1000,
    overlap_size: int = 200,
    separators: list[str] = None,
    min_chunk_length: int = 50
) -> list[str]:
    
    if not text or not text.strip():
        return []

    if separators is None:
        separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

    text_splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=overlap_size,
        length_function=len,
        is_separator_regex=False
    )
    
    chunks = text_splitter.split_text(text=text)
    
    valid_chunks = []
    for chunk in chunks:
        cleaned_chunk = chunk.strip()
        if cleaned_chunk and len(cleaned_chunk) >= min_chunk_length:
            cleaned_chunk = clean_extracted_text(cleaned_chunk)
            if cleaned_chunk and len(cleaned_chunk) >= min_chunk_length:
                valid_chunks.append(cleaned_chunk)
    
    return valid_chunks

def generate_embeddings(texts: list[str], task_type: str) -> list[list[float]]:
    if not texts or not isinstance(texts, list) or not all(isinstance(txt, str) for txt in texts):
        return []
    
    all_embeddings = []
    batch_size = 50
    delay_between_batches = 0.2
    max_retries = 5
    base_retry_delay = 1
    embedding_model = "models/text-embedding-004"

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_number = i // batch_size + 1
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                print(f"Processing batch {batch_number} (attempt {retry_count + 1}/{max_retries}) "
                      f"with {len(batch_texts)} texts...")

                result = client.models.embed_content(
                    model=embedding_model,
                    contents=batch_texts,
                    config=types.EmbedContentConfig(task_type=task_type)
                )

                if result and hasattr(result, 'embeddings') and result.embeddings:
                    batch_embeddings = []
                    for emb_obj in result.embeddings:
                        if hasattr(emb_obj, 'values') and emb_obj.values:
                            batch_embeddings.append(emb_obj.values)
                        else:
                            print(f"Warning: Invalid embedding object in batch {batch_number}")
                    
                    if len(batch_embeddings) == len(batch_texts):
                        all_embeddings.extend(batch_embeddings)
                        print(f"Successfully processed batch {batch_number}")
                        break
                    else:
                        print(f"Mismatch in batch {batch_number}: expected {len(batch_texts)}, got {len(batch_embeddings)}")
                        if retry_count + 1 >= max_retries:
                            return []
                else:
                    print(f"No embeddings returned for batch {batch_number}")
                    if retry_count + 1 >= max_retries:
                        return []
                
                retry_count += 1
                time.sleep(base_retry_delay * (2 ** retry_count))

            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "exhausted" in error_msg:
                    retry_count += 1
                    if retry_count >= max_retries:
                        print(f"Batch {batch_number} failed after max retries due to rate limits")
                        return []
                    sleep_time = base_retry_delay * (2 ** retry_count)
                    print(f"Rate limit hit, waiting {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    print(f"Non-retryable error in batch {batch_number}: {e}")
                    return []

        if i + batch_size < len(texts):
            time.sleep(delay_between_batches)

    if len(all_embeddings) != len(texts):
        print(f"Warning: Embedding count mismatch - expected {len(texts)}, got {len(all_embeddings)}")
        
    return all_embeddings

def store_in_chroma(
    collection_name: str,
    chunks: list[str],
    embeddings: list[list[float]],
    chroma_client
) -> chromadb.api.models.Collection.Collection | None:
    
    if not chroma_client:
        print("Error: ChromaDB client not provided")
        return None
    
    if not collection_name or not collection_name.strip():
        print("Error: Invalid collection name")
        return None
    
    if not chunks or not embeddings:
        print(f"Warning: No chunks or embeddings to store in {collection_name}")
        return None
    
    if len(chunks) != len(embeddings):
        print(f"Error: Chunk count ({len(chunks)}) != embedding count ({len(embeddings)})")
        return None

    try:
        try:
            existing_collection = chroma_client.get_collection(name=collection_name)
            chroma_client.delete_collection(name=collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except:
            pass

        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        chunk_ids = [f"{collection_name}_chunk_{i}" for i in range(len(chunks))]
        
        collection.add(
            embeddings=embeddings,
            documents=chunks,
            ids=chunk_ids
        )
        
        print(f"Successfully stored {len(chunks)} chunks in collection: {collection_name}")
        return collection
        
    except Exception as e:
        print(f"Error storing in ChromaDB collection {collection_name}: {str(e)}")
        return None

def extract_claims_from_hypothesis(hypothesis: str) -> list[str]:
    sentences = sent_tokenize(hypothesis)
    claims = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 20:
            continue
            
        sub_claims = re.split(r'[,;](?=\s*[A-Z])', sentence)
        for sub_claim in sub_claims:
            sub_claim = sub_claim.strip()
            if len(sub_claim) >= 20:
                claims.append(sub_claim)
    
    return claims

def calculate_semantic_similarity(embedding1: list[float], embedding2: list[float]) -> float:
    if not embedding1 or not embedding2:
        return 0.0
    
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def calculate_lexical_overlap(text1: str, text2: str) -> float:
    try:
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        
        tokens1 = set(lemmatizer.lemmatize(word.lower()) for word in word_tokenize(text1) 
                     if word.lower() not in stop_words and word.isalpha())
        tokens2 = set(lemmatizer.lemmatize(word.lower()) for word in word_tokenize(text2) 
                     if word.lower() not in stop_words and word.isalpha())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0.0
    except:
        return 0.0

def compute_information_density(
    hypothesis_text: str,
    collection_name: str,
    gemini_client,
    chroma_client,
    similarity_threshold: float = 0.65,
    lexical_threshold: float = 0.15,
    top_results_per_claim: int = 5,
    evidence_boost_factor: float = 1.2
) -> tuple[float, list[str], str]:

    if not hypothesis_text or not hypothesis_text.strip():
        return 0.0, [], "No hypothesis provided"
    
    if not collection_name or not gemini_client or not chroma_client:
        return 0.0, [], "Missing required parameters"

    try:
        collection = chroma_client.get_collection(name=collection_name)
        if collection.count() == 0:
            return 0.0, [], "Empty literature collection"
    except Exception as e:
        return 0.0, [], f"Cannot access collection: {str(e)}"

    claims = extract_claims_from_hypothesis(hypothesis_text)
    if not claims:
        return 0.0, [], "No valid claims extracted"

    supported_claims = 0
    evidence_details = []
    total_support_score = 0.0
    
    print(f"\nAnalyzing {len(claims)} claims for information density:")

    for claim_idx, claim in enumerate(claims):
        print(f"  Processing claim {claim_idx + 1}: {claim[:60]}...")
        
        claim_embeddings = generate_embeddings([claim], "RETRIEVAL_QUERY")
        if not claim_embeddings or not claim_embeddings[0]:
            print(f"    Failed to generate embedding for claim {claim_idx + 1}")
            continue
        
        claim_embedding = claim_embeddings[0]

        try:
            results = collection.query(
                query_embeddings=[claim_embedding],
                n_results=top_results_per_claim,
                include=['documents', 'distances']
            )
        except Exception as e:
            print(f"    Error querying for claim {claim_idx + 1}: {str(e)}")
            continue

        best_support_score = 0.0
        best_evidence = ""
        
        if results and results.get('documents') and results['documents'][0]:
            for doc_idx, document in enumerate(results['documents'][0]):
                distance = results['distances'][0][doc_idx]
                semantic_sim = 1.0 - distance
                
                lexical_sim = calculate_lexical_overlap(claim, document)
                
                combined_score = (semantic_sim * 0.7) + (lexical_sim * 0.3)
                
                if semantic_sim >= similarity_threshold and lexical_sim >= lexical_threshold:
                    combined_score *= evidence_boost_factor
                
                if combined_score > best_support_score:
                    best_support_score = combined_score
                    best_evidence = document
        
        if best_support_score >= similarity_threshold:
            supported_claims += 1
            total_support_score += best_support_score
            
            evidence_info = (
                f"Claim {claim_idx + 1}: {claim}\n"
                f"Support Score: {best_support_score:.3f}\n"
                f"Evidence: {best_evidence[:200]}..."
            )
            evidence_details.append(evidence_info)
            print(f"    SUPPORTED with score {best_support_score:.3f}")
        else:
            print(f"    NOT SUPPORTED (best score: {best_support_score:.3f})")

    if len(claims) == 0:
        density_score = 0.0
    else:
        base_density = (supported_claims / len(claims)) * 100
        
        if supported_claims > 0:
            avg_support_strength = total_support_score / supported_claims
            strength_multiplier = min(avg_support_strength, 1.0)
            density_score = base_density * strength_multiplier
        else:
            density_score = 0.0

    if density_score >= 80:
        status = "Strongly Supported"
    elif density_score >= 60:
        status = "Well Supported"
    elif density_score >= 40:
        status = "Moderately Supported"
    elif density_score >= 20:
        status = "Partially Supported"
    else:
        status = "Not Supported"
    
    return round(density_score, 2), evidence_details, status

def process_literature_document(
    pdf_path: str,
    collection_name: str,
    chroma_client,
    chunk_size: int = 800,
    overlap_size: int = 160
) -> tuple[bool, str, int | None]:
    
    if not chroma_client:
        return False, "ChromaDB client not provided", None
    
    print(f"\nProcessing literature document: {pdf_path}")
    print(f"Target collection: {collection_name}")
    
    extracted_text = extract_text_from_pdf(pdf_path)
    if not extracted_text:
        return False, f"Failed to extract text from {pdf_path}", None
    
    print(f"Extracted {len(extracted_text)} characters from PDF")
    
    text_chunks = create_text_chunks(
        text=extracted_text,
        chunk_size=chunk_size,
        overlap_size=overlap_size
    )
    
    if not text_chunks:
        return False, "No valid chunks created from text", None
    
    print(f"Created {len(text_chunks)} text chunks")
    
    chunk_embeddings = generate_embeddings(text_chunks, "RETRIEVAL_DOCUMENT")
    if not chunk_embeddings or len(chunk_embeddings) != len(text_chunks):
        return False, "Embedding generation failed", None
    
    print(f"Generated {len(chunk_embeddings)} embeddings")
    
    collection = store_in_chroma(collection_name, text_chunks, chunk_embeddings, chroma_client)
    
    if collection and collection.count() > 0:
        stored_count = collection.count()
        message = f"Successfully processed {pdf_path}. Stored {stored_count} chunks in {collection_name}"
        print(message)
        return True, message, stored_count
    else:
        message = f"Failed to store chunks in ChromaDB for {pdf_path}"
        print(message)
        return False, message, None

if __name__ == '__main__':
    print("=== Literature Processing and Information Density Analysis ===")

    if not CHROMA_CLIENT:
        print("CRITICAL ERROR: ChromaDB client not initialized")
        exit(1)
    
    print(f"Using ChromaDB client: {type(CHROMA_CLIENT)}")

    if 'client' not in globals() or not client:
        print("CRITICAL ERROR: Gemini client not available")
        exit(1)
    
    print(f"Using Gemini client: {type(client)}")

    print("\n[Phase 1: Processing Literature Document]")
    test_pdf_path = "sample_pdfs/2505.09053v1.pdf"
    test_collection_name = "literature_analysis_v2"

    if not os.path.exists(test_pdf_path):
        print(f"CRITICAL ERROR: Test PDF not found at {test_pdf_path}")
        exit(1)

    success, message, items_count = process_literature_document(
        pdf_path=test_pdf_path,
        collection_name=test_collection_name,
        chroma_client=CHROMA_CLIENT,
        chunk_size=600,
        overlap_size=120
    )

    print(f"\nProcessing Result:")
    print(f"Success: {success}")
    print(f"Message: {message}")
    print(f"Items Stored: {items_count}")

    if not success or not items_count:
        print("Document processing failed - skipping density analysis")
        exit(1)

    print(f"\n[Phase 2: Information Density Analysis]")
    
    test_hypotheses = [
        {
            "name": "Reaction-Diffusion Accuracy",
            "text": "In chemical reaction-diffusion networks with small-molecule populations, the accuracy of moment-based approximations for dynamic evolution is contingent on specific relationships between reaction orders, Hill functions, and spatial distribution, where deviations from these relationships lead to increased errors in capturing system behavior."
        },
        {
            "name": "Spatial Pattern Formation",
            "text": "In spatially distributed biochemical networks with small-molecule populations, the emergence of non-local correlations between molecular species, mediated by reaction-diffusion processes and modulated by Hill-type regulatory functions, can lead to the formation of self-organized, dynamically stable spatial patterns that encode and propagate epigenetic information across cellular compartments."
        },
        {
            "name": "Irrelevant Hypothesis",
            "text": "The study of astrophysics is the primary focus of humanity and will revolutionize our understanding of quantum mechanics in biological systems."
        },
        {
            "name": "Thermodynamic Framework",
            "text": "By integrating a non-equilibrium thermodynamic framework with chemical reaction-diffusion networks, the emergence of self-organized criticality in intracellular signaling pathways, specifically those governed by Hill functions, is driven by the system's tendency to maximize entropy production while maintaining structural stability."
        },
        {
            "name": "Concentration Effects",
            "text": "In chemical reaction-diffusion networks, increasing the concentration of reactants will increase the rate of product formation until a saturation point is reached due to limited diffusion and enzyme kinetics."
        }
    ]
    
    analysis_threshold = 0.60
    
    for idx, hypothesis_data in enumerate(test_hypotheses):
        hyp_name = hypothesis_data["name"]
        hyp_text = hypothesis_data["text"]
        
        print(f"\n--- Analyzing Hypothesis {idx + 1}: {hyp_name} ---")
        
        density, evidence, status = compute_information_density(
            hypothesis_text=hyp_text,
            collection_name=test_collection_name,
            gemini_client=client,
            chroma_client=CHROMA_CLIENT,
            similarity_threshold=analysis_threshold,
            lexical_threshold=0.12,
            top_results_per_claim=4
        )
        
        print(f"\nResults for '{hyp_name}':")
        print(f"Information Density: {density}%")
        print(f"Support Status: {status}")
        
        if evidence:
            print(f"Evidence Found ({len(evidence)} pieces):")
            for ev_idx, evidence_item in enumerate(evidence):
                print(f"\n  Evidence {ev_idx + 1}:")
                for line in evidence_item.split('\n'):
                    if line.strip():
                        print(f"    {line}")
        else:
            print("No supporting evidence found above threshold")
        
        print("=" * 60)

    print("\n=== Analysis Complete ===")