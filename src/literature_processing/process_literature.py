from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
import nltk
import os
from src.main import *
import chromadb
import time

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)




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
        CHROMA_CLIENT = chromadb.Client() # In-memory fallback
        print("ChromaDB in-memory client initialized.")
    except Exception as e_mem:
        print(f"Fatal Error: Could not initialize in-memory ChromaDB client either: {e_mem}.")
        CHROMA_CLIENT = None



def extract_text_pdf(pdf):
    """
    Returns
        text extracted from pdf 
    
    """
    pdf_text = ""
    try:
        reader = PdfReader(str(pdf))
        for page in reader.pages:
            page_txt = page.extract_text()
            if page_txt:
                pdf_text += page_txt +'\n'
    except Exception as e:
        print(f"Error occured during PDF text extraction : {str(e)}")
        return ""
    except FileNotFoundError:
        print(f"Error: PDF file not found at path: {pdf}")
        return ""
    return pdf_text.strip()


def chunk_txt(
    txt: str ,
    chunk_size_chars: int = 1000,
    chunk_overlap_chars: int = 100,
    custom_separators : list[str] = None,
    min_chunk_len : int = 10
    ) -> list[str]:
    
    """
    Chunks text using LangChain's RecursiveCharacterTextSplitter.

    Args:
        txt: The text content to be chunked.
        chunk_size_chars: The target maximum size of each chunk in characters.
        chunk_overlap_chars: The number of characters of overlap between consecutive chunks.
        custom_separators: Optional list of custom separators to use. If None,
                           defaults from RecursiveCharacterTextSplitter are used
                           (typically ["\n\n", "\n", " ", ""]).

    Returns:
        A list of text chunks. Returns an empty list if the input text is empty
        or if chunking results in no valid chunks.
    
    """
    if not txt or not txt.strip():
        return []

    if custom_separators is None:
        separators_to_use = ["\n\n", "\n", ". ", "? ", "! ", " ", "", "\u200B"]
    else:
        separators_to_use = custom_separators

    text_splitter = RecursiveCharacterTextSplitter(
        separators= separators_to_use,
        chunk_size = chunk_size_chars,
        chunk_overlap  = chunk_overlap_chars,
        length_function = len,
        is_separator_regex=False
    )    
    chunks = text_splitter.split_text(text=txt)
    
    return [chunk for chunk in chunks if chunk.strip() and len(chunk.strip()) > min_chunk_len]




def get_embeddings(texts : list[str], task_type: str ) -> list[float]:
    """
    Generates embeddings for a list of text strings using the Gemini embeddings

    Args:
        texts: A list of strings, where each string is a text chunk.    
        task_type: The task type for the embedding (e.g., "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY").


    Returns:
        A list of lists of floats, where each inner list is an embedding
        vector for the corresponding text chunk. Returns an empty list
        if the model is not loaded, input is empty, or an error occurs.
    """
    if not texts or not isinstance(texts,list) or not all (isinstance(txt , str) for txt in texts):
        return []
    
   
    all_embeddings_values = []
    batch_size = 100
    delay_between_batches_seconds = 0.1
    max_retries_per_batch = 3
    initial_retry_delay_seconds = 2
    embedding_model_name = "models/text-embedding-004"

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        current_batch_num = i // batch_size + 1
        retries = 0
        
        while retries < max_retries_per_batch:
            try:
                print(f"Processing embedding batch {current_batch_num} (attempt {retries + 1}/{max_retries_per_batch}) "
                      f"for {len(batch_texts)} texts, model: {embedding_model_name}, task: {task_type}...")

                result = client.models.embed_content(
                    model=embedding_model_name,
                    contents=batch_texts,
                    config=types.EmbedContentConfig(task_type=task_type)
                )

                if result and hasattr(result, 'embeddings') and result.embeddings:
                    batch_processed_count = 0
                    for emb_object in result.embeddings:
                        if hasattr(emb_object, 'values') and emb_object.values:
                            all_embeddings_values.append(emb_object.values)
                            batch_processed_count += 1
                        else:
                            print(f"Warning: ContentEmbedding object in batch {current_batch_num} missing 'values' or has empty values.")
                    
                    if batch_processed_count != len(batch_texts):
                        print(f"Warning: Mismatch in expected ({len(batch_texts)}) vs successfully processed ({batch_processed_count}) embeddings for batch {current_batch_num}.")
                    
                    print(f"Successfully processed batch {current_batch_num}.")
                    break 
                else:
                    print(f"Warning: No 'embeddings' in result or embeddings list empty for batch {current_batch_num} (attempt {retries + 1}).")
                    if retries + 1 >= max_retries_per_batch:
                        print(f"Batch {current_batch_num} failed after max retries due to empty/invalid result.")
                        return [] 
                    retries +=1 
                    time.sleep(initial_retry_delay_seconds * (2 ** retries)) 

            except Exception as e:
                error_message = str(e).lower()
                if "resource has been exhausted" in error_message or "429" in error_message:
                    print(f"Rate limit hit (429) on batch {current_batch_num}, attempt {retries + 1}. Retrying after delay...")
                    retries += 1
                    if retries >= max_retries_per_batch:
                        print(f"Batch {current_batch_num} failed after max retries due to rate limits.")
                        return [] 
                    sleep_time = initial_retry_delay_seconds * (2 ** (retries -1)) 
                    print(f"Waiting for {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                else:
                    print(f"Non-retryable error occurred during embedding batch {current_batch_num}: {e}")
                    return [] 

        if len(all_embeddings_values) != (i + len(batch_texts)) and retries >= max_retries_per_batch :
             print(f"Batch {current_batch_num} ultimately failed to process all its texts after retries.")
             return [] 

        if i + batch_size < len(texts): 
            print(f"Waiting {delay_between_batches_seconds:.2f}s before next batch...")
            time.sleep(delay_between_batches_seconds)

    if len(all_embeddings_values) != len(texts):
        print(f"Critical Warning: Final embedding count ({len(all_embeddings_values)}) "
              f"does not match input text count ({len(texts)}).")
        
    return all_embeddings_values

def store_chunks_in_chroma(
    collection_name : str ,
    chunks : list[str],
    embeddings : list[float],
    chroma_db_client
    )-> chromadb.api.models.Collection.Collection | None:
    """
    Stores text chunks and their corresponding embeddings in a ChromaDB collection.
    Deletes and recreates the collection if it already exists for a fresh start.

    Args:
        collection_name: The name for the ChromaDB collection.
        chunks: A list of text strings (the document chunks).
        embeddings: A list of lists of floats (the embedding vectors for the chunks).
        chroma_db_client: The initialized ChromaDB client instance.

    Returns:
        The ChromaDB collection object if successful, otherwise None.
    """
    if not chroma_db_client:
        print("Error: ChromaDB client not provided to store_chunks_in_chroma.")
        return None
    if not collection_name or not collection_name.strip():
        print("Error: Invalid collection name provided.")
        return None
    if not chunks or not embeddings:
        print(f"Warning for '{collection_name}': No chunks or embeddings provided to store.")
        return None
    if len(chunks)!=len(embeddings):
        print(f"For Collection - {collection_name} ': Mismatch - {len(chunks)} chunks vs {len(embeddings)} embeddings.")

    try:

        try:
            chroma_db_client.get_collection(name=chroma_db_client)
            chroma_db_client.delete_collection(name=collection_name)
            print(f"Info: Existing collection '{collection_name}' deleted for refresh.")
        except:
            pass

        collection = chroma_db_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Specifies the distance metric
            )
        ids = [f"{collection_name}_doc_{i}" for i in range(len(chunks))]

        collection.add(
            embeddings=embeddings,
            documents= chunks,
            ids=ids
            # addd metadata
        )
        print(f"Info: Successfully added {len(chunks)} items to ChromaDB collection '{collection_name}'.")
        return collection
    except Exception as e:
        print(f"Error during ChromaDB operation for collection '{collection_name}': {str(e)}")
        return None

from nltk.tokenize import sent_tokenize

def calculate_information_density(
    hypothesis_text: str,
    collection_name: str,
    gemini_client_for_claims,
    chroma_client_to_query,
    similarity_threshold: float = 0.70,
    top_n_results_per_claim: int = 3
) -> tuple[float, list[str], str]:

    if not hypothesis_text or not hypothesis_text.strip():
        return 0.0, [], "Not Applicable (No Hypothesis Provided)"
    if not collection_name:
        return 0.0, [], "Error: Collection name not provided"
    if not gemini_client_for_claims or not chroma_client_to_query:
        return 0.0, [], "Error: Gemini or ChromaDB client not provided"

    try:
        collection = chroma_client_to_query.get_collection(name=collection_name)
        if collection.count() == 0:
             return 0.0, [], "Error: Literature collection is empty"
    except Exception as e:
        return 0.0, [], f"Error: Cannot access collection '{collection_name}'. {str(e)}"

    claims = sent_tokenize(hypothesis_text)
    valid_claims = [c.strip() for c in claims if c.strip()]
    if not valid_claims:
        return 0.0, [], "Not Applicable (Hypothesis unparsable or empty)"

    supported_claims_count = 0
    retrieved_citations_details = []
    print(f"\n  Calculating Information Density for {len(valid_claims)} claim(s):")

    for i, claim_text in enumerate(valid_claims):
        print(f"    Processing Claim {i+1}/{len(valid_claims)}: \"{claim_text[:80]}...\"")
        
        claim_embeddings_list = get_embeddings( # Assumes get_embeddings uses global 'client'
            texts=[claim_text],
            task_type="RETRIEVAL_QUERY"
        ) 
        
        if not claim_embeddings_list or not claim_embeddings_list[0]:
            print(f"      Failed to generate embedding for claim {i+1}.")
            continue
        
        claim_embedding_vector = claim_embeddings_list[0]

        try:
            results = collection.query(
                query_embeddings=[claim_embedding_vector],
                n_results=top_n_results_per_claim,
                include=['documents', 'distances']
            )
        except Exception as e_query:
            print(f"      Error querying ChromaDB for claim {i+1}: {str(e_query)}")
            continue

        claim_supported_this_round = False
        if results and results.get('documents') and results['documents'][0]:
            for doc_idx in range(len(results['documents'][0])):
                distance = results['distances'][0][doc_idx]
                similarity_score = 1.0 - distance 

                if similarity_score >= similarity_threshold:
                    supported_claims_count += 1
                    evidence_snippet = results['documents'][0][doc_idx]
                    citation_info = (
                        f"Claim {i+1}: \"{claim_text}\"\n"
                        f"  Evidence (Similarity: {similarity_score:.3f}): "
                        f"\"{evidence_snippet[:250]}...\""
                    )
                    retrieved_citations_details.append(citation_info)
                    claim_supported_this_round = True
                    print(f"      Claim {i+1} SUPPORTED with similarity {similarity_score:.3f}.")
                    break 
            
            if not claim_supported_this_round:
                 print(f"      Claim {i+1} NOT SUPPORTED by top results (threshold: {similarity_threshold}). Closest similarity: {1.0 - results['distances'][0][0]:.3f} (if results).")
        else:
            print(f"      Claim {i+1} NOT SUPPORTED (no relevant documents found).")
            
    info_density_score_percent = (supported_claims_count / len(valid_claims)) * 100 if valid_claims else 0.0

    status = "Not Supported"
    if info_density_score_percent >= 75: status = "Strongly Supported"
    elif info_density_score_percent >= 50: status = "Moderately Supported"
    elif info_density_score_percent > 10: status = "Partially Supported"
    
    return round(info_density_score_percent, 2), list(set(retrieved_citations_details)), status

def process_pdf_and_store_embeddings(
    pdf : str,
    collection_name: str,
    chroma_client,
    chunk_sz : int = 1000,
    ):
    """
    Orchestrates the entire pipeline:
    1. Extracts text from the PDF.
    2. Chunks the extracted text.
    3. Generates embeddings for the chunks using Gemini.
    4. Stores the chunks and embeddings in ChromaDB.
     Args:
        pdf_file_path: Path to the PDF file.
        collection_name: Name for the ChromaDB collection.
        gemini_client_instance: Initialized Gemini API client.
        chroma_client_instance: Initialized ChromaDB client.
        chunk_sz: Target chunk size in characters.
        
    Returns:
        A tuple: (success_status, message, number_of_items_stored_or_None).
    """
    if not chroma_client:
        return False, "Error: ChromaDB client not provided.", None
    
    # 1. Extract text
    print(f"\nProcessing PDF: '{pdf}' for collection: '{collection_name}'")
    extracted_txt = extract_text_pdf(pdf=pdf)
    if not extracted_txt:
        return False, f"Failed to extract text from '{pdf}' or PDF is empty.", None
    print(f"Successfully extracted text from '{pdf}' ({len(extracted_txt)} chars).")

    # 2. Chunk Text
    txt_chunks = chunk_txt(txt=extracted_txt,chunk_size_chars=chunk_sz,chunk_overlap_chars= 100)
    if not txt_chunks:
        return False, "Text chunking resulted in no usable chunks.", None
    print(f"Text chunked into {len(txt_chunks)} parts.")

     # 3. Generate Embeddings
    chunk_embeddings = get_embeddings(texts = txt_chunks,task_type="RETRIEVAL_DOCUMENT")
    if not chunk_embeddings or len(chunk_embeddings) != len(txt_chunks):
        return False, "Embedding generation failed or mismatched with chunks.", None
    print(f"Generated {len(chunk_embeddings)} embeddings.")

    # 4. Store in ChromaDB
    created_collection = store_chunks_in_chroma(collection_name, txt_chunks, chunk_embeddings, chroma_client)

    if created_collection and created_collection.count() > 0:
        num_stored = created_collection.count()
        msg = f"Successfully processed '{pdf}'. {num_stored} items stored in collection '{collection_name}'."
        print(msg)
        return True, msg, num_stored
    elif created_collection and created_collection.count() == 0:
        msg = f"Processed '{pdf}', but 0 items were stored in ChromaDB. Check chunk content or size."
        print(msg)
        return False, msg, 0
    else:
        msg = f"Failed to store chunks and embeddings in ChromaDB for '{pdf}'."
        print(msg)
        return False, msg, None
    







if __name__ == '__main__':
    print("--- Comprehensive Test Suite: PDF Processing & Information Density ---")

    if 'CHROMA_CLIENT' not in globals() or CHROMA_CLIENT is None:
        print("CRITICAL ERROR: CHROMA_CLIENT is not initialized. Aborting.")
        exit()
    else:
        print(f"Using CHROMA_CLIENT: {type(CHROMA_CLIENT)}")

    gemini_client_is_valid = False
    if 'client' in globals() and client is not None:
        try: 
            if hasattr(client, 'models') and hasattr(client.models, 'embed_content'):
                 print(f"Using Global Gemini client from src.main: {type(client)}")
                 gemini_client_is_valid = True
            else:
                print("Warning: Global 'client' does not have expected 'models.embed_content' structure.")
        except Exception as e_client_check:
            print(f"Warning: Error checking global 'client': {e_client_check}")
    
    if not gemini_client_is_valid:
        print("CRITICAL WARNING: Global Gemini 'client' is not confirmed valid for embeddings.")

    print("\n[Test 1: Processing Sample PDF and Storing in ChromaDB]")
    pdf_to_process_path = "sample_pdfs/2505.09053v1.pdf" 
    main_test_collection_name = "density_test_collection_v1" 

    if not os.path.exists(pdf_to_process_path):
        print(f"  CRITICAL FAIL: Test PDF '{pdf_to_process_path}' not found. Cannot proceed with tests.")
        exit()

    print(f"Attempting to process PDF: '{pdf_to_process_path}'")
    success_proc, msg_proc, items_stored = process_pdf_and_store_embeddings(
        pdf=pdf_to_process_path, 
        collection_name=main_test_collection_name,
        chroma_client=CHROMA_CLIENT,
        chunk_sz=700 
    )

    print(f"\n  --- PDF Processing Result for '{pdf_to_process_path}' ---")
    print(f"  Success: {success_proc}")
    print(f"  Message: {msg_proc}")
    print(f"  Items Stored: {items_stored}")

    if not success_proc or items_stored is None or items_stored == 0:
        print("  FAIL: PDF Processing step failed or stored no items. Information Density test will be skipped.")
    else:
        print("  PASS: PDF Processing step successful.")
        try:
            verify_collection = CHROMA_CLIENT.get_collection(name=main_test_collection_name)
            print(f"  Verification: Collection '{main_test_collection_name}' has {verify_collection.count()} items.")
        except Exception as e_verify:
            print(f"  Verification ERROR: Could not retrieve collection: {e_verify}")

        print(f"\n[Test 2: Calculating Information Density against '{main_test_collection_name}']")
        
        sample_hypotheses_for_density_test = [
            {
                "text": "In chemical reaction-diffusion networks with small-molecule populations, the accuracy of moment-based approximations for dynamic evolution is contingent on specific relationships between reaction orders, Hill functions, and spatial distribution, where deviations from these relationships lead to increased errors in capturing system behavior.",
            },
            {
                "text": " In spatially distributed biochemical networks with small-molecule populations, the emergence of non-local correlations between molecular species, mediated by reaction-diffusion processes and modulated by Hill-type regulatory functions, can lead to the formation of self-organized, dynamically stable spatial patterns that encode and propagate epigenetic information across cellular compartments, influencing long-term cellular fate decisions.",
            },
            {
                "text": "The study of astrophysics is the primary focus of humanity", 
            },
            {
                "text": " By integrating a non-equilibrium thermodynamic framework with chemical reaction-diffusion networks, we hypothesize that the emergence of self-organized criticality in intracellular signaling pathways, specifically those governed by Hill functions, is driven by the system's tendency to maximize entropy production while maintaining structural stability, leading to scale-free dynamics and enhanced sensitivity to perturbations, which can be mathematically described by extending the moment-based approach to incorporate fluctuation theorems and irreversible thermodynamics.",
            },
            {
                "text": " In chemical reaction-diffusion networks, increasing the concentration of reactants will increase the rate of product formation until a saturation point is reached due to limited diffusion."
            }
        ]
        
        test_similarity_threshold_for_density = 0.60 

        for idx, hyp_data in enumerate(sample_hypotheses_for_density_test):
            hypothesis = hyp_data["text"]
            print(f"\n  Testing Hypothesis {idx + 1}: \"{hypothesis}\"")
            
            density, citations, prov_status = calculate_information_density(
                hypothesis_text=hypothesis,
                collection_name=main_test_collection_name,
                gemini_client_for_claims=client, 
                chroma_client_to_query=CHROMA_CLIENT,
                similarity_threshold=test_similarity_threshold_for_density,
                top_n_results_per_claim=2 
            )
            
            print(f"    -----------------------------------------------------")
            print(f"    Hypothesis: \"{hypothesis}\"")
            print(f"    Information Density: {density}%")
            print(f"    Provability Status: {prov_status}")
            if citations:
                print(f"    Citations Found ({len(citations)}):")
                for cit_idx, citation_detail in enumerate(citations):
                    indented_citation = citation_detail.replace('\n', '\n          ') # Replace newline with newline + indent
                    print(f"      Citation {cit_idx+1}:\n          {indented_citation}")
            else:
                print("    No citations found meeting the threshold.")
            print(f"    -----------------------------------------------------")

    print("\n--- Comprehensive Test Suite Complete ---")