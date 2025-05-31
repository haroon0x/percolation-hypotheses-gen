INCLUDED_SCIENTIFIC_ENTITY_LABELS = {
    # Core Chemical & Biological Entities
    "CHEMICAL", 
    "COMPOUND", # Often used by chemical NER
    "SIMPLE_CHEMICAL", # Some models might use this
    "DRUG", 
    "PHARMACOLOGICAL_SUBSTANCE",
    "GENE", 
    "GENE_OR_GENE_PRODUCT", # Common in BioNER
    "PROTEIN", 
    "PEPTIDE",
    "DNA", 
    "RNA", 
    "NUCLEOTIDE",
    "AMINO_ACID",
    "ENZYME",
    "RECEPTOR",
    "ANTIBODY",
    "HORMONE",

    # Disease & Clinical
    "DISEASE", 
    "DISORDER", 
    "SYNDROME", 
    "SYMPTOM", 
    "MEDICAL_CONDITION",
    "PATHOLOGY",

    # Organisms & Anatomy
    "SPECIES", 
    "ORGANISM", 
    "TAXON", # Taxonomic ranks
    "CELL", 
    "CELL_TYPE", 
    "CELL_LINE", 
    "TISSUE", 
    "ORGAN",
    "ANATOMY", # General anatomical parts
    "ANATOMICAL_STRUCTURE", 
    "BODY_PART", 
    "SYSTEM" # e.g., nervous system, endocrine system

    # Processes & Functions
    "PATHWAY", 
    "BIOLOGICAL_PROCESS", 
    "PHYSIOLOGICAL_PROCESS",
    "MOLECULAR_FUNCTION", 
    "CELLULAR_COMPONENT", # Gene Ontology term
    "MECHANISM",
    "SIGNALING_PATHWAY",

    # Methods & Concepts
    "METHOD", 
    "TECHNIQUE", 
    "ASSAY", 
    "EXPERIMENTAL_FACTOR",
    "DIAGNOSTIC_PROCEDURE",
    "THERAPEUTIC_PROCEDURE",
    "RESEARCH_ACTIVITY",
    "MODEL", # e.g., computational model, animal model
    "PHENOMENON", # Observable events
    "CONCEPT", # Abstract ideas or principles if model supports
    "THEORY",
    "PARAMETER", # Quantitative measures
    "VARIABLE", # Experimental variables
    "DATA", # e.g., sequence data, expression data

    # Might be relevant depending on the scientific domain
    "MATERIAL", 
    "SUBSTRATE",
    "DEVICE", # e.g., medical device
    "EQUIPMENT",
    "SOFTWARE" # e.g., bioinformatics tools
}
EXCLUDED_GENERAL_NER_LABELS = {
    "CARDINAL", "DATE", "MONEY", "ORDINAL", "PERCENT", "QUANTITY", "TIME",
    "LANGUAGE", "LAW", "LOC", "GPE", "NORP", "ORG", "PERSON", "WORK_OF_ART", "FAC", "EVENT","PERSON"
}