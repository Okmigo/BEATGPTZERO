# config.py: Configuration constants for models and parameters
PARAPHRASER_MODEL = "google/flan-t5-large"         # Seq2Seq model for rewriting
DETECTOR_MODEL    = "openai-community/roberta-base-openai-detector"  # Classifier for AI vs human
TOP_P             = 0.9    # Nucleus sampling parameter for diversity
TEMPERATURE       = 0.7    # Temperature for generation randomness
MAX_LENGTH        = 512    # Max tokens for generation
# Semantic similarity model (for potential meaning check; can omit if not needed)
SIMILARITY_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
