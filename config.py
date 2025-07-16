# Model and pipeline configuration

# Paraphraser model (T5-small fine-tuned for paraphrasing)
PARAPHRASER_MODEL = "mrm8488/t5-small-finetuned-quora-for-paraphrasing"
PARAPHRASER_NUM_OUTPUTS = 3  # number of paraphrase candidates to generate

# GPT-2 model for perplexity scoring
GPT2_MODEL = "gpt2"  # or "gpt2-medium" for higher capacity

# Detector model (RoBERTa fine-tuned for GPT-2 output detection)
DETECTOR_MODEL = "openai-community/roberta-base-openai-detector"

# SentenceTransformer model for semantic similarity
SEMANTIC_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"

# Synonym replacement probability (0 = off, up to 1.0)
SYNONYM_PROBABILITY = 0.1

# Similarity threshold for validation (unused here)
SIMILARITY_THRESHOLD = 0.8

# Chunk size (words) for splitting long text
CHUNK_SIZE = 400

# Flask server port (Cloud Run expects $PORT)
PORT = 8080

# Style memory for user tone adaptation
STYLE_MEMORY_ENABLED = True
STYLE_MEMORY_FILE = "user_style_profiles.json"

# AI detection loop parameters
DETECTION_THRESHOLD = 0.30  # GPT-2 score threshold (30%)
DETECTION_MAX_ROUNDS = 3    # Max rewrite attempts to reduce AI score
