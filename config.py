# Universal configuration values
MAX_REWRITE_ATTEMPTS = 5          # retries per request
SEMANTIC_THRESHOLD  = 0.90        # cosine similarity floor
DETECTOR_THRESHOLD  = 0.30        # ‘AI-probability’ ceiling
NUM_BEAMS           = 10          # Pegasus generation beams
NUM_CANDIDATES      = 5           # Pegasus candidates / try
CUDA_DEVICE         = 0           # −1 forces CPU
