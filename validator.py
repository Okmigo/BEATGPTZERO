from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import config

# Load a semantic similarity model
semantic_model = SentenceTransformer(config.SEMANTIC_MODEL)

def semantic_similarity(text1, text2):
    emb1 = semantic_model.encode(text1, convert_to_tensor=False)
    emb2 = semantic_model.encode(text2, convert_to_tensor=False)
    sim = cosine_similarity([emb1], [emb2])[0][0]
    return float(sim)
