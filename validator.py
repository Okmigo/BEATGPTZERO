import torch
from sentence_transformers import SentenceTransformer, util
from config import SEMANTIC_THRESHOLD, CUDA_DEVICE

DEVICE = f"cuda:{CUDA_DEVICE}" if CUDA_DEVICE >= 0 and torch.cuda.is_available() else "cpu"
_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to(DEVICE)

def is_semantically_close(orig: str, candidate: str) -> bool:
    emb1, emb2 = _embedder.encode([orig, candidate], convert_to_tensor=True, device=DEVICE)
    score = util.cos_sim(emb1, emb2).item()
    return score >= SEMANTIC_THRESHOLD, score
