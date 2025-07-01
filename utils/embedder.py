# utils/embedder.py

from sentence_transformers import SentenceTransformer

def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

