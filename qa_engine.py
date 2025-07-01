# qa_engine.py

import os
import faiss
import numpy as np
import pickle
import requests
from sentence_transformers import SentenceTransformer

# Constants
DATA_DIR = "data"
EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'
os.makedirs(DATA_DIR, exist_ok=True)

# Load the embedding model globally
embedder = SentenceTransformer(EMBED_MODEL_NAME)


# ----------------------------------------
# EMBEDDINGS & INDEX
# ----------------------------------------

def embed_chunks(chunks):
    """Convert text chunks to embeddings using SentenceTransformer."""
    return embedder.encode(chunks)


def create_faiss_index(embeddings):
    """Create FAISS index from given embeddings."""
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index


def save_index(index, chunks, name="index"):
    """Save FAISS index and chunk metadata locally."""
    faiss.write_index(index, os.path.join(DATA_DIR, f"{name}.faiss"))
    with open(os.path.join(DATA_DIR, f"{name}_chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)


def load_index(name="index"):
    """Load FAISS index and chunks if available."""
    try:
        index_path = os.path.join(DATA_DIR, f"{name}.faiss")
        chunks_path = os.path.join(DATA_DIR, f"{name}_chunks.pkl")

        if os.path.exists(index_path) and os.path.exists(chunks_path):
            index = faiss.read_index(index_path)
            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)
            return index, chunks
    except Exception as e:
        print("❌ Failed to load index:", e)

    return None, None


# ----------------------------------------
# SEARCH & ANSWER
# ----------------------------------------

def search_chunks(question, chunks, index, k=3):
    """Find top-k most relevant text chunks for the given question."""
    q_embed = embedder.encode([question])
    D, I = index.search(np.array(q_embed), k)
    return [chunks[i] for i in I[0]]


def ask_ollama(context, question, model='mistral'):
    """Query local LLM (Ollama) with the relevant document context and user question."""
    prompt = f"""
You are an intelligent assistant. Use the following context to answer the user's question clearly and concisely.

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            }
        )
        return response.json()['response']
    except Exception as e:
        print("❌ Error querying Ollama:", e)
        return "⚠️ Error: Could not connect to the local LLM (Ollama). Make sure it’s running."
