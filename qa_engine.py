
# qa_engine.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def embed_chunks(chunks):
    return embedder.encode(chunks)

def create_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index

def search_chunks(question, chunks, index, k=3):
    q_embed = embedder.encode([question])
    D, I = index.search(np.array(q_embed), k)
    return [chunks[i] for i in I[0]]

def ask_ollama(context, question, model='mistral'):
    prompt = f"""
Use the following context to answer the question. Be concise and specific.

Context:
{context}

Question:
{question}

Answer:
"""
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={"model": model, "prompt": prompt, "stream": False}
    )
    return response.json()['response']
