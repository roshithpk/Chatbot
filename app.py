
# app.py
import streamlit as st
import pdfplumber
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
import os

# Load model once
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Helper: Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# Helper: Chunk text
def chunk_text(text, chunk_size=300):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Helper: Create FAISS index
def create_faiss_index(chunks):
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

# Helper: Search relevant chunks
def search_chunks(index, question, chunks, k=3):
    q_embed = embedder.encode([question])
    D, I = index.search(np.array(q_embed), k)
    return [chunks[i] for i in I[0]]

# Helper: Query Ollama (local LLM)
def ask_ollama(context, question, model='mistral'):
    prompt = f"""
You are a helpful assistant. Use the following context to answer the question.

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

# Streamlit UI
st.title("📄 Chat with Your Documents (Offline AI)")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    st.success("PDF text extracted successfully.")
    chunks = chunk_text(text)
    index, embeddings = create_faiss_index(chunks)

    question = st.text_input("Ask a question about the document:")

    if question:
        relevant_chunks = search_chunks(index, question, chunks)
        combined_context = "\n\n".join(relevant_chunks)
        with st.spinner("Thinking..."):
            answer = ask_ollama(combined_context, question)
        st.markdown("**Answer:**")
        st.write(answer)
