# app.py

import streamlit as st
from utils.pdf_reader import extract_text_from_pdf
from utils.chunker import chunk_text
from qa_engine import embed_chunks, create_faiss_index, search_chunks, ask_ollama

st.title("📄 Chat with Your Documents (Offline, Free, Local)")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    st.success("✅ PDF text extracted.")
    
    chunks = chunk_text(text)
    st.write(f"📚 Document split into {len(chunks)} chunks.")

    embeddings = embed_chunks(chunks)
    index = create_faiss_index(embeddings)

    question = st.text_input("💬 Ask a question based on the document:")

    if question:
        relevant_chunks = search_chunks(question, chunks, index)
        context = "\n\n".join(relevant_chunks)
        with st.spinner("Thinking..."):
            answer = ask_ollama(context, question)
        st.markdown("**🤖 Answer:**")
        st.write(answer)
