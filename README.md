# Chat with Your Documents (Offline Q&A App)

This Streamlit app allows users to upload PDF files and ask questions using a local language model (via Ollama) — all **completely offline**.

### 🔧 Features
- Upload any PDF file
- Ask natural language questions
- Powered by:
  - Sentence Transformers (Embeddings)
  - FAISS (Semantic search)
  - Mistral or LLaMA (via Ollama)
  - Streamlit UI

### 🚀 How to Run

1. Clone the repo
2. Install dependencies:  
   `pip install -r requirements.txt`
3. Start Ollama:  
   `ollama run mistral`
4. Run the app:  
   `streamlit run app.py`

### 🔐 100% private — no data sent to the cloud!

