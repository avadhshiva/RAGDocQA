# SAP Document Q&A Bot (RAG)

This project is a Retrieval-Augmented Generation (RAG) based chatbot designed to answer questions from SAP documentation (e.g., SAP API Help, user manuals). It runs entirely locally using open-source tools.

## Architecture

*   **UI Framework:** Streamlit
*   **RAG Pipeline Framework:** LangChain
*   **Vector Store:** FAISS (Facebook AI Similarity Search)
*   **LLM & Embeddings:** Local models (e.g., Llama 3) served via Ollama

### How it Works
1.  **Document Ingestion:** Users upload SAP documentation in PDF format via the Streamlit UI.
2.  **Chunking:** The PDFs are parsed, and the text is split into smaller, overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`.
3.  **Embedding:** Text chunks are converted into vector embeddings using Ollama's local embedding models.
4.  **Vector Storage:** Embeddings are stored locally in a FAISS vector database for fast similarity search.
5.  **Retrieval:** When a user asks a question, the query is embedded, and the top most relevant document chunks are retrieved from FAISS.
6.  **Generation:** The retrieved chunks and the user's query are passed as context to a local LLM (via Ollama). The LLM generates a grounded response based solely on the provided SAP documentation.

## Prerequisites

1.  Python 3.9+
2.  [Ollama](https://ollama.com/) installed and running locally.
3.  Pull the required model(s) in Ollama:
    ```bash
    ollama run llama3
    ```

## Installation & Setup

1.  Navigate to the project directory:
    ```bash
    cd sap_doc_qa_bot
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    
    # On Windows:
    venv\Scripts\activate
    
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the App

1.  Make sure Ollama is running in the background.
2.  Start the Streamlit application:
    ```bash
    streamlit run app.py
    ```
3.  Open your browser to the URL provided by Streamlit (usually `http://localhost:8501`).
4.  Upload SAP PDF documents, click "Process Documents", and start asking questions!

## Value Addition
This project demonstrates an understanding of enterprise AI requirements, specifically the critical need for **data grounding** using Retrieval-Augmented Generation (RAG). By running locally via Ollama, it ensures strict data privacy, which is absolutely paramount when handling proprietary enterprise documentation like SAP manuals.
