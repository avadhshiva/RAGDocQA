import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="SAP Doc Q&A Bot", layout="wide")

st.title("SAP Document Q&A Bot (RAG)")
st.write("Upload SAP documentation (PDF) and ask questions. Powered by local Ollama.")

# Sidebar for configuration
with st.sidebar:
    st.header("Settings")
    ollama_model = st.text_input("Ollama Model Name", value="llama3")
    st.write("Ensure you have pulled the model using `ollama run <model_name>` locally.")

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

uploaded_files = st.file_uploader("Upload SAP PDFs", type="pdf", accept_multiple_files=True)

if st.button("Process Documents"):
    if uploaded_files:
        with st.spinner("Processing documents..."):
            documents = []
            for uploaded_file in uploaded_files:
                # Save uploaded file to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name
                
                # Load PDF
                loader = PyPDFLoader(temp_file_path)
                documents.extend(loader.load())
                os.unlink(temp_file_path) # Clean up temp file
            
            # Split text
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            
            # Create embeddings and vector store
            embeddings = OllamaEmbeddings(model=ollama_model)
            vector_store = FAISS.from_documents(splits, embeddings)
            st.session_state.vector_store = vector_store
            
            st.success("Documents processed successfully!")
    else:
        st.warning("Please upload at least one PDF.")

# Q&A Section
if st.session_state.vector_store is not None:
    st.subheader("Ask Questions")
    user_query = st.text_input("What would you like to know about the SAP documentation?")
    
    if user_query:
        with st.spinner("Generating answer..."):
            try:
                llm = OllamaLLM(model=ollama_model)
                
                prompt = ChatPromptTemplate.from_template("""
                You are a helpful SAP expert assistant. Answer the question based ONLY on the following context from SAP documentation.
                If you don't know the answer based on the context, say "I cannot find the answer in the provided documents."
                
                Context:
                {context}
                
                Question:
                {input}
                
                Answer:
                """)
                
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                
                response = retrieval_chain.invoke({"input": user_query})
                
                st.write("### Answer")
                st.write(response["answer"])
                
                with st.expander("View Source Documents"):
                    for i, doc in enumerate(response["context"]):
                        st.write(f"**Source {i+1}** (Page {doc.metadata.get('page', 'Unknown')}):")
                        st.write(doc.page_content)
            except Exception as e:
                st.error(f"Error generating answer. Make sure Ollama is running and the model '{ollama_model}' is pulled. Details: {e}")
