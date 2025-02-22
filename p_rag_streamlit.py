import os
import faiss
import time
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Constants
FAISS_STORE_PATH = "faiss_store"
DATA_DIR = "./extracted_sec_text"

def load_embeddings():
    """Load Google Generative AI Embeddings securely from Streamlit Secrets"""
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY")
    )

def load_faiss_index():
    """Load FAISS index or build it dynamically"""
    if os.path.exists(FAISS_STORE_PATH):
        try:
            return FAISS.load_local(FAISS_STORE_PATH, load_embeddings(), allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"⚠️ Error loading FAISS: {e}")
    return None

def vector_embedding(session):
    """Build FAISS index if missing"""
    st.warning("🛠️ Building FAISS index. This may take a few minutes...")

    embeddings = load_embeddings()
    loader = DirectoryLoader(DATA_DIR, glob="*.txt", loader_cls=TextLoader)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    chunks = text_splitter.split_documents(docs)

    session["vectors"] = FAISS.from_documents(chunks, embeddings)
    session["vectors"].save_local(FAISS_STORE_PATH)

    st.success("✅ FAISS index created successfully!")
