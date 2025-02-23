# import os
# import time
# import re
# import json
# import streamlit as st
# from bs4 import BeautifulSoup
# from dotenv import load_dotenv
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.document_loaders import DirectoryLoader, TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from groq import Groq

# # Load environment variables
# load_dotenv()

# # Configuration
# FAISS_STORE_PATH = "update_faiss_store_finance"  # Combined FAISS index location
# DATA_DIR = "./extracted_sec_text_test"
# EMBEDDED_FILES_PATH = "update_embedded_files.txt"
# URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')

# # Groq API Key (make sure it's set in your .env file as GROQ_API_KEY)
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# # --------------------------- Helper Functions ---------------------------
# def load_embedded_files():
#     if os.path.exists(EMBEDDED_FILES_PATH):
#         with open(EMBEDDED_FILES_PATH, "r") as f:
#             return set(line.strip() for line in f if line.strip())
#     return set()

# def save_embedded_files(new_files):
#     existing = load_embedded_files()
#     updated = existing.union(set(new_files))
#     with open(EMBEDDED_FILES_PATH, "w") as f:
#         f.write("\n".join(updated))

# def save_faiss_index(vectorstore):
#     vectorstore.save_local(FAISS_STORE_PATH)
#     st.success("💾 Combined vector store updated")

# def load_faiss_index():
#     if os.path.exists(FAISS_STORE_PATH):
#         try:
#             embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#             vector_store = FAISS.load_local(FAISS_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
#             st.success("✅ Successfully loaded FAISS index.")
#             return vector_store
#         except Exception as e:
#             st.error(f"⚠️ Error loading FAISS index: {e}")
#     else:
#         st.error(f"❌ FAISS index not found at {FAISS_STORE_PATH}.")
#     return None

# def financial_preprocessor(text):
#     text = BeautifulSoup(text, "html.parser").get_text()
#     text = URL_PATTERN.sub('', text)
#     return text.strip()

# def financial_chunking(text):
#     text_lower = text.lower()
#     if any(keyword in text_lower for keyword in ["balance sheet", "income statement", "cash flow"]):
#         return 1024
#     if len(re.findall(r'\$\d+', text)) > 3:
#         return 768
#     return 512

# # --------------------------- Rapid Embedding Function ---------------------------
# def rapid_embedding(vectorstore, texts, metadatas, max_retries=3, max_wait=300):
#     attempt = 0
#     while attempt < max_retries:
#         try:
#             vectorstore.add_texts(texts, metadatas=metadatas)
#             return True
#         except Exception as e:
#             error_str = str(e)
#             if "429" in error_str or "RATE_LIMIT_EXCEEDED" in error_str:
#                 wait_time = (2 ** attempt) * 5
#                 if wait_time > max_wait:
#                     st.error(f"Wait time {wait_time} exceeds max_wait {max_wait}. Skipping embedding.")
#                     return False
#                 st.warning(f"Rate limit exceeded, waiting {wait_time}s (Attempt {attempt+1}/{max_retries})")
#                 time.sleep(wait_time)
#                 attempt += 1
#             else:
#                 st.error(f"Error in rapid_embedding: {e}")
#                 return False
#     return False

# # --------------------------- Sequential File Processing ---------------------------
# def process_files_sequentially(new_files):
#     embedded_files = load_embedded_files()
#     embedded_count = 0
#     skipped_files = []
#     for idx, file_path in enumerate(new_files):
#         if file_path in embedded_files:
#             continue
#         st.write(f"🔨 Processing: {os.path.basename(file_path)}")
#         start_time = time.time()
#         try:
#             loader = TextLoader(file_path)
#             doc = loader.load()[0]
#             doc.page_content = financial_preprocessor(doc.page_content)
#             chunk_size = financial_chunking(doc.page_content)
#             text_splitter = RecursiveCharacterTextSplitter(
#                 chunk_size=chunk_size,
#                 chunk_overlap=int(chunk_size * 0.2),
#                 separators=["\n\n", "\n", r"(?<=\. )", " "]
#             )
#             chunks = text_splitter.split_documents([doc])
#             if chunks:
#                 success = rapid_embedding(session["vectors"], [c.page_content for c in chunks],
#                                           [c.metadata for c in chunks])
#                 if success:
#                     st.success(f"✅ Embedded {len(chunks)} chunks in {time.time()-start_time:.2f}s")
#                     embedded_files.add(file_path)
#                     save_embedded_files([file_path])
#                     embedded_count += 1
#                 else:
#                     st.error(f"❌ Skipping {file_path} due to rate limits.")
#                     skipped_files.append(file_path)
#                 if idx < len(new_files)-1:
#                     st.info("⏳ Cooling period: 90s")
#                     time.sleep(90)
#         except Exception as e:
#             st.error(f"❌ Failed to process {file_path}: {str(e)}")
#             with open("processing_errors.log", "a") as f:
#                 f.write(f"{file_path}: {str(e)}\n")
#             skipped_files.append(file_path)
#     return {"total_files": len(new_files), "embedded_count": embedded_count, "skipped_files": skipped_files}

# # --------------------------- Combined Embedding Workflow ---------------------------
# def vector_embedding(session):
#     # Load existing combined index if available
#     existing_vectors = load_faiss_index()
#     session["embeddings"] = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     if existing_vectors:
#         session["vectors"] = existing_vectors
#         st.write("🔄 Checking for new files...")
#     else:
#         st.write("🆕 No combined index found. Creating new vector store...")
#         session["vectors"] = FAISS.from_documents([], session["embeddings"])
#     try:
#         loader = DirectoryLoader(DATA_DIR, loader_cls=TextLoader)
#         docs = loader.load()
#         current_files = {doc.metadata["source"] for doc in docs}
#         new_files = [f for f in current_files if f not in load_embedded_files()]
#         if new_files:
#             st.write(f"Found {len(new_files)} new files to process")
#             results = process_files_sequentially(new_files)
#             save_faiss_index(session["vectors"])
#             st.write("-------------------------------------")
#             st.write(f"New files processed: {results['embedded_count']}")
#             st.write(f"Failed files: {len(results['skipped_files'])}")
#             st.write("-------------------------------------")
#             if results["skipped_files"]:
#                 retry = st.text_input("Retry failed files? (y/n): ", value="n")
#                 if retry.lower() == "y":
#                     process_files_sequentially(results["skipped_files"])
#                     save_faiss_index(session["vectors"])
#         else:
#             st.write("✅ All files already embedded")
#     except Exception as e:
#         st.error(f"❌ Error loading directory: {str(e)}")

# # --------------------------- Direct Groq Query Function ---------------------------
# def query_llm_groq(question, retriever):
#     try:
#         relevant_docs = retriever.get_relevant_documents(question)
#         retrieved_text = "\n\n".join([doc.page_content for doc in relevant_docs])
#         client = Groq(api_key=GROQ_API_KEY)
#         response = client.chat.completions.create(
#             model="mixtral-8x7b-32768",
#             messages=[{
#                 "role": "user",
#                 "content": (
#                     "You are a seasoned financial analyst AI. Provide concise, data-driven, and accurate answers using only the provided context. "
#                     "Your answer should be clear and actionable, highlighting key financial metrics where applicable.\n\n"
#                     f"Retrieved Context:\n{retrieved_text}\n\nQuestion: {question}"
#                 )
#             }],
#             temperature=0.3,
#             max_tokens=512,
#             top_p=1,
#             stream=False,
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         return f"❌ Groq API Error: {str(e)}"

# # --------------------------- Streamlit UI ---------------------------
# st.title("Financial RAG System (Combined Index)")
# st.write("This system retrieves and processes financial documents to provide concise, data-driven answers using the Groq API.")

# # Sidebar: Company Selection (for context tagging)
# st.sidebar.header("Select Company")
# def list_companies():
#     companies = set()
#     for file_name in os.listdir(DATA_DIR):
#         if file_name.endswith(".txt"):
#             base = os.path.basename(file_name)
#             m = re.match(r"^(.*?)_", base)
#             if m:
#                 companies.add(m.group(1))
#             else:
#                 companies.add(os.path.splitext(base)[0])
#     return sorted(companies)
# companies = list_companies()
# selected_company = st.sidebar.selectbox("Company", companies)

# # Sidebar: Query History
# if "queries" not in st.session_state:
#     st.session_state["queries"] = []
# st.sidebar.markdown("### Query History")
# for idx, q in enumerate(st.session_state["queries"], 1):
#     st.sidebar.write(f"{idx}. {q}")

# # Main: Query Input
# question_input = st.text_input("Enter your financial question:")

# if st.button("Submit Query"):
#     if not question_input:
#         st.warning("Please enter a question.")
#     else:
#         final_query = f"[Company: {selected_company}] {question_input}"
#         st.session_state["queries"].append(final_query)
#         with st.spinner("Loading combined vector store..."):
#             vector_store = load_faiss_index()
#         if vector_store is None:
#             st.error("Vector store not available. Please run the embedding process first.")
#         else:
#             retriever = vector_store.as_retriever(search_kwargs={"k": 4})
#             with st.spinner("Querying Groq API..."):
#                 start_time = time.time()
#                 answer = query_llm_groq(final_query, retriever)
#                 elapsed = time.time() - start_time
#             st.markdown(f"**Response Time:** {elapsed:.2f} seconds")
#             st.markdown("**Answer:**")
#             st.write(answer)

# if st.button("Show Relevant Sources"):
#     if st.session_state["queries"]:
#         final_query = st.session_state["queries"][-1]
#         vector_store = load_faiss_index()
#         if vector_store is None:
#             st.error("Vector store not available.")
#         else:
#             retriever = vector_store.as_retriever(search_kwargs={"k": 4})
#             relevant_docs = retriever.get_relevant_documents(final_query)
#             st.markdown("### Supporting Documentation")
#             st.write(f"Retrieved {len(relevant_docs)} documents.")
#             for i, doc in enumerate(relevant_docs, 1):
#                 with st.expander(f"Source {i}: {doc.metadata.get('source', 'unknown')}"):
#                     st.write(doc.page_content)
#     else:
#         st.warning("Please submit a query first.")
import os
import time
import re
import json
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Groq

# Load environment variables
load_dotenv()

# Configuration
FAISS_STORE_PATH = "update_faiss_store_finance"  # Combined FAISS index location
DATA_DIR = "./extracted_sec_text_test"
EMBEDDED_FILES_PATH = "update_embedded_files.txt"
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')

# Groq API Key (set in your .env file)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --------------------------- Helper Functions ---------------------------
def load_embedded_files():
    if os.path.exists(EMBEDDED_FILES_PATH):
        with open(EMBEDDED_FILES_PATH, "r") as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def save_embedded_files(new_files):
    existing = load_embedded_files()
    updated = existing.union(set(new_files))
    with open(EMBEDDED_FILES_PATH, "w") as f:
        f.write("\n".join(updated))

def save_faiss_index(vectorstore):
    vectorstore.save_local(FAISS_STORE_PATH)
    st.success("💾 Combined vector store updated")

def load_faiss_index():
    if os.path.exists(FAISS_STORE_PATH):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.load_local(FAISS_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
            st.success("✅ Successfully loaded FAISS index.")
            return vector_store
        except Exception as e:
            st.error(f"⚠️ Error loading FAISS index: {e}")
    else:
        st.error(f"❌ FAISS index not found at {FAISS_STORE_PATH}.")
    return None

def financial_preprocessor(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = URL_PATTERN.sub('', text)
    return text.strip()

def financial_chunking(text):
    text_lower = text.lower()
    if any(keyword in text_lower for keyword in ["balance sheet", "income statement", "cash flow"]):
        return 1024  # Use a larger chunk for structured financial statements
    if len(re.findall(r'\$\d+', text)) > 3:
        return 768   # Use a moderate chunk size for texts with many currency values
    return 512      # Default chunk size

# --------------------------- Rapid Embedding Function ---------------------------
def rapid_embedding(vectorstore, texts, metadatas, max_retries=3, max_wait=300):
    attempt = 0
    while attempt < max_retries:
        try:
            vectorstore.add_texts(texts, metadatas=metadatas)
            return True
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RATE_LIMIT_EXCEEDED" in error_str:
                wait_time = (2 ** attempt) * 5
                if wait_time > max_wait:
                    st.error(f"Wait time {wait_time} exceeds maximum threshold of {max_wait} seconds. Skipping embedding.")
                    return False
                st.warning(f"Rate limit exceeded, waiting {wait_time}s (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                attempt += 1
            else:
                st.error(f"Error in rapid_embedding: {e}")
                return False
    return False

# --------------------------- Sequential File Processing ---------------------------
def process_files_sequentially(session, new_files):
    embedded_files = load_embedded_files()
    embedded_count = 0
    skipped_files = []
    for idx, file_path in enumerate(new_files):
        if file_path in embedded_files:
            continue
        st.write(f"\n🔨 Processing: {os.path.basename(file_path)}")
        start_time = time.time()
        try:
            loader = TextLoader(file_path)
            doc = loader.load()[0]
            doc.page_content = financial_preprocessor(doc.page_content)
            chunk_size = financial_chunking(doc.page_content)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=int(chunk_size * 0.2),
                separators=["\n\n", "\n", r"(?<=\. )", " "]
            )
            chunks = text_splitter.split_documents([doc])
            if chunks:
                success = rapid_embedding(
                    session["vectors"],
                    [c.page_content for c in chunks],
                    [c.metadata for c in chunks]
                )
                if success:
                    st.success(f"✅ Embedded {len(chunks)} chunks in {time.time()-start_time:.2f}s")
                    embedded_files.add(file_path)
                    save_embedded_files([file_path])
                    embedded_count += 1
                else:
                    st.error(f"❌ Skipping {file_path} due to rate limits.")
                    skipped_files.append(file_path)
                if idx < len(new_files)-1:
                    st.info("⏳ Cooling period: 90s")
                    time.sleep(90)
        except Exception as e:
            st.error(f"❌ Failed to process {file_path}: {str(e)}")
            with open("processing_errors.log", "a") as f:
                f.write(f"{file_path}: {str(e)}\n")
            skipped_files.append(file_path)
    return {"total_files": len(new_files), "embedded_count": embedded_count, "skipped_files": skipped_files}

# --------------------------- Combined Embedding Workflow ---------------------------
def vector_embedding(session):
    """Embeds all files into a single combined FAISS index with error handling."""
    existing_vectors = load_faiss_index()
    session["embeddings"] = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if existing_vectors:
        session["vectors"] = existing_vectors
        st.write("🔄 Checking for new files...")
    else:
        st.write("🆕 No combined index found. Creating new vector store...")
        session["vectors"] = FAISS.from_documents([], session["embeddings"])
    try:
        loader = DirectoryLoader(DATA_DIR, loader_cls=TextLoader)
        docs = loader.load()
        current_files = {doc.metadata["source"] for doc in docs}
        new_files = [f for f in current_files if f not in load_embedded_files()]
        if new_files:
            st.write(f"Found {len(new_files)} new files to process")
            results = process_files_sequentially(session, new_files)
            save_faiss_index(session["vectors"])
            st.write("-------------------------------------")
            st.write(f"New files processed: {results['embedded_count']}")
            st.write(f"Failed files: {len(results['skipped_files'])}")
            st.write("-------------------------------------")
            if results["skipped_files"]:
                retry = st.text_input("Retry failed files? (y/n): ", value="n")
                if retry.lower() == "y":
                    process_files_sequentially(session, results["skipped_files"])
                    save_faiss_index(session["vectors"])
        else:
            st.write("✅ All files already embedded")
    except Exception as e:
        st.error(f"❌ Error loading directory: {str(e)}")

# --------------------------- Direct Groq Query Function ---------------------------
def query_llm_groq(question, retriever):
    """
    Queries the Groq API directly with an advanced prompt. 
    The prompt instructs the model to act as a world-class financial analyst and provide a direct, concise answer.
    It should incorporate multiple chunks as context if necessary for accuracy.
    """
    try:
        relevant_docs = retriever.get_relevant_documents(question)
        retrieved_text = "\n\n".join([doc.page_content for doc in relevant_docs])
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{
                "role": "user",
                "content": (
                    "You are a world-class financial analyst. Provide a direct and concise answer based solely on the retrieved context below. "
                    "Include calculations and additional detail only if necessary for accuracy.\n\n"
                    f"Retrieved Context:\n{retrieved_text}\n\n"
                    f"Question: {question}"
                )
            }],
            temperature=0.3,
            max_tokens=512,
            top_p=1,
            stream=False,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Groq API Error: {str(e)}"

# --------------------------- Streamlit UI ---------------------------
st.title("Financial RAG System (Combined Index)")
st.write("This system retrieves and processes financial documents to provide direct, concise answers using the Groq API.")

# Sidebar: Company Selection (for context tagging)
st.sidebar.header("Select Company")
def list_companies():
    companies = set()
    for file_name in os.listdir(DATA_DIR):
        if file_name.endswith(".txt"):
            base = os.path.basename(file_name)
            m = re.match(r"^(.*?)_", base)
            if m:
                companies.add(m.group(1))
            else:
                companies.add(os.path.splitext(base)[0])
    return sorted(companies)
companies = list_companies()
selected_company = st.sidebar.selectbox("Company", companies)

# Sidebar: Query History
if "queries" not in st.session_state:
    st.session_state["queries"] = []
st.sidebar.markdown("### Query History")
for idx, q in enumerate(st.session_state["queries"], 1):
    st.sidebar.write(f"{idx}. {q}")

# Main: Query Input
question_input = st.text_input("Enter your financial question:")

if st.button("Submit Query"):
    if not question_input:
        st.warning("Please enter a question.")
    else:
        final_query = f"[Company: {selected_company}] {question_input}"
        st.session_state["queries"].append(final_query)
        with st.spinner("Loading combined vector store..."):
            vector_store = load_faiss_index()
        if vector_store is None:
            st.error("Vector store not available. Please run the embedding process first.")
        else:
            retriever = vector_store.as_retriever(search_kwargs={"k": 8})
            with st.spinner("Querying Groq API..."):
                start_time = time.time()
                answer = query_llm_groq(final_query, retriever)
                elapsed = time.time() - start_time
            st.markdown(f"**Response Time:** {elapsed:.2f} seconds")
            st.markdown("**Answer:**")
            st.write(answer)

if st.button("Show Relevant Sources"):
    if st.session_state["queries"]:
        final_query = st.session_state["queries"][-1]
        vector_store = load_faiss_index()
        if vector_store is None:
            st.error("Vector store not available.")
        else:
            retriever = vector_store.as_retriever(search_kwargs={"k": 8})
            relevant_docs = retriever.get_relevant_documents(final_query)
            st.markdown("### Supporting Documentation")
            st.write(f"Retrieved {len(relevant_docs)} documents.")
            for i, doc in enumerate(relevant_docs, 1):
                with st.expander(f"Source {i}: {doc.metadata.get('source', 'unknown')}"):
                    st.write(doc.page_content)
    else:
        st.warning("Please submit a query first.")
