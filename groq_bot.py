# --- SAFE IMPORT FOR SQLITE (Cloud vs Local) ---
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass  # If pysqlite3 is missing (local PC), use standard sqlite3

import streamlit as st
import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIGURATION ---
DATA_PATH = "./data"
CHROMA_PATH = "./chroma_db"
LLM_MODEL = "llama-3.1-8b-instant" 

st.set_page_config(
    page_title="Shastika Global | AI Export Desk", 
    page_icon="🥥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- DARK MODE & CLEAN UI CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&family=Open+Sans:wght@400;600&display=swap');

/* --- GLOBAL TEXT COLORS (High Contrast White) --- */
.stApp, .stMarkdown, p, h1, h2, h3, h4, h5, h6, span, div {
    color: #E0E0E0 !important;
}

/* --- MAIN BACKGROUND (Dark Gradient) --- */
.stApp {
    background: linear-gradient(180deg, #0E1117 0%, #051a05 100%);
}

/* --- TYPOGRAPHY --- */
html, body, [class*="css"] {
    font-family: 'Open Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Montserrat', sans-serif;
    color: #4CAF50 !important; /* Bright Green for Headings */
    font-weight: 700;
}

/* --- NUCLEAR FOOTER REMOVAL (Force Hide) --- */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
[data-testid="stFooter"] { display: none !important; }
.stDeployButton { display: none !important; }
div[class^='viewerBadge'] { display: none !important; }

/* --- EMBED OPTIMIZATION --- */
.block-container {
    padding: 1rem 1rem 2rem 1rem !important;
}

/* --- HEADER STYLING (Dark Card) --- */
.brand-header {
    background: #1E1E1E;
    border-bottom: 3px solid #4CAF50;
    padding: 20px;
    text-align: center;
    border-radius: 0 0 15px 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    margin-bottom: 20px;
}

.brand-title {
    font-size: 1rem;
    font-weight: 800;
    color: #B39DDB !important; /* Light Purple for contrast */
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 5px;
}

.brand-subtitle {
    font-size: 2.5rem;
    color: #4CAF50 !important; /* Bright Green */
    font-weight: 600;
    letter-spacing: 2px;
    text-shadow: 0px 0px 10px rgba(76, 175, 80, 0.3);
    margin: 0;
}

/* --- CHAT BUBBLES --- */
/* User Bubble (Green) */
.st-emotion-cache-1c7v0n0 {
    background-color: #1B5E20 !important; /* Darker Green */
    border: 1px solid #2E7D32;
    border-radius: 15px 15px 0 15px;
}
.st-emotion-cache-1c7v0n0 * {
    color: #E0E0E0 !important;
}

/* Assistant Bubble (Dark Grey) */
.st-emotion-cache-4oy32j {
    background-color: #262730 !important;
    border: 1px solid #444;
    border-left: 4px solid #F9A825; /* Amber accent */
}
.st-emotion-cache-4oy32j * {
    color: #E0E0E0 !important;
}

/* --- INPUT FIELD --- */
.stTextInput > div > div > input {
    background-color: #262730 !important;
    color: #FFFFFF !important;
    border: 2px solid #4CAF50 !important;
    border-radius: 30px;
}
.stTextInput > div > div > input::placeholder {
    color: #AAAAAA !important;
}

/* ================= SELECTBOX DARK FIX ================= */
div[data-baseweb="select"] > div {
    background-color: #262730 !important;
    color: #FFFFFF !important;
    border: 2px solid #4CAF50 !important;
    border-radius: 12px !important;
}

div[data-baseweb="select"] span {
    color: #FFFFFF !important;
    font-weight: 600;
}

div[data-baseweb="select"] svg {
    fill: #FFFFFF !important;
}

/* Dropdown Menu Items */
ul[data-baseweb="menu"] {
    background-color: #262730 !important;
    border: 1px solid #444 !important;
    border-radius: 12px !important;
}

ul[data-baseweb="menu"] li {
    background-color: #262730 !important;
    color: #E0E0E0 !important;
}

ul[data-baseweb="menu"] li:hover {
    background-color: #1B5E20 !important; /* Green hover */
}

ul[data-baseweb="menu"] li[aria-selected="true"] {
    background-color: #2E7D32 !important;
    color: #FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)

# --- CLEAN HEADER (No Address) ---
st.markdown("""
<div class="brand-header">
    <div class="brand-subtitle">Shastika Global Impex</div>
    <div class="brand-title">Export Assistant</div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_and_process_data():
    """Loads knowledge base SILENTLY."""
    if not os.path.exists(DATA_PATH): return None 
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        try:
            return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        except Exception: pass

    loader = DirectoryLoader(DATA_PATH, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}, silent_errors=False)
    documents = loader.load()
    if not documents: return None 

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    try:
        vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=CHROMA_PATH)
        return vectorstore
    except Exception: return None

def get_rag_chain(vectorstore, selected_language):
    """Defines the Groq LLM with SECURE API HANDLING."""
    
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        groq_api_key = os.environ.get("GROQ_API_KEY")

    if not groq_api_key:
        st.error("🚨 System Error: API Key missing. Please configure 'GROQ_API_KEY' in Streamlit Secrets.")
        st.stop()

    llm = ChatGroq(model=LLM_MODEL, temperature=0, api_key=groq_api_key)

    # --- PROMPT ---
    template = f"""
    You are the **AI Export Assistant** for **Shastika Global Impex**.
    
    **LANGUAGE SETTINGS:**
    - User Language: **{selected_language}**
    - **PROCESS:** Translate Input to English -> **AUTO-CORRECT TYPOS** -> Match with Context -> Translate Answer to **{selected_language}**.
    
    **INTELLIGENT LOGIC:**
    1. **Typos:** Interpret "docnut" -> Coconut, "coirpeet" -> Coir Pith.
    2. **Tone:** Professional, Warm, Helpful.
    
    **STRICT DATABASE RULES:**
    1. **SOURCE OF TRUTH:** Answer **ONLY** using the "CONTEXT" below.
    2. **ZERO HALLUCINATION:** If asked about products NOT in the context, politely reply:
        *"I specialize in Shastika's core exports: Coconuts, Coir, and Bananas. I don't have information on that specific item."*

    **FORMATTING:**
    - Use **Bold** for Product Names.
    - Use *Bullet Points* for specifications.

    **CONTEXT:**
    {{context}}

    **USER INPUT:**
    {{question}}

    **YOUR RESPONSE (in {selected_language}):**
    """
    
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": vectorstore.as_retriever(search_kwargs={"k": 3}), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- MAIN APP LOGIC ---
def main():
    
    LANGUAGE_OPTIONS = ["English", "Spanish", "French", "Hindi", "Tamil"]
    
    if "selected_language" not in st.session_state:
        st.session_state.selected_language = "English"

    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        selected_language = st.selectbox(
            "Select Language:", 
            options=LANGUAGE_OPTIONS,
            index=LANGUAGE_OPTIONS.index(st.session_state.selected_language),
            key="lang_select",
            label_visibility="collapsed"
        )

    if selected_language != st.session_state.selected_language:
        st.session_state.selected_language = selected_language
        st.session_state.messages = [] 

    # --- REMOVED SIDEBAR, DATA LOADS AUTOMATICALLY ---
    vectorstore = load_and_process_data()
    if vectorstore is None:
        st.error("⚠ System Offline: Database not found in './data'.")
        st.stop()
    
    rag_chain = get_rag_chain(vectorstore, selected_language)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- CHAT LOOP ---
    for message in st.session_state.messages:
        avatar_icon = "🧑‍💼" if message["role"] == "user" else "🌿"
        with st.chat_message(message["role"], avatar=avatar_icon):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about Coconuts, Bananas, or Coir..."):
        
        # --- SECRET COMMAND: /refresh ---
        if prompt.lower().strip() == "/refresh":
            with st.spinner("♻️ Refreshing Knowledge Base..."):
                if os.path.exists(CHROMA_PATH):
                    shutil.rmtree(CHROMA_PATH)
                st.cache_resource.clear()
            st.success("✅ Database Updated! Please reload the page.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(prompt)
        
        with st.chat_message("assistant", avatar="🌿"):
            response_placeholder = st.empty()
            full_response = ""
            try:
                for chunk in rag_chain.stream(prompt):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"System Error: {e}")

if __name__ == "__main__":
    main()