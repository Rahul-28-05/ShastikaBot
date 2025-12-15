__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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

# --- NUCLEAR CSS TO REMOVE "BUILT WITH STREAMLIT" ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&family=Open+Sans:wght@400;600&display=swap');

/* --- GLOBAL COLORS --- */
.stApp, .stMarkdown, p, h1, h2, h3, h4, h5, h6, span, div {
    color: #E0E0E0 !important;
}

/* --- BACKGROUND --- */
.stApp {
    background: linear-gradient(180deg, #0E1117 0%, #051a05 100%);
}

/* --- CRITICAL: NUCLEAR OPTION TO HIDE BRANDING --- */
/* 1. Hides the top hamburger menu */
#MainMenu {visibility: hidden;}

/* 2. Hides the footer text */
footer {visibility: hidden;}

/* 3. Hides the "Deploy" button */
.stDeployButton {display:none;}

/* 4. Hides the top header bar */
header {visibility: hidden;}

/* 5. AGGRESSIVE: Hides the specific Streamlit footer container */
[data-testid="stFooter"] {
    display: none !important;
    visibility: hidden !important;
    height: 0px !important;
}

/* 6. Hides the "Viewer" badge (the little icon at bottom right) */
.viewerBadge_container__1QSob {
    display: none !important;
}

/* 7. Generic catch-all for viewer badges */
div[class*="viewerBadge"] {
    display: none !important;
}

/* --- HEADER & TYPOGRAPHY --- */
h1, h2, h3 {
    font-family: 'Montserrat', sans-serif;
    color: #4CAF50 !important;
    font-weight: 700;
}

.block-container {
    padding: 1rem 1rem 0rem 1rem !important;
}

/* --- CUSTOM HEADER --- */
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
    color: #B39DDB !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.brand-subtitle {
    font-size: 2.5rem;
    color: #4CAF50 !important;
    font-weight: 600;
    letter-spacing: 2px;
    text-shadow: 0px 0px 10px rgba(76, 175, 80, 0.3);
    margin: 0;
}

/* --- CHAT BUBBLES --- */
.st-emotion-cache-1c7v0n0 {
    background-color: #1B5E20 !important;
    border: 1px solid #2E7D32;
    border-radius: 15px 15px 0 15px;
}
.st-emotion-cache-4oy32j {
    background-color: #262730 !important;
    border: 1px solid #444;
    border-left: 4px solid #F9A825;
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

/* --- SELECTBOX FIX --- */
div[data-baseweb="select"] > div {
    background-color: #262730 !important;
    color: #FFFFFF !important;
    border: 2px solid #4CAF50 !important;
}
div[data-baseweb="select"] span { color: #FFFFFF !important; }
ul[data-baseweb="menu"] { background-color: #262730 !important; border: 1px solid #444 !important; }
ul[data-baseweb="menu"] li { background-color: #262730 !important; color: #E0E0E0 !important; }
ul[data-baseweb="menu"] li:hover { background-color: #1B5E20 !important; }
ul[data-baseweb="menu"] li[aria-selected="true"] { background-color: #2E7D32 !important; }
</style>
""", unsafe_allow_html=True)

# --- HEADER (CLEAN) ---
st.markdown("""
<div class="brand-header">
    <div class="brand-subtitle">Shastika Global Impex</div>
    <div class="brand-title">Export Assistant</div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_and_process_data():
    if not os.path.exists(DATA_PATH): return None 
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        try: return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        except Exception: pass
    loader = DirectoryLoader(DATA_PATH, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}, silent_errors=False)
    documents = loader.load()
    if not documents: return None 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    try: return Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=CHROMA_PATH)
    except Exception: return None

def get_rag_chain(vectorstore, selected_language):
    try: groq_api_key = st.secrets["GROQ_API_KEY"]
    except: groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key: st.stop()
    llm = ChatGroq(model=LLM_MODEL, temperature=0, api_key=groq_api_key)
    template = f"""
    You are the **AI Export Assistant** for **Shastika Global Impex**.
    **LANGUAGE:** {selected_language}
    **RULES:** Answer ONLY using the CONTEXT. No hallucinations.
    **CONTEXT:** {{context}}
    **INPUT:** {{question}}
    **RESPONSE (in {selected_language}):**
    """
    prompt = ChatPromptTemplate.from_template(template)
    return ({"context": vectorstore.as_retriever(search_kwargs={"k": 3}), "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())

def main():
    LANGUAGE_OPTIONS = ["English", "Spanish", "French", "Hindi", "Tamil"]
    if "selected_language" not in st.session_state: st.session_state.selected_language = "English"
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        selected_language = st.selectbox("Select Language:", options=LANGUAGE_OPTIONS, index=LANGUAGE_OPTIONS.index(st.session_state.selected_language), key="lang_select", label_visibility="collapsed")
    if selected_language != st.session_state.selected_language:
        st.session_state.selected_language = selected_language
        st.session_state.messages = [] 
    vectorstore = load_and_process_data()
    if vectorstore is None: st.stop()
    rag_chain = get_rag_chain(vectorstore, selected_language)
    if "messages" not in st.session_state: st.session_state.messages = []
    for message in st.session_state.messages:
        avatar_icon = "🧑‍💼" if message["role"] == "user" else "🌿"
        with st.chat_message(message["role"], avatar=avatar_icon): st.markdown(message["content"])
    if prompt := st.chat_input("Ask about Coconuts, Bananas, or Coir..."):
        if prompt.lower().strip() == "/refresh":
            if os.path.exists(CHROMA_PATH): shutil.rmtree(CHROMA_PATH)
            st.cache_resource.clear()
            st.stop()
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💼"): st.markdown(prompt)
        with st.chat_message("assistant", avatar="🌿"):
            response_placeholder = st.empty()
            full_response = ""
            try:
                for chunk in rag_chain.stream(prompt):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception: pass

if __name__ == "__main__":
    main()