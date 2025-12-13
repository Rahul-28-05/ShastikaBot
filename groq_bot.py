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

# --- 1. SETUP PAGE ---
st.set_page_config(
    page_title="Shastika Global | AI Export Desk", 
    page_icon="🥥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. ORGANIC GREEN THEME (CSS) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

/* --- MAIN BACKGROUND (Soft Green Gradient) --- */
.stApp {
    background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); /* Mint to Sage Green */
    color: #1a1a1a;
}

/* --- TYPOGRAPHY --- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3 {
    color: #1B5E20; /* Dark Forest Green */
    font-weight: 600;
}

/* --- IFRAME / SIDEBAR OPTIMIZATION (THE TWEAK) --- */
/* This reduces padding so it fits in the website popup */
.block-container {
    padding-top: 1rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    padding-bottom: 1rem !important;
}

/* --- CHAT HEADER --- */
.glass-header {
    background: rgba(255, 255, 255, 0.6); /* Semi-transparent White */
    backdrop-filter: blur(12px);
    border-bottom: 1px solid #a5d6a7;
    padding: 15px; /* Reduced padding for sidebar */
    margin-bottom: 15px;
    border-radius: 0 0 20px 20px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
}

.header-title {
    font-size: 1.8rem; /* Slightly smaller for sidebar */
    font-weight: 800;
    color: #1B5E20; 
    letter-spacing: 1px;
}

.header-subtitle {
    font-size: 0.8rem;
    color: #2E7D32; 
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 600;
}

/* --- CHAT BUBBLES --- */

/* User Bubble (Deep Green) */
.st-emotion-cache-1c7v0n0 { 
    background-color: #2E7D32; /* Forest Green */
    border: none;
    border-radius: 15px 15px 0 15px;
    color: #FFFFFF;
    box-shadow: 0 3px 6px rgba(46, 125, 50, 0.3);
}

/* AI Bubble (Crisp White) */
.st-emotion-cache-4oy32j { 
    background-color: #FFFFFF; /* Pure White */
    border: 1px solid #A5D6A7; /* Green Border */
    border-radius: 15px 15px 15px 0;
    color: #1B5E20; 
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}

/* --- INPUT FIELD --- */
.stTextInput > div > div > input {
    color: #1B5E20; 
    background-color: #FFFFFF; 
    border: 2px solid #A5D6A7 !important; 
    border-radius: 12px; 
    padding: 12px; /* Smaller padding */
    font-size: 14px;
}
.stTextInput > div > div > input:focus {
    border: 2px solid #2E7D32 !important;
    box-shadow: 0 0 15px rgba(46, 125, 50, 0.2);
}

/* Select Box Styling */
div[data-baseweb="select"] > div {
    background-color: #FFFFFF;
    border-color: #A5D6A7;
    color: #1B5E20;
}

/* Hide Streamlit Elements */
header {visibility: hidden !important;}
footer {visibility: hidden !important;}
.stSpinner {color: #2E7D32;}
</style>
""", unsafe_allow_html=True)

# --- HEADER RENDER ---
st.markdown("""
<div class="glass-header">
    <div class="header-subtitle">Shastika Global Impex</div>
    <div class="header-title">AI EXPORT DESK</div>
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
    """Defines the Groq LLM with ADVANCED TYPO CORRECTION Logic."""
    
    try: groq_api_key = st.secrets.get("GROQ_API_KEY")
    except: groq_api_key = os.environ.get("GROQ_API_KEY")
    
    if not groq_api_key: 
        groq_api_key = "YOUR_FALLBACK_KEY_HERE" 

    llm = ChatGroq(model=LLM_MODEL, temperature=0, api_key=groq_api_key)

    # --- PROMPT: INTELLIGENT TYPO CORRECTION ---
    template = f"""
    You are the **AI Export Desk** for Shastika Global Impex.
    
    **LANGUAGE SETTINGS:**
    - User Language: **{selected_language}**
    - **PROCESS:** Translate Input to English -> **AUTO-CORRECT TYPOS** -> Match with Context -> Translate Answer to **{selected_language}**.
    
    **INTELLIGENT TYPO & INTENT HANDLING:**
    1. **Detect & Correct:** Users may type "docnut" (Coconut), "bannana" (Banana), "huskd" (Husked), "ceor" (Coir).
       - You MUST silently interpret "docnut" as "Coconut".
       - You MUST silently interpret "banna" as "Banana".
    2. **Context Matching:** If the user asks for "the brown fruit we drink", infer **Semi-Husked Coconut**.
    
    **STRICT DATABASE RULES:**
    1. **SOURCE OF TRUTH:** Answer **ONLY** using the "CONTEXT" below.
    2. **ZERO HALLUCINATION:** If the *corrected* product (e.g., Apple, Rice) is NOT in the Context, strictly reply: 
       *"I apologize, but we strictly deal in Coconuts, Coir, and Bananas. I do not have information on that product."*

    **INTERACTION STYLE:**
    - **Greeting:** Only greet if the user says "Hi/Hello". Otherwise, answer immediately.
    - **Formatting:** Use **Bold** for headers and *Bullet Points* for specs.
    - **No Fluff:** Do not explain that you corrected the typo. Just answer the question.

    **CONTEXT (YOUR ONLY KNOWLEDGE):**
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
    
    # --- LANGUAGE SELECTION ---
    LANGUAGE_OPTIONS = ["English", "Spanish", "French", "Hindi", "Tamil"]
    
    if "selected_language" not in st.session_state:
        st.session_state.selected_language = "English"

    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        selected_language = st.selectbox(
            "Select Communication Language:", 
            options=LANGUAGE_OPTIONS,
            index=LANGUAGE_OPTIONS.index(st.session_state.selected_language),
            key="lang_select",
            label_visibility="collapsed"
        )

    if selected_language != st.session_state.selected_language:
        st.session_state.selected_language = selected_language
        st.session_state.messages = [] 

    with st.sidebar:
        st.header("Database Control")
        if st.button("Refresh Knowledge Base"):
             if os.path.exists(CHROMA_PATH):
                shutil.rmtree(CHROMA_PATH)
                st.cache_resource.clear()
                st.rerun()

    vectorstore = load_and_process_data()
    if vectorstore is None:
        st.error("⚠ System Offline: Database not found in './data'.")
        st.stop()
    
    rag_chain = get_rag_chain(vectorstore, selected_language)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- CHAT LOOP ---
    for message in st.session_state.messages:
        # Green Avatars for Organic Theme
        avatar_icon = "🧑‍💼" if message["role"] == "user" else "🌿"
        with st.chat_message(message["role"], avatar=avatar_icon):
            st.markdown(message["content"])

    if prompt := st.chat_input("Type your inquiry here..."):
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