import os
import streamlit as st
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="DS Cheat Sheets RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force path resolving for imports
import sys
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from src.retrieve import ask_question, get_anthropic_client

# Style customizations for a premium dark/light mode experience
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E90FF, #9370DB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    .source-card {
        padding: 0.8rem;
        border-radius: 0.5rem;
        background-color: rgba(30, 144, 255, 0.05);
        border-left: 4px solid #1E90FF;
        margin-bottom: 0.5rem;
    }
    .chunk-box {
        font-size: 0.85rem;
        background-color: rgba(0, 0, 0, 0.03);
        border: 1px solid rgba(0, 0, 0, 0.1);
        border-radius: 0.3rem;
        padding: 0.6rem;
        margin-top: 0.4rem;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🤖 RAG Configuration")
    
    # API Key override
    api_key_input = st.text_input(
        "Anthropic API Key",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Paste your Anthropic Claude API Key here. If empty, the app will read the ANTHROPIC_API_KEY from environment or .env file."
    )
    if api_key_input:
        os.environ["ANTHROPIC_API_KEY"] = api_key_input
        
    # K parameter
    k_val = st.slider(
        "Top-k retrieved chunks",
        min_value=1,
        max_value=10,
        value=4,
        help="Number of relevant reference blocks to pass to the model."
    )

    st.markdown("---")
    st.markdown("### 📊 Index Statistics")
    st.markdown("- **Dataset:** `timoboz/data-science-cheat-sheets`")
    st.markdown("- **Version:** Tier 2 (Full Corpus, 194 PDFs)")
    st.markdown("- **Vector Store:** Chroma DB (Local)")
    st.markdown("- **Embeddings:** Local BGE-small-en-v1.5")
    
    st.markdown("---")
    st.markdown("### 🔍 Supported Topics")
    st.caption("Algorithms, Artificial Intelligence, Big Data, Data Engineering, Data Mining, Data Science, Data Visualization, Data Warehouse, Deep Learning, DevOps, Docker & Kubernetes, Excel, Git, Interview Questions, Linux, Machine Learning, Mathematics, Matlab, NLP, Numpy, Ordinary Differential Equations, Pandas, Probability, Python, R Cheat Sheet, SQL, Scala, Statistics")

# Main content area
st.markdown("<h1 class='main-title'>DS Cheat Sheets RAG Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Ask questions over the complete corpus of 194 cheat sheets. Answers are strictly grounded and cited.</p>", unsafe_allow_html=True)

# Initialize chat messages session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if they exist in the message metadata
        if "sources" in message and message["sources"]:
            with st.expander("🔍 Citations & Chunks Used"):
                cols = st.columns(2)
                for idx, src in enumerate(message["sources"]):
                    col = cols[idx % 2]
                    col.markdown(
                        f"<div class='source-card'><strong>📄 File:</strong> {Path(src['source_file']).name}<br/>"
                        f"<strong>Page:</strong> {src['page']} | <strong>Section:</strong> {src['section']}</div>",
                        unsafe_allow_html=True
                    )
                st.markdown("**Retrieved Context Texts:**")
                for idx, chk in enumerate(message["raw_chunks"]):
                    st.markdown(f"**Chunk {idx+1} (Source: {Path(chk['metadata']['source_file']).name}, Page {chk['metadata']['page']}):**")
                    st.markdown(f"<div class='chunk-box'>{chk['text']}</div>", unsafe_allow_html=True)

# Accept user query
if user_query := st.chat_input("Ask a question (e.g. 'How do I merge two dataframes in Pandas?')"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving references and generating response..."):
            try:
                # Call ask_question from retrieval module
                result = ask_question(user_query, k=k_val)
                st.markdown(result["answer"])
                
                # Render sources
                if result["sources"]:
                    with st.expander("🔍 Citations & Chunks Used"):
                        cols = st.columns(2)
                        for idx, src in enumerate(result["sources"]):
                            col = cols[idx % 2]
                            col.markdown(
                                f"<div class='source-card'><strong>📄 File:</strong> {Path(src['source_file']).name}<br/>"
                                f"<strong>Page:</strong> {src['page']} | <strong>Section:</strong> {src['section']}</div>",
                                unsafe_allow_html=True
                            )
                        st.markdown("**Retrieved Context Texts:**")
                        for idx, chk in enumerate(result["raw_chunks"]):
                            st.markdown(f"**Chunk {idx+1} (Source: {Path(chk['metadata']['source_file']).name}, Page {chk['metadata']['page']}):**")
                            st.markdown(f"<div class='chunk-box'>{chk['text']}</div>", unsafe_allow_html=True)
                
                # Save assistant message with sources to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "raw_chunks": result.get("raw_chunks", [])
                })
            except Exception as e:
                err_msg = f"An error occurred while answering your question: {e}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
