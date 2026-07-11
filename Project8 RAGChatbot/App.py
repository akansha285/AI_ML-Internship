
import os
import tempfile
from pathlib import Path

import streamlit as st

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import requests
import tempfile


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Samsung Washing Machine AI Assistant",
    page_icon="🫧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# CUSTOM CSS
# =========================
st.markdown(
    """
    <style>
    /* App background */
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(59,130,246,0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(168,85,247,0.18), transparent 28%),
            radial-gradient(circle at bottom left, rgba(16,185,129,0.16), transparent 28%),
            linear-gradient(135deg, #081120 0%, #0f172a 50%, #111827 100%);
        color: #f8fafc;
    }

    /* Main block spacing */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }

    /* Hide Streamlit default menu/footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Hero */
    .hero-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.12), rgba(255,255,255,0.06));
        border: 1px solid rgba(255,255,255,0.12);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border-radius: 26px;
        padding: 34px 30px 26px 30px;
        box-shadow: 0 18px 50px rgba(0,0,0,0.28);
        position: relative;
        overflow: hidden;
        animation: floatIn 0.8s ease-out;
    }

    .hero-card::before {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        background: radial-gradient(circle, rgba(59,130,246,0.28), transparent 65%);
        top: -70px;
        right: -70px;
        border-radius: 50%;
    }

    .hero-card::after {
        content: "";
        position: absolute;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(168,85,247,0.22), transparent 65%);
        bottom: -80px;
        left: -80px;
        border-radius: 50%;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        line-height: 1.15;
        margin-bottom: 0.6rem;
        color: white;
        position: relative;
        z-index: 2;
    }

    .gradient-text {
        background: linear-gradient(90deg, #60a5fa, #c084fc, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .hero-subtitle {
        font-size: 1.02rem;
        color: #dbeafe;
        opacity: 0.92;
        max-width: 880px;
        line-height: 1.7;
        position: relative;
        z-index: 2;
    }

    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 18px;
        position: relative;
        z-index: 2;
    }

    .feature-badge {
        padding: 9px 14px;
        border-radius: 999px;
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.12);
        color: #f8fafc;
        font-size: 0.9rem;
        font-weight: 600;
        box-shadow: 0 8px 18px rgba(0,0,0,0.18);
        transition: all 0.25s ease;
    }

    .feature-badge:hover {
        transform: translateY(-2px) scale(1.02);
        background: rgba(255,255,255,0.16);
    }

    /* Section card */
    .glass-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.10), rgba(255,255,255,0.05));
        border: 1px solid rgba(255,255,255,0.10);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border-radius: 22px;
        padding: 22px;
        box-shadow: 0 16px 35px rgba(0,0,0,0.22);
        margin-bottom: 18px;
        animation: fadeUp 0.7s ease;
    }

    .mini-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 10px;
    }

    .muted-text {
        color: #cbd5e1;
        font-size: 0.96rem;
        line-height: 1.6;
    }

    /* Stats */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 16px;
        margin-top: 8px;
    }

    .metric-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.10), rgba(255,255,255,0.06));
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 18px;
        text-align: center;
        transition: all 0.25s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 28px rgba(0,0,0,0.22);
    }

    .metric-number {
        font-size: 1.45rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 4px;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #cbd5e1;
    }

    /* Chat bubbles */
    .user-bubble {
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        color: white;
        padding: 14px 16px;
        border-radius: 18px 18px 4px 18px;
        width: fit-content;
        max-width: 88%;
        margin-left: auto;
        margin-top: 10px;
        margin-bottom: 10px;
        box-shadow: 0 12px 25px rgba(37, 99, 235, 0.25);
        font-size: 0.97rem;
        line-height: 1.6;
    }

    .assistant-bubble {
        background: rgba(255,255,255,0.10);
        color: #f8fafc;
        padding: 14px 16px;
        border-radius: 18px 18px 18px 4px;
        width: fit-content;
        max-width: 92%;
        margin-right: auto;
        margin-top: 10px;
        margin-bottom: 10px;
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 12px 25px rgba(0, 0, 0, 0.18);
        font-size: 0.97rem;
        line-height: 1.7;
    }

    /* Suggestion chips */
    div.stButton > button {
        width: 100%;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.10);
        background: linear-gradient(135deg, rgba(59,130,246,0.18), rgba(168,85,247,0.18));
        color: white;
        font-weight: 600;
        padding: 0.65rem 0.9rem;
        transition: all 0.25s ease;
        box-shadow: 0 10px 20px rgba(0,0,0,0.18);
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        border: 1px solid rgba(255,255,255,0.18);
        background: linear-gradient(135deg, rgba(59,130,246,0.28), rgba(168,85,247,0.28));
    }

    /* Inputs */
    .stTextInput > div > div > input,
    .stTextArea textarea {
        background: rgba(255,255,255,0.08) !important;
        color: white !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
    }

    .stSelectbox div[data-baseweb="select"] > div,
    .stNumberInput input {
        background: rgba(255,255,255,0.08) !important;
        color: white !important;
        border-radius: 14px !important;
    }

    /* File uploader */
    section[data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.06);
        border: 1px dashed rgba(255,255,255,0.18);
        border-radius: 18px;
        padding: 8px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220 0%, #101826 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }

    /* Divider label */
    .section-heading {
        font-size: 1.18rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.75rem;
    }

    .footer-note {
        text-align: center;
        color: #cbd5e1;
        opacity: 0.9;
        padding: 16px 0 6px 0;
        font-size: 0.92rem;
    }

    @keyframes floatIn {
        0% { opacity: 0; transform: translateY(18px) scale(0.98); }
        100% { opacity: 1; transform: translateY(0) scale(1); }
    }

    @keyframes fadeUp {
        0% { opacity: 0; transform: translateY(14px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# SESSION STATE
# =========================
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = 0

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_context_docs" not in st.session_state:
    st.session_state.last_context_docs = []

if "manual_name" not in st.session_state:
    st.session_state.manual_name = None


# =========================
# HELPER FUNCTIONS
# =========================
def save_uploaded_file(uploaded_file):
    """Save uploaded HTML file to a temp path."""
    suffix = Path(uploaded_file.name).suffix if uploaded_file.name else ".html"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name


def build_rag_pipeline(html_path, openai_api_key, model_name, chunk_size, chunk_overlap):
    """Build the RAG pipeline from uploaded HTML manual."""
    os.environ["OPENAI_API_KEY"] = openai_api_key

    loader = UnstructuredHTMLLoader(file_path=html_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_api_key
    )

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        openai_api_key=openai_api_key
    )

    prompt = ChatPromptTemplate.from_template(
        """You are an intelligent Samsung washing machine manual assistant.
Answer the user's question using only the retrieved manual context.

Guidelines:
- Be accurate, concise, and helpful.
- If the answer is not present in the context, say you don't know based on the manual provided.
- Prefer clear steps if the user asks what to do.
- Keep the answer readable and practical.

Question: {question}

Context:
{context}

Answer:"""
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return rag_chain, retriever, len(splits)


def ask_rag(question):
    """Query RAG chain and also collect retrieved docs for display."""
    retriever = st.session_state.retriever
    rag_chain = st.session_state.rag_chain

    if retriever is None or rag_chain is None:
        return "Please build the knowledge base first.", []

    retrieved_docs = retriever.invoke(question)
    response = rag_chain.invoke(question).content
    return response, retrieved_docs


def render_chat_history():
    """Render chat bubbles from session state."""
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="glass-card">
                <div class="mini-title">👋 Welcome</div>
                <div class="muted-text">
                    Upload a Samsung washing machine HTML manual, build the knowledge base,
                    and ask questions like cycle details, cleaning instructions, wash modes,
                    or troubleshooting guidance.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">{msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="assistant-bubble">{msg["content"]}</div>',
                unsafe_allow_html=True
            )


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## ⚙️ Control Center")
    st.caption("Configure your RAG assistant and load the washing machine manual.")


def download_html():
    response = requests.get(HTML_URL)

    if response.status_code != 200:
        raise Exception("Unable to download HTML file from GitHub.")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")

    temp_file.write(response.content)
    temp_file.close()

    return temp_file.name

    st.markdown("### 🧠 Model & Chunking")
    model_name = st.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4.1-mini"],
        index=0
    )

    chunk_size = st.slider(
        "Chunk Size",
        min_value=500,
        max_value=2000,
        value=1000,
        step=100
    )

    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=50,
        max_value=500,
        value=200,
        step=25
    )

    st.markdown("### 🚀 Build Knowledge Base")
    build_btn = st.button("Build RAG Knowledge Base", use_container_width=True)

    st.markdown("---")
    st.markdown("### 💡 Example Questions")
    st.markdown(
        """
        - What is the cycle for DRUM CLEAN?  
        - What should I do for DAILY WASH?  
        - Which mode is best for delicate clothes?  
        - How do I clean the drum?  
        """
    )


# =========================
# BUILD RAG ON BUTTON CLICK
# =========================
if build_btn:
    if not openai_api_key:
        st.error("Please enter your OpenAI API key.")
    elif uploaded_file is None:
        st.error("Please upload the Samsung HTML manual file.")
    else:
        try:
            with st.spinner("Building the RAG knowledge base... this may take a few moments."):
                temp_html_path = download_html()

                rag_chain, retriever, num_chunks = build_rag_pipeline(
                    html_path=temp_html_path,
                    openai_api_key=openai_api_key,
                    model_name=model_name,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

                st.session_state.rag_chain = rag_chain
                st.session_state.retriever = retriever
                st.session_state.vectorstore_ready = True
                st.session_state.doc_chunks = num_chunks
                st.session_state.manual_name = uploaded_file.name
                st.session_state.messages = []
                st.session_state.last_context_docs = []

            st.success("Knowledge base built successfully. Your AI assistant is ready.")
        except Exception as e:
            st.exception(e)


# =========================
# HERO
# =========================
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">
            Samsung Washing Machine <span class="gradient-text">AI Assistant</span>
        </div>
        <div class="hero-subtitle">
            A premium Retrieval-Augmented Generation (RAG) Streamlit application for Samsung washing machine manuals.
            Upload the manual, build the knowledge base, and ask natural-language questions about cycles,
            cleaning instructions, troubleshooting, warnings, and recommended actions.
        </div>
        <div class="badge-row">
            <div class="feature-badge">⚡ Streamlit Frontend</div>
            <div class="feature-badge">🧠 LangChain RAG</div>
            <div class="feature-badge">📚 Chroma Vector Store</div>
            <div class="feature-badge">🔎 Context-Aware Retrieval</div>
            <div class="feature-badge">✨ Premium UI</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")


# =========================
# STATUS / METRICS
# =========================
colA, colB, colC = st.columns(3)

with colA:
    manual_label = st.session_state.manual_name if st.session_state.manual_name else "Not loaded"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-number">📄</div>
            <div class="metric-label"><b>Manual</b><br>{manual_label}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with colB:
    rag_status = "Ready" if st.session_state.vectorstore_ready else "Waiting"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-number">🧠</div>
            <div class="metric-label"><b>RAG Status</b><br>{rag_status}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with colC:
    chunks = st.session_state.doc_chunks if st.session_state.doc_chunks else 0
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-number">{chunks}</div>
            <div class="metric-label"><b>Document Chunks</b><br>Indexed in vector DB</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")


# =========================
# MAIN LAYOUT
# =========================
left_col, right_col = st.columns([1.45, 0.95], gap="large")

with left_col:
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-heading">💬 Ask the Manual</div>
            <div class="muted-text">
                Ask any question about washing cycles, drum clean mode, usage instructions,
                warnings, or troubleshooting. The assistant will answer using the uploaded Samsung manual.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    render_chat_history()

    st.write("")

    user_question = st.chat_input("Ask something about the washing machine manual...")

    if user_question:
        if not st.session_state.vectorstore_ready:
            st.warning("Please upload the HTML manual and build the knowledge base first.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_question})

            with st.spinner("Searching manual and generating answer..."):
                answer, context_docs = ask_rag(user_question)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.last_context_docs = context_docs

            st.rerun()

with right_col:
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-heading">✨ Suggested Questions</div>
            <div class="muted-text">
                Click a prompt to quickly test your RAG chatbot.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    suggestions = [
        "What is the cycle for DRUM CLEAN?",
        "What should I do for DAILY WASH?",
        "How do I use the various washing machine modes?",
        "Which wash mode should I use for delicate clothes?",
        "How should I clean the drum?",
        "What precautions should I follow while using the machine?",
    ]

    for q in suggestions:
        if st.button(q, key=f"suggestion_{q}"):
            if not st.session_state.vectorstore_ready:
                st.warning("Please build the knowledge base first.")
            else:
                st.session_state.messages.append({"role": "user", "content": q})
                with st.spinner("Generating answer..."):
                    answer, context_docs = ask_rag(q)

                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.last_context_docs = context_docs
                st.rerun()

    st.write("")

    st.markdown(
        """
        <div class="glass-card">
            <div class="section-heading">📚 Retrieved Context</div>
            <div class="muted-text">
                The most recent source chunks used for answering are shown below.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.session_state.last_context_docs:
        for i, doc in enumerate(st.session_state.last_context_docs, start=1):
            with st.expander(f"Chunk {i}", expanded=False):
                st.write(doc.page_content)
    else:
        st.info("Ask a question to see retrieved context chunks here.")

    st.write("")

    st.markdown(
        """
        <div class="glass-card">
            <div class="section-heading">🛠️ How this app works</div>
            <div class="muted-text">
                1. Upload Samsung washing machine manual in HTML format.<br>
                2. The document is split into chunks.<br>
                3. Chunks are embedded and stored in Chroma.<br>
                4. Relevant chunks are retrieved for each question.<br>
                5. The LLM answers using only the manual context.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# FOOTER
# =========================
st.markdown(
    """
    <div class="footer-note">
        Built with Streamlit • LangChain • OpenAI Embeddings • Chroma • RAG for technical documentation
    </div>
    """,
    unsafe_allow_html=True
)

