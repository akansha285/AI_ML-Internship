```python
import os
import pickle
import hashlib
import numpy as np
import faiss
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import google.generativeai as genai

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Star Health Context-Aware Chatbot",
    page_icon="💬",
    layout="wide"
)

# ============================================================
# LOAD GEMINI API KEY
# Priority:
# 1. Streamlit secrets (for deployment)
# 2. .env file (for local use)
# ============================================================
load_dotenv()

GEMINI_API_KEY = None
if "GEMINI_API_KEY" in st.secrets:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Add it in Streamlit secrets or .env file.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f8fbff 0%, #eef7ff 100%);
}
.block-container {
    max-width: 1200px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
.hero-card {
    background: linear-gradient(135deg, #0f172a, #1e3a8a);
    padding: 28px;
    border-radius: 22px;
    color: white;
    box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    margin-bottom: 18px;
}
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 0.4rem;
}
.hero-sub {
    font-size: 1rem;
    opacity: 0.96;
    line-height: 1.6;
}
.info-box {
    background: rgba(255,255,255,0.92);
    border: 1px solid rgba(148,163,184,0.18);
    backdrop-filter: blur(8px);
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    margin-bottom: 16px;
}
.badge {
    display: inline-block;
    background: #dbeafe;
    color: #1d4ed8;
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 700;
    margin-right: 8px;
    margin-top: 8px;
}
.tip-box {
    background: #eff6ff;
    border-left: 5px solid #2563eb;
    padding: 14px;
    border-radius: 12px;
    margin-top: 10px;
    margin-bottom: 14px;
    color: #0f172a;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #172554 100%);
}
section[data-testid="stSidebar"] * {
    color: white !important;
}
.sidebar-card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.10);
    padding: 14px;
    border-radius: 16px;
    margin-bottom: 12px;
}
.stButton > button {
    width: 100%;
    border-radius: 14px;
    border: none;
    padding: 0.7rem 1rem;
    font-weight: 700;
    transition: 0.25s ease;
    box-shadow: 0 8px 18px rgba(37,99,235,0.18);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 24px rgba(37,99,235,0.25);
}
[data-testid="stChatMessage"] {
    border-radius: 18px;
}
.streamlit-expanderHeader {
    font-weight: 700;
}
.small-note {
    color: #475569;
    font-size: 0.92rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================
HTML_PATH_DEFAULT = "HealthInsurance.html"

FAISS_FOLDER = "faiss_store"
FAISS_INDEX_PATH = os.path.join(FAISS_FOLDER, "index.faiss")
CHUNKS_PATH = os.path.join(FAISS_FOLDER, "chunks.pkl")
META_PATH = os.path.join(FAISS_FOLDER, "meta.pkl")

EMBED_MODEL = "models/embedding-001"
CHAT_MODEL = "gemini-1.5-flash"

SYSTEM_PROMPT = """
You are a context-aware chatbot for Star Health Insurance.

Rules:
1. Answer ONLY from the provided Star Health context.
2. Use the conversation history to understand follow-up questions.
3. If the answer is not available in the provided context, say:
   "I couldn't find that information in the provided Star Health content."
4. Keep answers clear, structured, and user-friendly.
5. Use bullet points whenever suitable.
6. Do not make up policy names, features, or benefits.
"""

# ============================================================
# SESSION STATE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retrieved_context" not in st.session_state:
    st.session_state.retrieved_context = []

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = False

if "kb_source" not in st.session_state:
    st.session_state.kb_source = None

if "kb_status" not in st.session_state:
    st.session_state.kb_status = None


# ============================================================
# HELPERS
# ============================================================
def file_md5(path: str) -> str:
    """Return md5 hash of a file for change detection."""
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def extract_text_from_html_file(html_path: str) -> str:
    """Read HTML from local file path and extract clean text."""
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")

    # Remove noisy tags
    for tag in soup(["script", "style", "noscript", "svg", "img", "footer", "header", "nav"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    clean_text = "\n".join(lines)
    return clean_text


def split_text(text: str, chunk_size: int = 1200, overlap: int = 200):
    """Split long text into overlapping chunks."""
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def embed_document(text: str):
    """Gemini embedding for a document chunk."""
    response = genai.embed_content(
        model=EMBED_MODEL,
        content=text,
        task_type="retrieval_document"
    )
    return np.array(response["embedding"], dtype=np.float32)


def embed_query(query: str):
    """Gemini embedding for a user query."""
    response = genai.embed_content(
        model=EMBED_MODEL,
        content=query,
        task_type="retrieval_query"
    )
    return np.array(response["embedding"], dtype=np.float32)


def build_faiss_index(chunks):
    """Build FAISS index from chunk embeddings."""
    embeddings = [embed_document(chunk) for chunk in chunks]
    embeddings = np.array(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def save_index(index, chunks, meta: dict):
    """Save FAISS index, chunks and metadata to disk."""
    os.makedirs(FAISS_FOLDER, exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)


def load_index():
    """Load FAISS index, chunks and metadata from disk."""
    if not (
        os.path.exists(FAISS_INDEX_PATH)
        and os.path.exists(CHUNKS_PATH)
        and os.path.exists(META_PATH)
    ):
        return None, [], None

    index = faiss.read_index(FAISS_INDEX_PATH)

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)

    return index, chunks, meta


def retrieve_relevant_chunks(query, index, chunks, top_k=4):
    """Retrieve top-k relevant chunks for a user query."""
    if index is None or not chunks:
        return []

    q_emb = embed_query(query).reshape(1, -1)
    distances, indices = index.search(q_emb, top_k)

    results = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            results.append(chunks[idx])

    return results


def build_prompt(user_query, retrieved_chunks, chat_history):
    """Build final prompt using system prompt + history + retrieved context."""
    history_text = ""
    for msg in chat_history[-8:]:
        role = msg["role"].upper()
        history_text += f"{role}: {msg['content']}\n"

    context_text = "\n\n".join(retrieved_chunks)

    prompt = f"""
{SYSTEM_PROMPT}

CHAT HISTORY:
{history_text}

RETRIEVED STAR HEALTH CONTEXT:
{context_text}

CURRENT USER QUESTION:
{user_query}

Answer the question using ONLY the retrieved context and chat history.
If the exact answer is not available in the context, say so clearly.
"""
    return prompt


def generate_answer(user_query, chat_history, index, chunks, model_name=CHAT_MODEL):
    """Retrieve chunks and generate final answer with Gemini."""
    retrieved_chunks = retrieve_relevant_chunks(user_query, index, chunks, top_k=4)
    prompt = build_prompt(user_query, retrieved_chunks, chat_history)

    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    answer = response.text if hasattr(response, "text") else "No response generated."

    return answer, retrieved_chunks


def ensure_knowledge_base(html_path: str):
    """
    Ensure FAISS knowledge base exists and matches current HTML file.
    Rebuild only if:
    - no saved index exists, or
    - HTML file hash has changed.
    """
    if not os.path.exists(html_path):
        raise FileNotFoundError(
            f"'{html_path}' not found. Keep HealthInsurance.html in the same folder as app.py "
            f"and commit it to your GitHub repo."
        )

    current_hash = file_md5(html_path)
    saved_index, saved_chunks, saved_meta = load_index()

    if (
        saved_index is not None
        and saved_chunks
        and saved_meta is not None
        and saved_meta.get("html_hash") == current_hash
        and saved_meta.get("source_path") == html_path
    ):
        return saved_index, saved_chunks, "Loaded existing FAISS index"

    # Rebuild
    clean_text = extract_text_from_html_file(html_path)
    chunks = split_text(clean_text, chunk_size=1200, overlap=200)

    if not chunks:
        raise ValueError("No readable text could be extracted from HealthInsurance.html")

    index = build_faiss_index(chunks)

    meta = {
        "source_path": html_path,
        "html_hash": current_hash,
        "num_chunks": len(chunks)
    }
    save_index(index, chunks, meta)
    return index, chunks, f"Built new FAISS index with {len(chunks)} chunks"


@st.cache_resource(show_spinner=False)
def load_or_build_kb_cached(html_path: str):
    """Cache KB load/build so it doesn't rerun on every Streamlit interaction."""
    return ensure_knowledge_base(html_path)


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="hero-card">
    <div class="hero-title">💬 Star Health Context-Aware Chatbot</div>
    <div class="hero-sub">
        Ask questions about Star Health insurance plans using <b>Gemini API</b>, a repo-hosted HTML source,
        FAISS retrieval, and conversation memory for follow-up questions.
    </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="badge">Gemini API</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="badge">FAISS Retrieval</div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="badge">Context-Aware Chat</div>', unsafe_allow_html=True)

# ============================================================
# LOAD / BUILD KNOWLEDGE BASE AUTOMATICALLY
# ============================================================
try:
    with st.spinner("Preparing Star Health knowledge base..."):
        index, chunks, status = load_or_build_kb_cached(HTML_PATH_DEFAULT)

    st.session_state.index = index
    st.session_state.chunks = chunks
    st.session_state.kb_ready = True
    st.session_state.kb_source = HTML_PATH_DEFAULT
    st.session_state.kb_status = status

except Exception as e:
    st.session_state.kb_ready = False
    st.error(f"Knowledge base setup failed: {e}")
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("⚙️ Chatbot Info")

    st.markdown(
        f"""
        <div class="sidebar-card">
            <b>Knowledge Base Source</b><br>
            {st.session_state.kb_source}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="sidebar-card">
            <b>RAG Status</b><br>
            {st.session_state.kb_status}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="sidebar-card">
            <b>Total Chunks</b><br>
            {len(st.session_state.chunks)}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Chat Controls")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.retrieved_context = []
        st.success("Chat history cleared.")

    st.markdown("### Sample Questions")
    st.markdown("""
- What are different maternity health insurance plans in Star Health?
- What benefits do health insurance policies offer?
- Why should you get a health insurance policy when you're young?
- How many types of health insurance policies are there?
- What are the different types of health insurance schemes in India?
""")

# ============================================================
# INFO BLOCK
# ============================================================
st.markdown("""
<div class="info-box">
    <h3>How this app works</h3>
    <ol>
        <li>The app reads <code>HealthInsurance.html</code> from your GitHub repo.</li>
        <li>It extracts the page text, splits it into chunks, and creates embeddings.</li>
        <li>Those chunks are stored in a FAISS vector index for retrieval.</li>
        <li>When you ask a question, the app retrieves the most relevant Star Health content and sends it to Gemini.</li>
        <li>The chat history is included so follow-up questions stay context-aware.</li>
    </ol>
    <p class="small-note"><b>Tip:</b> If you update the HTML file in GitHub, the app will automatically rebuild the FAISS index because it checks the file hash.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# CHAT DISPLAY
# ============================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ============================================================
# CHAT INPUT
# ============================================================
user_query = st.chat_input("Ask about Star Health health insurance plans...")

if user_query:
    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, retrieved_chunks = generate_answer(
                    user_query=user_query,
                    chat_history=st.session_state.messages,
                    index=st.session_state.index,
                    chunks=st.session_state.chunks,
                    model_name=CHAT_MODEL
                )

                st.session_state.retrieved_context = retrieved_chunks
                st.markdown(answer)

            except Exception as e:
                answer = f"Error while generating response: {e}"
                st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

# ============================================================
# RETRIEVED CONTEXT VIEWER
# ============================================================
if st.session_state.retrieved_context:
    with st.expander("🔍 Retrieved Context Used for Latest Answer"):
        for i, chunk in enumerate(st.session_state.retrieved_context, start=1):
            st.markdown(f"**Chunk {i}:**")
            st.write(chunk)
            st.markdown("---")
```
