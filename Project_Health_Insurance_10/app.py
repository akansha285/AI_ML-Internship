import os
import pickle
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
/* App background */
.stApp {
    background: linear-gradient(135deg, #f8fbff 0%, #eef7ff 100%);
}

/* Main width */
.block-container {
    max-width: 1200px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Hero card */
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

/* Info cards */
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

/* Tip box */
.tip-box {
    background: #eff6ff;
    border-left: 5px solid #2563eb;
    padding: 14px;
    border-radius: 12px;
    margin-top: 10px;
    margin-bottom: 14px;
    color: #0f172a;
}

/* Sidebar */
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

/* Buttons */
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

/* Chat */
[data-testid="stChatMessage"] {
    border-radius: 18px;
}

/* Expander */
.streamlit-expanderHeader {
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================
HTML_PATH_DEFAULT = "data/star_health.html"
FAISS_FOLDER = "faiss_store"
FAISS_INDEX_PATH = os.path.join(FAISS_FOLDER, "index.faiss")
CHUNKS_PATH = os.path.join(FAISS_FOLDER, "chunks.pkl")

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

if "vector_ready" not in st.session_state:
    st.session_state.vector_ready = False

if "retrieved_context" not in st.session_state:
    st.session_state.retrieved_context = []

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def extract_text_from_html(html_path: str) -> str:
    """Extract clean visible text from a local HTML file."""
    with open(html_path, "r", encoding="utf-8") as f:
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

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def embed_document(text: str):
    """Gemini embedding for a document chunk."""
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return np.array(response["embedding"], dtype=np.float32)


def embed_query(query: str):
    """Gemini embedding for a user query."""
    response = genai.embed_content(
        model="models/embedding-001",
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


def save_index(index, chunks):
    """Save FAISS index and chunks to disk."""
    os.makedirs(FAISS_FOLDER, exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)


def load_index():
    """Load FAISS index and chunks from disk."""
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        return None, []

    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


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
    # use recent history only
    for msg in chat_history[-6:]:
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


def generate_answer(user_query, chat_history, index, chunks, model_name="gemini-1.5-flash"):
    """Retrieve chunks and generate final answer with Gemini."""
    retrieved_chunks = retrieve_relevant_chunks(user_query, index, chunks, top_k=4)
    prompt = build_prompt(user_query, retrieved_chunks, chat_history)

    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)

    answer = response.text if hasattr(response, "text") else "No response generated."
    return answer, retrieved_chunks


@st.cache_resource(show_spinner=False)
def build_knowledge_base_cached(html_path: str):
    """Build and cache the knowledge base."""
    text = extract_text_from_html(html_path)
    chunks = split_text(text, chunk_size=1200, overlap=200)
    index = build_faiss_index(chunks)
    save_index(index, chunks)
    return index, chunks, len(chunks)


@st.cache_resource(show_spinner=False)
def load_existing_index_cached():
    """Load and cache an existing index."""
    return load_index()

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="hero-card">
    <div class="hero-title">💬 Star Health Context-Aware Chatbot</div>
    <div class="hero-sub">
        Ask questions about Star Health insurance plans using <b>Gemini API</b>, a locally saved Star Health webpage,
        retrieval with FAISS, and conversation memory for follow-up questions.
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
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("⚙️ Chatbot Setup")

    st.markdown('<div class="sidebar-card">Use your local Star Health HTML file to build the knowledge base.</div>', unsafe_allow_html=True)

    html_path = st.text_input("Local HTML path", value=HTML_PATH_DEFAULT)

    st.markdown("### Knowledge Base")
    if st.button("🔨 Build Knowledge Base"):
        try:
            with st.spinner("Reading HTML, creating chunks, generating embeddings, and building FAISS index..."):
                index, chunks, chunk_count = build_knowledge_base_cached(html_path)

                st.session_state.index = index
                st.session_state.chunks = chunks
                st.session_state.vector_ready = True

            st.success(f"Knowledge base built successfully. Indexed {chunk_count} chunks.")
        except Exception as e:
            st.error(f"Failed to build knowledge base: {e}")

    if st.button("📂 Load Existing Index"):
        try:
            with st.spinner("Loading saved FAISS index..."):
                index, chunks = load_existing_index_cached()

                if index is None or not chunks:
                    st.error("No saved FAISS index found. Please build the knowledge base first.")
                else:
                    st.session_state.index = index
                    st.session_state.chunks = chunks
                    st.session_state.vector_ready = True
                    st.success("Existing FAISS index loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load FAISS index: {e}")

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
# MAIN CONTENT BEFORE INDEX READY
# ============================================================
if not st.session_state.vector_ready:
    st.markdown("""
<div class="info-box">
    <h3>How to use this app</h3>
    <ol>
        <li>Download the Star Health webpage and save it locally as <code>data/star_health.html</code>.</li>
        <li>Click <b>Build Knowledge Base</b> from the sidebar.</li>
        <li>Ask insurance-related questions in the chat.</li>
        <li>The chatbot will use webpage content plus conversation history to answer follow-up questions.</li>
    </ol>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="tip-box">
    <b>Context-aware example:</b> If you first ask <i>"What are the types of health insurance policies?"</i>
    and then ask <i>"Which one is best for parents?"</i>, the chatbot will use the previous conversation
    to understand what <i>"which one"</i> refers to.
</div>
""", unsafe_allow_html=True)

    st.stop()

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
    # show user msg
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
                    model_name="gemini-1.5-flash"
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
