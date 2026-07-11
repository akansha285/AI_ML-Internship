import os
import tempfile
import requests

import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredHTMLLoader

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Samsung Washing Machine AI Assistant",
    page_icon="🫧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# GITHUB HTML
# =====================================================

HTML_URL = (
    "https://raw.githubusercontent.com/"
    "akansha285/AI_ML-Internship/main/"
    "Project8%20RAGChatbot/samsung.html"
)

# =====================================================
# OPENAI KEY
# =====================================================

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# =====================================================
# SESSION STATE
# =====================================================

defaults = {

    "rag_chain": None,

    "retriever": None,

    "vectorstore_ready": False,

    "doc_chunks": 0,

    "messages": [],

    "context_docs": [],

    "manual_name": "Samsung Manual (GitHub)"
}

for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value
        st.markdown(
"""
<style>

.stApp{

background:
linear-gradient(135deg,#050816,#0F172A,#1E293B);

color:white;

}

/* Hide Streamlit */

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

/* Sidebar */

section[data-testid="stSidebar"]{

background:#081120;

}

/* Hero */

.hero{

padding:35px;

border-radius:25px;

background:linear-gradient(
135deg,
rgba(255,255,255,.12),
rgba(255,255,255,.05)
);

backdrop-filter:blur(20px);

border:1px solid rgba(255,255,255,.15);

box-shadow:0 20px 50px rgba(0,0,0,.35);

margin-bottom:20px;

}

.hero h1{

font-size:44px;

font-weight:800;

}

.hero p{

font-size:18px;

opacity:.9;

}

/* Cards */

.card{

background:rgba(255,255,255,.08);

padding:20px;

border-radius:18px;

border:1px solid rgba(255,255,255,.1);

margin-bottom:15px;

transition:.3s;

}

.card:hover{

transform:translateY(-3px);

box-shadow:0 12px 25px rgba(0,0,0,.25);

}

/* Buttons */

.stButton>button{

width:100%;

border-radius:15px;

height:48px;

font-weight:700;

font-size:16px;

}

/* Chat */

.user{

background:#2563EB;

padding:15px;

border-radius:15px;

margin:10px 0;

}

.bot{

background:#1F2937;

padding:15px;

border-radius:15px;

margin:10px 0;

}

</style>
""",
unsafe_allow_html=True
)
        with st.sidebar:

    st.title("⚙️ Control Center")

    model_name = st.selectbox(

        "OpenAI Model",

        [

            "gpt-4.1-mini",

            "gpt-4o-mini"

        ]

    )

    chunk_size = st.slider(

        "Chunk Size",

        500,

        2000,

        1000,

        100

    )

    chunk_overlap = st.slider(

        "Chunk Overlap",

        50,

        500,

        200,

        25

    )

    build_btn = st.button(

        "🚀 Build Knowledge Base",

        use_container_width=True

    )

    st.divider()

    st.markdown("### Example Questions")

    st.markdown("""

• What is Drum Clean?

• What is Daily Wash?

• How to clean the washing machine?

• Child Lock

• Eco Bubble

• Error Codes

""")
    def download_html():

    response = requests.get(HTML_URL)

    if response.status_code != 200:

        raise Exception("Unable to download HTML from GitHub.")

    tmp = tempfile.NamedTemporaryFile(

        delete=False,

        suffix=".html"

    )

    tmp.write(response.content)

    tmp.close()

    return tmp.name
    # =====================================================
# BUILD RAG KNOWLEDGE BASE
# =====================================================

@st.cache_resource(show_spinner=False)
def build_rag_pipeline(
    html_path: str,
    model_name: str,
    chunk_size: int,
    chunk_overlap: int,
):
    """
    Build Chroma vector database and RAG pipeline.
    """

    # -----------------------------
    # Load HTML
    # -----------------------------
    loader = UnstructuredHTMLLoader(html_path)
    documents = loader.load()

    # -----------------------------
    # Split Documents
    # -----------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    splits = splitter.split_documents(documents)

    # -----------------------------
    # Embeddings
    # -----------------------------
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )

    # -----------------------------
    # Chroma
    # -----------------------------
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="samsung_manual"
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 4
        }
    )

    # -----------------------------
    # LLM
    # -----------------------------
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=OPENAI_API_KEY
    )

    # -----------------------------
    # Prompt
    # -----------------------------
    prompt = ChatPromptTemplate.from_template(
        """
You are an expert Samsung Washing Machine assistant.

Answer ONLY using the manual context.

If the answer is not present,
reply:

"I couldn't find that information in the Samsung manual."

Be concise.

Question:
{question}

Context:
{context}

Answer:
"""
    )

    # -----------------------------
    # Format Retrieved Docs
    # -----------------------------
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

    return (
        rag_chain,
        retriever,
        len(splits)
    )
    # =====================================================
# STATUS
# =====================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Manual",
        st.session_state.manual_name
    )

with col2:

    st.metric(
        "Chunks",
        st.session_state.doc_chunks
    )

with col3:

    st.metric(
        "Status",
        "Ready"
        if st.session_state.vectorstore_ready
        else "Waiting"
    )
    # =====================================================
# ASK RAG
# =====================================================

def ask_rag(question: str):
    """
    Retrieve relevant documents and generate an answer.
    """

    if (
        st.session_state.rag_chain is None
        or
        st.session_state.retriever is None
    ):
        return (
            "Please build the Knowledge Base first.",
            []
        )

    # Retrieve relevant chunks
    docs = st.session_state.retriever.invoke(question)

    # Generate answer
    response = st.session_state.rag_chain.invoke(question)

    answer = (
        response.content
        if hasattr(response, "content")
        else str(response)
    )

    return answer, docs
    # =====================================================
# CHAT HISTORY
# =====================================================

def render_chat():

    if len(st.session_state.messages) == 0:

        st.info(
            "👋 Ask anything about your Samsung Washing Machine manual."
        )

        return

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])
            # =====================================================
# HERO
# =====================================================

st.markdown(
"""
<div class="hero">

<h1>

🫧 Samsung Washing Machine AI Assistant

</h1>

<p>

Ask natural language questions about washing modes,
cleaning,
error codes,
maintenance,
installation,
and troubleshooting.

Powered by OpenAI + LangChain + Chroma RAG.

</p>

</div>
""",
unsafe_allow_html=True
)
# =====================================================
# CHAT WINDOW
# =====================================================

render_chat()

question = st.chat_input(
    "Ask a question about your washing machine..."
)
# =====================================================
# HANDLE QUESTION
# =====================================================

if question:

    if not st.session_state.vectorstore_ready:

        st.warning(
            "Please build the Knowledge Base first."
        )

        st.stop()

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching Samsung manual..."):

            answer, docs = ask_rag(question)

        st.markdown(answer)

    # Save assistant reply
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    # Save retrieved docs
    st.session_state.context_docs = docs

    st.rerun()
    # =====================================================
# CHAT CONTROLS
# =====================================================

col1, col2 = st.columns(2)

with col1:

    if st.button(
        "🗑 Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []
        st.session_state.context_docs = []

        st.rerun()

with col2:

    if st.button(
        "📚 Clear Retrieved Context",
        use_container_width=True
    ):

        st.session_state.context_docs = []

        st.rerun()
        # =====================================================
# MAIN DASHBOARD LAYOUT
# =====================================================

st.divider()

left_col, right_col = st.columns(
    [1.6, 1],
    gap="large"
)
with left_col:

    st.markdown(
        """
<div class="card">

<h3>💬 AI Conversation</h3>

Ask anything about:

• Wash Programs

• Eco Bubble

• Child Lock

• Drum Clean

• Installation

• Maintenance

• Error Codes

</div>
""",
        unsafe_allow_html=True
    )

    if len(st.session_state.messages) == 0:

        st.info(
            "Start a conversation using the chat box below."
        )
        with right_col:

    st.markdown(
        """
<div class="card">

<h3>📊 Knowledge Base</h3>

</div>
""",
        unsafe_allow_html=True
    )

    st.metric(
        "Chunks",
        st.session_state.doc_chunks
    )

    st.metric(
        "Status",
        "Ready"
        if st.session_state.vectorstore_ready
        else "Waiting"
    )

    st.metric(
        "Messages",
        len(st.session_state.messages)
    )
    # =====================================================
# QUICK QUESTIONS
# =====================================================

st.markdown("## 💡 Suggested Questions")

questions = [

    "What is Drum Clean?",

    "What is Daily Wash?",

    "How do I clean the washing machine?",

    "How do I use Child Lock?",

    "What is Eco Bubble?",

    "How do I clean the filter?",

    "What do error codes mean?",

    "Which mode is best for delicate clothes?"

]

cols = st.columns(2)

for i, q in enumerate(questions):

    with cols[i % 2]:

        if st.button(
            q,
            key=f"quick_{i}",
            use_container_width=True
        ):

            if not st.session_state.vectorstore_ready:

                st.warning(
                    "Please build the Knowledge Base first."
                )

            else:

                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": q
                    }
                )

                with st.spinner("Searching manual..."):

                    answer, docs = ask_rag(q)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

                st.session_state.context_docs = docs

                st.rerun()
                # =====================================================
# HOW IT WORKS
# =====================================================

with st.expander(
    "⚙️ How this AI Assistant Works",
    expanded=False
):

    st.markdown(
        """

1. Downloads the Samsung manual directly from GitHub.

2. Splits the manual into semantic chunks.

3. Creates OpenAI embeddings.

4. Stores them inside ChromaDB.

5. Retrieves the most relevant chunks.

6. GPT answers ONLY from those chunks.

"""
    )

