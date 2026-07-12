"""
Dhurandhar 2 — Movie Review NLP Analyzer (Streamlit App)
==========================================================
A Streamlit front-end for the NLP pipeline originally built in the
Dhurandhar_2_review_with_LLM notebook.

Features
--------
1. Sentiment Analysis (DistilBERT SST-2)
2. Machine Translation (English -> Spanish)
3. Question Answering over a review
4. Text Summarization
5. Model Evaluation (Accuracy / F1 on a labeled dataset, BLEU for translation)

Run with:
    streamlit run app.py
"""

import io
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Movie Review NLP Analyzer",
    page_icon="🎬",
    layout="wide",
)

# ----------------------------------------------------------------------
# Cached model loaders
# ----------------------------------------------------------------------
# NOTE: We deliberately avoid `pipeline("summarization"/"translation_en_to_es"/
# "question-answering", ...)`. Some environments resolve a very recent
# `transformers` release whose pipeline *task registry* no longer includes
# those exact task-name strings (only e.g. "sentiment-analysis",
# "text-classification", etc. are guaranteed present), which raises
# `KeyError: Unknown task ...`. Loading the tokenizer/model directly and
# running inference by hand sidesteps that registry entirely, so the app
# keeps working no matter which transformers version gets installed.

@st.cache_resource(show_spinner="Loading sentiment analysis model...")
def load_sentiment_pipeline():
    from transformers import pipeline
    # "text-classification" is the canonical, stable task name
    # ("sentiment-analysis" is just an alias for it).
    return pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )


@st.cache_resource(show_spinner="Loading translation model...")
def load_translation_model():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    model_name = "Helsinki-NLP/opus-mt-en-es"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def translate_text(text: str) -> str:
    import torch
    tokenizer, model = load_translation_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


@st.cache_resource(show_spinner="Loading question answering model...")
def load_qa_model():
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    model_name = "distilbert-base-cased-distilled-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def answer_question(question: str, context: str) -> dict:
    import torch
    tokenizer, model = load_qa_model()
    inputs = tokenizer(
        question, context, return_tensors="pt", truncation=True, max_length=384
    )
    with torch.no_grad():
        outputs = model(**inputs)
    start_logits, end_logits = outputs.start_logits, outputs.end_logits
    start_idx = int(torch.argmax(start_logits))
    end_idx = int(torch.argmax(end_logits))
    if end_idx < start_idx:
        end_idx = start_idx
    answer_ids = inputs["input_ids"][0][start_idx : end_idx + 1]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
    score = float(
        torch.softmax(start_logits, dim=-1)[0][start_idx]
        * torch.softmax(end_logits, dim=-1)[0][end_idx]
    )
    return {"answer": answer if answer.strip() else "(no answer found)", "score": score}


@st.cache_resource(show_spinner="Loading summarization model...")
def load_summarization_model():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    model_name = "sshleifer/distilbart-cnn-12-6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def summarize_text(text: str, max_length: int = 60, min_length: int = 15) -> str:
    import torch
    tokenizer, model = load_summarization_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            min_length=min(min_length, max_length - 1),
            do_sample=False,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


@st.cache_resource(show_spinner=False)
def load_metrics():
    import evaluate
    return {
        "accuracy": evaluate.load("accuracy"),
        "f1": evaluate.load("f1"),
        "bleu": evaluate.load("bleu"),
    }


# ----------------------------------------------------------------------
# Sample dataset (used if the user doesn't upload their own CSV)
# ----------------------------------------------------------------------
SAMPLE_DATA = pd.DataFrame(
    {
        "Review": [
            "The movie was fantastic! Great acting and story.",
            "Absolutely loved the direction and cinematography.",
            "It was a complete waste of time, very boring.",
            "The plot made no sense and the acting was terrible.",
            "A brilliant sequel with powerful performances.",
        ],
        "Class": ["POSITIVE", "POSITIVE", "NEGATIVE", "NEGATIVE", "POSITIVE"],
    }
)


def _parse_csv_bytes(raw: bytes):
    """Try semicolon first (matches the original notebook), then comma."""
    for sep in [";", ","]:
        try:
            df = pd.read_csv(io.BytesIO(raw), delimiter=sep)
            if "Review" in df.columns and "Class" in df.columns:
                return df
        except Exception:
            continue
    return None


def normalize_github_url(url: str) -> str:
    """Convert a normal GitHub 'blob' URL into a raw.githubusercontent.com URL."""
    url = url.strip()
    if "github.com" in url and "/blob/" in url:
        url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    return url


@st.cache_data(show_spinner="Fetching CSV from GitHub...", ttl=600)
def load_dataset_from_github(url: str):
    import requests
    raw_url = normalize_github_url(url)
    response = requests.get(raw_url, timeout=15)
    response.raise_for_status()
    return response.content, raw_url


def load_dataset(uploaded_file):
    """Load a semicolon-delimited CSV with Review/Class columns, falling back to sample data."""
    if uploaded_file is None:
        return SAMPLE_DATA.copy(), "sample"
    raw = uploaded_file.read()
    df = _parse_csv_bytes(raw)
    if df is not None:
        return df, "uploaded"
    st.error("Could not find 'Review' and 'Class' columns in the uploaded CSV. Using sample data instead.")
    return SAMPLE_DATA.copy(), "sample"


# ----------------------------------------------------------------------
# Sidebar — data source
# ----------------------------------------------------------------------
st.sidebar.title("🎬 Movie Review NLP Analyzer")
st.sidebar.markdown("Choose where to load your reviews from:")

data_source = st.sidebar.radio(
    "Data source",
    ["GitHub URL", "Upload CSV", "Sample data"],
    index=0,
)

df, source = SAMPLE_DATA.copy(), "sample"

if data_source == "GitHub URL":
    st.sidebar.caption(
        "Paste a link to the CSV in your repo — either a normal "
        "`github.com/.../blob/...` link or a `raw.githubusercontent.com` link. "
        "For a **private** repo, use the raw URL and add a token below."
    )
    github_url = st.sidebar.text_input(
        "GitHub CSV URL",
        placeholder="https://github.com/user/repo/blob/main/data/reviews.csv",
    )
    github_token = st.sidebar.text_input(
        "GitHub token (optional, for private repos)", type="password"
    )
    fetch_btn = st.sidebar.button("Load from GitHub", type="primary")

    if fetch_btn and github_url.strip():
        try:
            raw_url = normalize_github_url(github_url)
            if github_token.strip():
                import requests
                resp = requests.get(
                    raw_url,
                    headers={"Authorization": f"token {github_token.strip()}"},
                    timeout=15,
                )
                resp.raise_for_status()
                raw_bytes = resp.content
            else:
                raw_bytes, raw_url = load_dataset_from_github(github_url)

            parsed = _parse_csv_bytes(raw_bytes)
            if parsed is not None:
                df, source = parsed, "github"
                st.session_state["github_df"] = df
                st.sidebar.success(f"Loaded {len(df)} reviews from GitHub.")
            else:
                st.sidebar.error(
                    "Fetched the file but couldn't find 'Review' and 'Class' columns. "
                    "Falling back to sample data."
                )
        except Exception as e:
            st.sidebar.error(f"Failed to fetch from GitHub: {e}")

    # Keep using the previously fetched GitHub dataset across reruns
    if "github_df" in st.session_state and not fetch_btn:
        df, source = st.session_state["github_df"], "github"

elif data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload reviews CSV", type=["csv"])
    df, source = load_dataset(uploaded_file)

else:
    df, source = SAMPLE_DATA.copy(), "sample"

st.sidebar.success(f"Using {source} dataset ({len(df)} reviews)")
st.sidebar.dataframe(df, use_container_width=True, height=200)

# ----------------------------------------------------------------------
# Main tabs
# ----------------------------------------------------------------------
st.title("Dhurandhar 2 — Movie Review NLP Pipeline")
st.caption(
    "An end-to-end NLP pipeline covering sentiment analysis, translation, "
    "question answering, summarization, and model evaluation."
)

tab_sentiment, tab_translate, tab_qa, tab_summary, tab_eval = st.tabs(
    ["😀 Sentiment", "🌐 Translation", "❓ Question Answering", "📝 Summarization", "📊 Evaluation"]
)

# ----------------------------------------------------------------------
# Tab 1: Sentiment Analysis
# ----------------------------------------------------------------------
with tab_sentiment:
    st.subheader("Sentiment Analysis")
    st.write("Uses `distilbert-base-uncased-finetuned-sst-2-english` for POSITIVE/NEGATIVE classification.")

    col1, col2 = st.columns([2, 1])
    with col1:
        custom_review = st.text_area(
            "Try a custom review",
            value="KGF 2 is an amazing movie with powerful action and excellent performance.",
            height=100,
        )
    with col2:
        st.write("")
        st.write("")
        analyze_btn = st.button("Analyze Sentiment", type="primary", use_container_width=True)

    if analyze_btn and custom_review.strip():
        classifier = load_sentiment_pipeline()
        result = classifier(custom_review)[0]
        label, score = result["label"], result["score"]
        emoji = "🟢" if label == "POSITIVE" else "🔴"
        st.metric("Predicted Sentiment", f"{emoji} {label}", f"confidence: {score:.2%}")

    st.divider()
    st.write("**Batch sentiment analysis on the loaded dataset**")
    if st.button("Run on Full Dataset"):
        classifier = load_sentiment_pipeline()
        reviews = df["Review"].astype(str).tolist()
        predictions = classifier(reviews)
        results_df = df.copy()
        results_df["Predicted Sentiment"] = [p["label"] for p in predictions]
        results_df["Confidence"] = [round(p["score"], 4) for p in predictions]
        st.session_state["predicted_labels"] = predictions
        st.session_state["results_df"] = results_df
        st.dataframe(results_df, use_container_width=True)

# ----------------------------------------------------------------------
# Tab 2: Translation
# ----------------------------------------------------------------------
with tab_translate:
    st.subheader("Machine Translation (English → Spanish)")
    st.write("Uses `Helsinki-NLP/opus-mt-en-es`.")

    review_options = df["Review"].astype(str).tolist()
    selected_review = st.selectbox("Pick a review to translate", review_options, index=0)
    custom_text = st.text_area("Or enter your own text", value="", height=80)
    text_to_translate = custom_text.strip() if custom_text.strip() else selected_review

    if st.button("Translate", type="primary"):
        translation = translate_text(text_to_translate)
        st.write("**Original (EN):**", text_to_translate)
        st.write("**Translation (ES):**", translation)

# ----------------------------------------------------------------------
# Tab 3: Question Answering
# ----------------------------------------------------------------------
with tab_qa:
    st.subheader("Question Answering over a Review")
    st.write("Uses `distilbert-base-cased-distilled-squad` to extract answers from review text.")

    context = st.text_area(
        "Context (review text)",
        value=df["Review"].astype(str).iloc[0] if len(df) else "",
        height=100,
    )
    question = st.text_input("Question", value="What did the reviewer think of the movie?")

    if st.button("Get Answer", type="primary"):
        result = answer_question(question, context)
        st.write("**Answer:**", result["answer"])
        st.write(f"**Confidence:** {result['score']:.2%}")

# ----------------------------------------------------------------------
# Tab 4: Summarization
# ----------------------------------------------------------------------
with tab_summary:
    st.subheader("Text Summarization")
    st.write("Uses `sshleifer/distilbart-cnn-12-6` to generate concise summaries.")

    all_reviews_text = st.text_area(
        "Text to summarize (defaults to all reviews combined)",
        value=" ".join(df["Review"].astype(str).tolist()),
        height=150,
    )
    max_len = st.slider("Max summary length", 20, 200, 60)
    min_len = st.slider("Min summary length", 5, max_len - 1, min(15, max_len - 1))

    if st.button("Summarize", type="primary"):
        summary = summarize_text(all_reviews_text, max_length=max_len, min_length=min_len)
        st.write("**Summary:**")
        st.info(summary)

# ----------------------------------------------------------------------
# Tab 5: Evaluation
# ----------------------------------------------------------------------
with tab_eval:
    st.subheader("Model Evaluation")
    st.write("Compares predicted sentiment against the ground-truth `Class` column using Accuracy and F1 Score.")

    if st.button("Evaluate Sentiment Model", type="primary"):
        classifier = load_sentiment_pipeline()
        metrics = load_metrics()

        reviews = df["Review"].astype(str).tolist()
        real_labels = df["Class"].astype(str).tolist()
        predicted_labels = classifier(reviews)

        references = [1 if label.upper() == "POSITIVE" else 0 for label in real_labels]
        predictions = [1 if p["label"].upper() == "POSITIVE" else 0 for p in predicted_labels]

        accuracy_result = metrics["accuracy"].compute(references=references, predictions=predictions)["accuracy"]
        f1_result = metrics["f1"].compute(references=references, predictions=predictions)["f1"]

        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{accuracy_result:.2%}")
        col2.metric("F1 Score", f"{f1_result:.2%}")

        detail_df = df.copy()
        detail_df["Predicted Sentiment"] = [p["label"] for p in predicted_labels]
        detail_df["Confidence"] = [round(p["score"], 4) for p in predicted_labels]
        detail_df["Correct"] = [
            r.upper() == p["label"].upper() for r, p in zip(real_labels, predicted_labels)
        ]
        st.dataframe(detail_df, use_container_width=True)

    st.divider()
    st.write("**BLEU Score for Translation** (requires reference translations)")
    st.caption(
        "Provide one reference Spanish translation per line, matching the order of reviews, "
        "to compute a BLEU score against the model's translations."
    )
    references_text = st.text_area("Reference translations (one per line)", height=100)

    if st.button("Compute BLEU Score"):
        refs = [r.strip() for r in references_text.strip().split("\n") if r.strip()]
        reviews = df["Review"].astype(str).tolist()
        if len(refs) != len(reviews):
            st.error(f"Expected {len(reviews)} reference lines (one per review), got {len(refs)}.")
        else:
            metrics = load_metrics()
            predictions = [translate_text(r) for r in reviews]
            bleu_result = metrics["bleu"].compute(
                predictions=predictions, references=[[r] for r in refs]
            )
            st.metric("BLEU Score", f"{bleu_result['bleu']:.4f}")
            st.dataframe(
                pd.DataFrame({"Review": reviews, "Model Translation": predictions, "Reference": refs}),
                use_container_width=True,
            )

# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.divider()
st.caption(
    "Built with 🤗 Transformers + Streamlit — adapted from the "
    "Dhurandhar_2_review_with_LLM notebook."
)
