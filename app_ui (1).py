import streamlit as st
import fitz
import pandas as pd
import faiss
import numpy as np
import pickle
import os

from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ==========================================
# LOAD MODELS (CACHE)
# ==========================================
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
    generator = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return summarizer, generator, embedder

summarizer, generator, embedder = load_models()

# ==========================================
# FUNCTIONS
# ==========================================
def extract_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return " ".join([p.get_text() for p in doc])

def extract_csv(file):
    df = pd.read_csv(file).fillna("")
    rows = []
    for _, row in df.iterrows():
        text = " | ".join([f"{col}: {val}" for col, val in row.items() if str(val).strip()])
        if text:
            rows.append(text)
    return " ".join(rows)

def split_text(text, chunk_size=400):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def safe_truncate(text):
    tokens = summarizer.tokenizer(
        text,
        truncation=True,
        max_length=1024,
        return_tensors="pt"
    )
    return summarizer.tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)

def summarize_batch(chunks):
    chunks = [safe_truncate(c) for c in chunks]
    outputs = summarizer(chunks, max_length=80, min_length=25, do_sample=False, batch_size=4)
    return [o["summary_text"] for o in outputs]

def hierarchical_summary(text):
    chunks = split_text(text)
    summaries = summarize_batch(chunks)

    while len(summaries) > 1:
        grouped = [" ".join(summaries[i:i+5]) for i in range(0, len(summaries), 5)]
        summaries = summarize_batch(grouped)

    return summaries[0]

def bullet_summary(text):
    prompt = f"""
Give exactly 5 bullet points.

- point 1
- point 2
- point 3
- point 4
- point 5

Text:
{text}
"""
    return generator(prompt, max_length=200)[0]["generated_text"]

def generate_questions(text):
    prompt = f"""
Generate 5 questions:

1.
2.
3.
4.
5.

Text:
{text}
"""
    return generator(prompt, max_length=200)[0]["generated_text"]

def build_index(chunks):
    embeddings = embedder.encode(chunks, batch_size=32)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index

def retrieve(query, index, chunks, k=3):
    q_vec = embedder.encode([query])
    _, idx = index.search(np.array(q_vec), k)
    return [chunks[i] for i in idx[0]]

def answer(query, context_chunks):
    context = " ".join(context_chunks)
    prompt = f"""
Answer using context:

{context}

Question: {query}
"""
    return generator(prompt, max_length=200)[0]["generated_text"]

# ==========================================
# UI
# ==========================================
st.set_page_config(page_title="CLASSMATE LLM", layout="wide")

st.title("📚 CLASSMATE LLM")
st.write("Upload a PDF or CSV and interact with it")

uploaded_file = st.file_uploader("Upload File", type=["pdf", "csv"])

if uploaded_file:

    if "text" not in st.session_state:
        with st.spinner("Reading document..."):
            if uploaded_file.type == "application/pdf":
                text = extract_pdf(uploaded_file)
            else:
                text = extract_csv(uploaded_file)

            st.session_state.text = text
            st.session_state.chunks = split_text(text)

    text = st.session_state.text
    chunks = st.session_state.chunks

    st.success(f"Document loaded ({len(text.split())} words)")

    # ==========================================
    # SUMMARY
    # ==========================================
    if st.button("Generate Summary"):
        with st.spinner("Summarizing..."):
            summary = hierarchical_summary(text)
            st.session_state.summary = summary

    if "summary" in st.session_state:
        st.subheader("📌 Summary")
        st.write(st.session_state.summary)

        st.subheader("📌 Bullet Points")
        st.write(bullet_summary(st.session_state.summary))

        st.subheader("❓ Questions")
        st.write(generate_questions(st.session_state.summary))

    # ==========================================
    # RAG CHAT
    # ==========================================
    if "index" not in st.session_state:
        with st.spinner("Building search index..."):
            st.session_state.index = build_index(chunks)

    st.subheader("💬 Ask Questions")

    query = st.text_input("Type your question:")

    if query:
        with st.spinner("Thinking..."):
            ctx = retrieve(query, st.session_state.index, chunks)
            response = answer(query, ctx)
            st.write("### Answer")
            st.write(response)