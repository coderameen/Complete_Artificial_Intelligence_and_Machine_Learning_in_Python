#pip install streamlit ollama faiss-cpu sentence-transformers numpy
import os
import faiss
import numpy as np
import streamlit as st
import ollama
from sentence_transformers import SentenceTransformer

# ----------------------------
# Configuration
# ----------------------------

OLLAMA_MODEL = "llama3.2:1b"
DOCUMENT_FOLDER = "documents"
CHUNK_SIZE = 500

st.set_page_config(page_title="Simple RAG Chatbot")
st.title("🤖 Simple RAG Chatbot (Ollama + Streamlit)")

# ----------------------------
# Load Embedding Model
# ----------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# ----------------------------
# Read Documents
# ----------------------------

def load_documents():
    texts = []

    if not os.path.exists(DOCUMENT_FOLDER):
        return texts

    for file in os.listdir(DOCUMENT_FOLDER):
        if file.endswith(".txt"):
            path = os.path.join(DOCUMENT_FOLDER, file)

            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())

    return texts

# ----------------------------
# Split into Chunks
# ----------------------------

def chunk_text(text):
    chunks = []

    for i in range(0, len(text), CHUNK_SIZE):
        chunks.append(text[i:i + CHUNK_SIZE])

    return chunks

# ----------------------------
# Build Vector Database
# ----------------------------

@st.cache_resource
def build_vector_store():

    documents = load_documents()

    all_chunks = []

    for doc in documents:
        all_chunks.extend(chunk_text(doc))

    if len(all_chunks) == 0:
        return None, []

    embeddings = embedding_model.encode(all_chunks)

    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, all_chunks

index, chunks = build_vector_store()

# ----------------------------
# Retrieve Context
# ----------------------------

def retrieve(query, top_k=3):

    if index is None:
        return ""

    query_embedding = embedding_model.encode([query]).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    context = []

    for idx in indices[0]:
        if idx < len(chunks):
            context.append(chunks[idx])

    return "\n\n".join(context)

# ----------------------------
# Chat History
# ----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ----------------------------
# Chat Input
# ----------------------------

question = st.chat_input("Ask something...")

if question:

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.write(question)

    context = retrieve(question)

    prompt = f"""
You are a helpful AI assistant.

Use ONLY the provided context to answer.

If the answer is not found, say:
"I couldn't find that information in the documents."

Context:
{context}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    with st.chat_message("assistant"):
        st.write(answer)