# main.py

import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load data
with open("data.txt", "r") as f:
    raw = f.read()

chunks = [line.strip() for line in raw.split(".") if line.strip()]

# Create or load FAISS index
if not os.path.exists("faiss_index.index"):
    print("🔧 Creating FAISS index...")
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    faiss.write_index(index, "faiss_index.index")
    np.save("chunks.npy", chunks)
else:
    print("📂 Loading FAISS index...")
    index = faiss.read_index("faiss_index.index")
    chunks = np.load("chunks.npy", allow_pickle=True)


# Retrieve context
def get_context(query, top_k=2):
    query_embedding = embedder.encode([query])
    D, I = index.search(query_embedding, top_k)
    return "\n".join([chunks[i] for i in I[0]])


# Generate response with Ollama
def ask_ollama(question, context):
    prompt = f"""
You are a helpful assistant. Use the context provided below to answer the question completely and clearly.

### Context:
{context}

### Question:
{question}

### Answer:
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2", "prompt": prompt, "stream": False},
    )

    return response.json().get("response", "No answer returned.")


# Main loop
print("\n🤖 Ask me anything based on data.txt!")
while True:
    q = input("\nYou: ")
    if q.lower() in ["exit", "quit"]:
        break
    ctx = get_context(q)
    answer = ask_ollama(q, ctx)
    print(f"Bot: {answer}")
