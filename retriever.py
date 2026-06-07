import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = "all-MiniLM-L6-v2"
CHROMA_PATH = "./chroma_db"
TOP_K = 3
MIN_DISTANCE = 2.0        # ← raised very high so nothing gets blocked

_model = SentenceTransformer(EMBED_MODEL)
_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _client.get_collection("college_docs")
_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

def retrieve(query):
    query_embed = _model.encode([query]).tolist()
    results = _collection.query(
        query_embeddings=query_embed,
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )
    chunks = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    return chunks, metas, distances

def build_prompt(query, chunks, metas):
    context_parts = []
    for chunk, meta in zip(chunks, metas):
        context_parts.append(
            f"[Source: {meta['doc_name']}, Page {meta['page']}]\n{chunk}"
        )
    context = "\n\n".join(context_parts)
    return f"""You are a helpful FAQ assistant for a college.
Answer ONLY from the context below. If the context does not contain the answer, say:
"I don't have enough information in the college documents to answer that."

Context:
{context}

Question: {query}
Answer:"""

def answer(query):
    chunks, metas, distances = retrieve(query)

    best_distance = distances[0] if distances else 1.0
    low_confidence = best_distance > MIN_DISTANCE

    if low_confidence:
        return {
            "answer": "I don't have reliable information about that in the college documents. Please contact the administration office.",
            "sources": [],
            "confidence": "low"
        }

    prompt = build_prompt(query, chunks, metas)
    chat = _groq.chat.completions.create(
# NEW - current model
model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512
    )
    llm_answer = chat.choices[0].message.content.strip()

    seen, sources = set(), []
    for meta in metas:
        key = (meta["doc_name"], meta["page"])
        if key not in seen:
            seen.add(key)
            sources.append(meta)

    return {
        "answer": llm_answer,
        "sources": sources,
        "confidence": "high"
    }