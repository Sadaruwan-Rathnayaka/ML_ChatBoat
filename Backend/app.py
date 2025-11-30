"""
app.py

Flask backend for a RAG (PDF → embeddings → Retriever → Generator) chatbot.

Endpoints:
- POST /upload  -> upload PDF file, extract text, chunk, embed, and persist knowledge base
- POST /ask     -> ask a question (JSON {"question": "..."}), returns answer + sources
- GET  /health  -> simple health check

Notes:
- Expects GEMINI_API_KEY in environment variables.
- Uses google.generativeai for embeddings and generation (same as your Colab code).
- Stores knowledge base to disk as 'knowledge_base.pkl' to avoid re-embedding.
- Uses simple cosine similarity (numpy) for retrieval. For production, swap to FAISS or another ANN index.

Run (inside your venv):
1. pip install -r requirements.txt
2. Set GEMINI_API_KEY in env
3. python app.py

Requirements (requirements.txt):
flask
flask-cors
google-generativeai
pypdf
numpy
joblib
"""
import os
import io
import threading
import pickle
import time
from typing import List, Dict, Tuple, Any

from flask import Flask, request, jsonify
from flask_cors import CORS
import pypdf
import numpy as np
import google.generativeai as genai

from dotenv import load_dotenv
load_dotenv()

# ----------------- Configuration -----------------
KB_FILE = "knowledge_base.pkl"
CHUNK_MIN_CHARS = 50
TOP_K = 3
EMBEDDING_MODEL = "models/text-embedding-004"
GEN_MODEL_NAME = "gemini-2.5-flash-preview-09-2025"
SYSTEM_INSTRUCTION = (
    "You are a helpful assistant. Your task is to answer the user's question based *only* on the provided context.\n"
    "- Do not use any external knowledge.\n"
    "- If the answer is not found in the context, you *must* state: \"I'm sorry, but that information is not available in the provided document.\"\n"
    "- Be concise and directly answer the question using only the information from the context."
)

# Retry config for API calls
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds

# Thread lock for safe access to the KB file
kb_lock = threading.Lock()

app = Flask(__name__)
CORS(app)

# Initialize Gemini (expects GEMINI_API_KEY environment variable)
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Please set GEMINI_API_KEY environment variable before running the server.")

genai.configure(api_key=api_key)

# Create generation model handle
generation_model = genai.GenerativeModel(model_name=GEN_MODEL_NAME, system_instruction=SYSTEM_INSTRUCTION)

# ----------------- Utilities -----------------


def retry_api_call(fn, *args, **kwargs) -> Any:
    """Simple retry wrapper for transient API errors."""
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            app.logger.warning(f"API call failed (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
    # raise the last exception after retries
    app.logger.exception("API call failed after retries")
    raise last_exc


def extract_and_chunk_pdf_bytes(file_bytes: bytes) -> List[str]:
    """Extract text from PDF bytes and produce text chunks."""
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        pages.append(page.extract_text() or "")

    # join pages with double newlines, then split on double newlines
    joined = "\n\n".join(pages)
    raw_chunks = joined.split("\n\n")
    chunks = [c.strip() for c in raw_chunks if len(c.strip()) >= CHUNK_MIN_CHARS]
    return chunks


def _clean_embedding_result(result) -> List[float]:
    """Normalize various SDK return shapes into a plain python list of floats."""
    if result is None:
        return []
    # SDK might return dict with "embedding" key, or maybe directly a list
    if isinstance(result, dict) and "embedding" in result:
        emb = result["embedding"]
    else:
        emb = result
    # Convert numpy to list, or leave list
    if isinstance(emb, np.ndarray):
        emb = emb.tolist()
    return list(emb)


def embed_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
    """Return embedding vector for given text (as a plain python list)."""
    def call():
        return genai.embed_content(model=EMBEDDING_MODEL, content=text, task_type=task_type)
    result = retry_api_call(call)
    emb = _clean_embedding_result(result)
    return emb


def persist_kb(kb: List[Dict]):
    with kb_lock:
        with open(KB_FILE, "wb") as f:
            pickle.dump(kb, f)


def load_kb() -> List[Dict]:
    if not os.path.exists(KB_FILE):
        return []
    with kb_lock:
        with open(KB_FILE, "rb") as f:
            return pickle.load(f)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def find_most_similar_chunks_local(query: str, kb: List[Dict], top_k: int = TOP_K) -> Tuple[str, List[Dict]]:
    """Return concatenated context and list of source metadata for top_k matches."""
    if not kb:
        return "", []

    query_emb = embed_text(query, task_type="RETRIEVAL_QUERY")
    if not query_emb:
        return "", []

    qv = np.array(query_emb)

    sims = []
    for item in kb:
        iv = np.array(item["embedding"])
        score = cosine_similarity(qv, iv)
        sims.append({"score": score, "text": item["text"], "meta": item.get("meta")})

    sims.sort(key=lambda x: x["score"], reverse=True)
    top = sims[:top_k]
    context = "\n---\n".join([t["text"] for t in top])
    sources = [t.get("meta") for t in top]
    return context, sources


def generate_answer(query: str, context: str) -> str:
    final_prompt = f"""
Context:
---
{context}
---

Question:
{query}

Answer:
"""
    def call():
        return generation_model.generate_content(
            final_prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.0)
        )
    response = retry_api_call(call)
    # handle common response shapes
    if hasattr(response, "text"):
        return response.text
    if isinstance(response, dict) and "output" in response:
        return str(response["output"])
    return str(response)


# ----------------- Routes -----------------


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})


@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        file_bytes = f.read()
        chunks = extract_and_chunk_pdf_bytes(file_bytes)
        if not chunks:
            return jsonify({"error": "No extractable text found in PDF."}), 400

        kb = []
        for i, c in enumerate(chunks):
            emb = embed_text(c, task_type="RETRIEVAL_DOCUMENT")
            # normalize and store as plain list (recommended)
            if emb:
                arr = np.array(emb, dtype=float)
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = (arr / norm).tolist()
                else:
                    arr = arr.tolist()
            else:
                arr = []
            kb.append({"text": c, "embedding": arr, "meta": {"chunk_index": i}})

        persist_kb(kb)
        return jsonify({"status": "ok", "chunks_stored": len(kb)})
    except Exception:
        app.logger.exception("Upload error")
        return jsonify({"error": "Failed to process file."}), 500


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' in JSON body"}), 400

    question = data["question"]
    kb = load_kb()
    if not kb:
        return jsonify({"answer": "I'm sorry, but that information is not available in the provided document.", "sources": []})

    context, sources = find_most_similar_chunks_local(question, kb, top_k=TOP_K)
    if not context.strip():
        return jsonify({"answer": "I'm sorry, but that information is not available in the provided document.", "sources": []})

    answer = generate_answer(question, context)
    return jsonify({"answer": answer, "sources": sources})


if __name__ == "__main__":
    # In development use: python app.py
    # For production use a WSGI server such as gunicorn
    app.run(host="0.0.0.0", port=5000, debug=True)
