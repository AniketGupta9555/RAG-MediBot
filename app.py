"""
Full app.py — Local RAG Medibot using:
  - Local embeddings: sentence-transformers/all-MiniLM-L6-v2
  - Vector DB: Pinecone (your existing index, e.g. medibot-rag-d384)
  - Local LLM: Ollama (HTTP API at http://localhost:11434, model e.g. llama3.1:8b)

Place this file at the root of your project. Ensure:
  - templates/chat.html exists (simple chat UI)
  - .env contains PINECONE_API_KEY and PINECONE_INDEX (and optional PORT/TOP_K)
  - Ollama server is running (ollama serve) and model pulled (ollama pull ...)
  - Pinecone index already has your embeddings uploaded

Run:
  python app.py
"""

import os
import traceback
from typing import List

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import requests

# Pinecone (new package name)
from pinecone import Pinecone

# local embedder
from sentence_transformers import SentenceTransformer

load_dotenv()

# ---------- Config ----------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "medibot-rag-d384")
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "2500"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
PORT = int(os.getenv("PORT", "5000"))
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))  # seconds

# ---------- Initialize ----------
app = Flask(__name__, template_folder="templates")

print("Initializing local embedder (all-MiniLM-L6-v2)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedder loaded.")

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not set in environment")

print("Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

# rename variable to avoid clobbering by route function
pinecone_index = pc.Index(PINECONE_INDEX)
print(f"Connected to Pinecone index: {PINECONE_INDEX}")

# ---------- Utility functions ----------


def embed_query_local(text: str) -> List[float]:
    """Return 1-D list float vector for a query string."""
    vec = embedder.encode(text, show_progress_bar=False)
    try:
        return vec.tolist()
    except Exception:
        return [float(x) for x in vec]


def query_pinecone(vec: List[float], top_k: int = TOP_K):
    """
    Query Pinecone and return normalized list of matches.
    Each match is normalized to a dict with keys: id, score, metadata.
    """
    res = pinecone_index.query(vector=vec, top_k=top_k, include_metadata=True)
    matches = []
    # support both dict and object-style responses across SDK versions
    if isinstance(res, dict):
        raw_matches = res.get("matches", []) or []
    else:
        raw_matches = getattr(res, "matches", []) or []

    for m in raw_matches:
        if isinstance(m, dict):
            matches.append(
                {
                    "id": m.get("id"),
                    "score": m.get("score"),
                    "metadata": m.get("metadata", {}),
                }
            )
        else:
            md = {}
            try:
                md = getattr(m, "metadata", {}) or {}
            except Exception:
                md = {}
            matches.append(
                {
                    "id": getattr(m, "id", None),
                    "score": getattr(m, "score", None),
                    "metadata": md,
                }
            )
    return matches


def build_context_from_matches(matches, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    Build a concatenated context string from the top matches' metadata.
    Looks for 'preview' or 'text' fields in metadata.
    """
    parts = []
    total = 0
    for m in matches:
        meta = m.get("metadata", {}) or {}
        preview = ""
        if isinstance(meta, dict):
            preview = meta.get("preview") or meta.get("text") or meta.get("content") or meta.get("chunk") or ""
        else:
            try:
                preview = getattr(meta, "preview", "") or getattr(meta, "text", "")
            except Exception:
                preview = ""

        if not preview:
            if isinstance(meta, str) and meta.strip():
                preview = meta.strip()
        if not preview:
            continue

        if total + len(preview) > max_chars:
            allowed = max_chars - total
            parts.append(preview[:allowed])
            break

        parts.append(preview)
        total += len(preview)

    return "\n\n".join(parts).strip()


def generate_answer_ollama(context: str, question: str) -> str:
    """
    Call local Ollama HTTP API. If Ollama call fails, return an extractive fallback.
    """
    system_text = (
        "You are Medibot, a medically-informed assistant. You MUST NOT provide definitive diagnoses. "
        "Always recommend consulting a qualified healthcare professional. "
        "If the user describes an emergency (chest pain, severe bleeding, difficulty breathing, sudden weakness or slurred speech), instruct them to call emergency services immediately."
    )

    prompt = (
        f"{system_text}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Using ONLY the context above and general medical knowledge, answer concisely and safely. If the context is insufficient, say so and recommend seeing a doctor."
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        j = resp.json()
        if isinstance(j, dict) and "response" in j:
            return j.get("response", "").strip()
        return str(j)
    except Exception as e:
        print("Ollama call failed:", repr(e))
        return ollama_fallback_extractive(context, note_error=str(e))


def ollama_fallback_extractive(context: str, note_error: str = "") -> str:
    """Return a short extractive fallback built from context snippets."""
    if not context or not context.strip():
        msg = "Local LLM unavailable and no retrieved context. Please try again later or consult a medical professional."
        if note_error:
            msg += f" (error: {note_error})"
        return msg

    snippets = [p.strip() for p in context.split("\n\n") if p.strip()]
    top = snippets[:3]
    summary = "\n\n".join(top)

    fallback = (
        "The local language model is currently unavailable or returned an error. "
        "Below are the most relevant excerpts from the documents:\n\n"
        f"{summary}\n\n"
        "This is NOT a diagnosis. If you have severe or urgent symptoms (chest pain, difficulty breathing, heavy bleeding), call emergency services immediately. "
        "For a full evaluation, consult a qualified healthcare professional."
    )
    if note_error:
        fallback += f"\n\n[debug: {note_error}]"
    return fallback


# ---------- Flask routes ----------
@app.route("/")
def home():
    try:
        return render_template("chat.html")
    except Exception:
        return "<h3>Medibot RAG</h3><p>UI not found. Create templates/chat.html or use /ask endpoint.</p>"


@app.route("/ask", methods=["POST"])
def ask():
    """
    Body JSON: { "question": "<text>" }
    Returns: { "answer": "...", "context_preview": "..." } or error JSON.
    """
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "empty question"}), 400

    try:
        q_vec = embed_query_local(question)
        matches = query_pinecone(q_vec, top_k=TOP_K)
        context = build_context_from_matches(matches, max_chars=MAX_CONTEXT_CHARS)
        answer = generate_answer_ollama(context, question)
        preview = context[:1000] + ("..." if len(context) > 1000 else "")
        return jsonify({"answer": answer, "context_preview": preview})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return {"status": "ok", "index": PINECONE_INDEX}


# ---------- Run ----------
if __name__ == "__main__":
    print(f"Starting Medibot (Pinecone index: {PINECONE_INDEX}) on port {PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=True)
