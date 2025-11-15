# embed_texts_groq.py
"""
Reads chunks.jsonl and produces embeddings.jsonl using Groq (preferred) or local sentence-transformers fallback.
Outputs lines:
{"id": "...", "embedding": [...], "metadata": {...}}
"""

import os
import time
import json
from pathlib import Path
from dotenv import load_dotenv
import requests
from tqdm import tqdm

load_dotenv()

CHUNKS_FILE = Path("chunks.jsonl")
OUT_FILE = Path("embeddings.jsonl")

# Groq config
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "embed-english-v1").strip()
GROQ_EMBED_URL = "https://api.groq.ai/v1/embeddings"

# Tunables
BATCH_SLEEP = float(os.getenv("BATCH_SLEEP", "0.08"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))

# Local fallback
USE_LOCAL_FALLBACK = not bool(GROQ_API_KEY)

# Try to import sentence-transformers only if needed
local_embedder = None
if USE_LOCAL_FALLBACK:
    try:
        from sentence_transformers import SentenceTransformer
        local_embedder = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        print("Local fallback requested but sentence-transformers failed to import or load:", e)
        print("Install sentence-transformers and torch or set GROQ_API_KEY in .env to use Groq.")
        raise

def groq_embed_one(text):
    """
    Call Groq embeddings endpoint for a single text.
    Returns list[float].
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "input": text
    }
    attempt = 0
    while True:
        attempt += 1
        try:
            r = requests.post(GROQ_EMBED_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                print(f"[groq] status={r.status_code} body={r.text[:1000]}")
            r.raise_for_status()
            data = r.json()
            # Groq typical response: {"data":[{"embedding":[...], "index":0}], ...}
            if "data" in data and isinstance(data["data"], list) and len(data["data"])>0:
                emb = data["data"][0].get("embedding") or data["data"][0]
                return emb
            # if direct top-level
            if "embedding" in data:
                return data["embedding"]
            raise ValueError("Unexpected Groq embedding response: " + str(data)[:1000])
        except requests.HTTPError as he:
            status = getattr(he.response, "status_code", None)
            if status and 400 <= status < 500:
                # client errors: do not retry forever
                raise
            if attempt >= MAX_RETRIES:
                raise
            backoff = 2 ** attempt
            print(f"[groq] transient HTTP error, retrying in {backoff}s...")
            time.sleep(backoff)
        except Exception as e:
            if attempt >= MAX_RETRIES:
                raise
            backoff = 2 ** attempt
            print(f"[groq] transient error: {e}. retrying in {backoff}s...")
            time.sleep(backoff)

def local_embed_one(text):
    """
    Use sentence-transformers model to embed one text.
    Returns list[float].
    """
    global local_embedder
    emb = local_embedder.encode(text, show_progress_bar=False)
    # convert numpy to python list if necessary
    try:
        return emb.tolist()
    except Exception:
        return list(map(float, emb))

def embed_text(text):
    if not USE_LOCAL_FALLBACK:
        return groq_embed_one(text)
    else:
        return local_embed_one(text)

def main():
    if not CHUNKS_FILE.exists():
        print("chunks.jsonl not found. Run extract_texts.py first.")
        return

    # read chunks
    chunks = []
    with CHUNKS_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except Exception as e:
                print("Skipping malformed line:", e)

    total = len(chunks)
    print("Total chunks to embed:", total)
    if total == 0:
        return

    # If Groq selected, test a quick call first (to fail early)
    if not USE_LOCAL_FALLBACK:
        print("Using Groq embeddings with model:", GROQ_MODEL)
    else:
        print("Using local sentence-transformers fallback: all-MiniLM-L6-v2")

    # Write output; overwrite to start fresh
    with OUT_FILE.open("w", encoding="utf-8") as outfh:
        for i, c in enumerate(tqdm(chunks, desc="Embedding chunks", unit="chunk"), start=1):
            cid = c.get("id") or f"chunk_{i}"
            try:
                emb = embed_text(c["text"])
            except Exception as e:
                print(f"❌ Failed to embed id={cid}: {e}")
                raise
            metadata = {
                "source": c.get("source"),
                "page": c.get("page"),
                "chunk_id": c.get("chunk_id"),
                "preview": c.get("text")[:300]
            }
            out = {"id": cid, "embedding": emb, "metadata": metadata}
            outfh.write(json.dumps(out, ensure_ascii=False) + "\n")
            time.sleep(BATCH_SLEEP)

    print("Wrote", OUT_FILE)

if __name__ == "__main__":
    main()
