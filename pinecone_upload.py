# upload_pinecone_safe.py
import os, json, time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
INDEX_NAME = os.getenv("PINECONE_INDEX", "medibot-rag")
EMBEDDINGS_FILE = "embeddings.jsonl"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))

def detect_dim(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            return len(obj["embedding"])
    raise RuntimeError("No embeddings found in file")

def get_existing_indexes(pc):
    try:
        # new SDK returns an object with .names()
        existing = pc.list_indexes().names()
    except Exception:
        try:
            existing = pc.list_indexes()
        except Exception:
            existing = []
    # normalize to list of strings
    if isinstance(existing, (list, tuple)):
        return list(existing)
    return list(existing)

def try_get_index_dimension(pc, index_name):
    # Try several methods because SDKs differ
    # 1) Try pc.describe_index(index_name)
    try:
        info = None
        if hasattr(pc, "describe_index"):
            info = pc.describe_index(index_name)
            # info might be dict or object
            if isinstance(info, dict):
                dim = info.get("dimension") or (info.get("index") or {}).get("dimension")
                if dim:
                    return int(dim)
        # 2) Try calling pc.Index(index_name).describe_index() (some SDKs)
        try:
            idx = pc.Index(index_name)
            if hasattr(idx, "describe_index"):
                info2 = idx.describe_index()
                if isinstance(info2, dict):
                    dim = info2.get("dimension") or (info2.get("index") or {}).get("dimension")
                    if dim:
                        return int(dim)
        except Exception:
            pass
        # 3) Some SDKs expose describe_index_stats but it doesn't give dimension reliably => skip
    except Exception:
        pass
    return None

def main():
    dim = detect_dim(EMBEDDINGS_FILE)
    print("Detected embedding dimension:", dim)

    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing = get_existing_indexes(pc)
    print("Existing Pinecone indexes:", existing)

    target_index = INDEX_NAME

    if INDEX_NAME in existing:
        existing_dim = try_get_index_dimension(pc, INDEX_NAME)
        if existing_dim is not None:
            print(f"Index '{INDEX_NAME}' exists with dimension {existing_dim}")
            if int(existing_dim) != int(dim):
                # dimension mismatch -> create new index name
                new_name = f"{INDEX_NAME}-d{dim}"
                print(f"Dimension mismatch. Will create and use new index: '{new_name}'")
                target_index = new_name
        else:
            # could not determine existing dimension -> be conservative and create a new index
            new_name = f"{INDEX_NAME}-d{dim}"
            print(f"Could not read existing index dimension. Will create and use new index: '{new_name}'")
            target_index = new_name
    else:
        print(f"Index '{INDEX_NAME}' does not exist. Will create it.")

    # Create target index if missing
    if target_index not in existing:
        print(f"Creating index '{target_index}' with dimension {dim}...")
        pc.create_index(
            name=target_index,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
        )
        # small wait
        time.sleep(1)

    index = pc.Index(target_index)
    print("Uploading vectors to index:", target_index)

    items = []
    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            items.append((obj["id"], obj["embedding"], obj.get("metadata", {})))

    print("Total vectors to upload:", len(items))
    for i in tqdm(range(0, len(items), BATCH_SIZE), desc="Upserting"):
        batch = items[i:i+BATCH_SIZE]
        vectors = [{"id": vid, "values": emb, "metadata": meta} for vid, emb, meta in batch]
        index.upsert(vectors=vectors)

    print("Upload finished to index:", target_index)
    if target_index != INDEX_NAME:
        print()
        print("NOTE: your original index was preserved. If you wish to replace it, delete it manually from the Pinecone console or via API.")
        print(f"Your new index is: {target_index}")

if __name__ == "__main__":
    main()
