from dotenv import load_dotenv
import os, requests
load_dotenv()

MODEL = "embedding-gecko-001"   # force correct model
url = f"https://generativelanguage.googleapis.com/v1/models/{MODEL}:embedText"

payload = {
    "model": MODEL,
    "text": "This is a final test embedding after unlocking my account."
}

resp = requests.post(
    url,
    params={"key": os.getenv("GEMINI_API_KEY")},
    json=payload,
    timeout=30
)

print("STATUS:", resp.status_code)
print(resp.text[:500])
