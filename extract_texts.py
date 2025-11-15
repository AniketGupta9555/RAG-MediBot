# extract_texts.py
import os, json, time
from pathlib import Path
from dotenv import load_dotenv
import fitz  # pymupdf
from pdf2image import convert_from_path
import pytesseract

load_dotenv()

PDF_DIR = Path("pdfs")
OUT_FILE = Path("chunks.jsonl")

POPPLER_PATH = os.getenv("POPPLER_PATH", "")
CHUNK_SIZE_WORDS = int(os.getenv("CHUNK_SIZE_WORDS", "300"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "50"))
OCR_THRESHOLD_CHARS = 60
PDF_DPI_FOR_OCR = 200

def extract_chunks_from_pdf_mupdf(pdf_path: str, chunk_size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS):
    doc = fitz.open(pdf_path)
    chunks = []
    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        text = page.get_text("text") or ""
        if len(text.strip()) < OCR_THRESHOLD_CHARS:
            try:
                if POPPLER_PATH:
                    pil_images = convert_from_path(pdf_path, first_page=page_index+1, last_page=page_index+1, dpi=PDF_DPI_FOR_OCR, poppler_path=POPPLER_PATH)
                else:
                    pil_images = convert_from_path(pdf_path, first_page=page_index+1, last_page=page_index+1, dpi=PDF_DPI_FOR_OCR)
                if pil_images:
                    img = pil_images[0]
                    text = pytesseract.image_to_string(img)
            except Exception as e:
                print(f"⚠️ OCR fallback failed for {pdf_path} page {page_index+1}: {e}")
                text = ""
        if not text.strip():
            continue
        words = text.split()
        i = 0
        chunk_id = 0
        while i < len(words):
            end = i + chunk_size
            chunk_text = " ".join(words[i:end]).strip()
            if chunk_text:
                chunks.append({
                    "id": f"{Path(pdf_path).stem}_p{page_index+1}_c{chunk_id}",
                    "source": Path(pdf_path).name,
                    "page": page_index + 1,
                    "chunk_id": chunk_id,
                    "text": chunk_text
                })
            chunk_id += 1
            i += (chunk_size - overlap)
    doc.close()
    return chunks

def main():
    files = list(PDF_DIR.glob("*.pdf"))
    if not files:
        print("No PDFs in", PDF_DIR)
        return
    # overwrite output
    with OUT_FILE.open("w", encoding="utf-8") as fout:
        total = 0
        for p in files:
            print("Processing", p.name)
            chunks = extract_chunks_from_pdf_mupdf(str(p))
            for c in chunks:
                fout.write(json.dumps(c, ensure_ascii=False) + "\n")
            print(f"  extracted {len(chunks)} chunks")
            total += len(chunks)
    print("Wrote", OUT_FILE, "total chunks:", total)

if __name__ == "__main__":
    main()
