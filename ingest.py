import fitz
import chromadb
from sentence_transformers import SentenceTransformer
import os
import glob
import pytesseract
from PIL import Image
import io

# Tesseract path — change this if you installed in a different location
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

PDF_DIR = "./data"
CHROMA_PATH = "./chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBED_MODEL = "all-MiniLM-L6-v2"

def extract_text_from_page(page):
    # Try normal text extraction first
    text = page.get_text().strip()
    if len(text) > 50:
        return text

    # Page is scanned — use OCR
    print(f"      Using OCR on this page...")
    pix = page.get_pixmap(dpi=300)
    img_bytes = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_bytes))
    ocr_text = pytesseract.image_to_string(img, lang='eng')
    return ocr_text.strip()

def extract_chunks(pdf_path):
    doc = fitz.open(pdf_path)
    doc_name = os.path.basename(pdf_path)
    chunks = []
    total_pages = len(doc)

    for page_num, page in enumerate(doc, start=1):
        print(f"    Page {page_num}/{total_pages}", end="\r")
        text = extract_text_from_page(page)

        if not text or len(text) < 50:
            continue

        for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk = text[i : i + CHUNK_SIZE].strip()
            if len(chunk) < 80:
                continue
            chunks.append({
                "text": chunk,
                "doc_name": doc_name,
                "page": page_num
            })
    return chunks

def build_vectorstore():
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection("college_docs")
        print("Cleared existing collection.")
    except Exception as e:
        print(f"No existing collection: {e}")

    collection = client.create_collection("college_docs")

    pdfs = glob.glob(f"{PDF_DIR}/*.pdf")
    print(f"Found {len(pdfs)} PDF(s)\n")

    total_chunks = 0
    skipped = 0

    for pdf in pdfs:
        print(f"Processing: {os.path.basename(pdf)}")
        chunks = extract_chunks(pdf)

        if not chunks:
            print(f"  Skipped - no text found even after OCR\n")
            skipped += 1
            continue

        texts = [c["text"] for c in chunks]
        embeds = model.encode(texts, show_progress_bar=True).tolist()

        if not embeds:
            print(f"  Skipped - embedding failed\n")
            skipped += 1
            continue

        ids = [f"{os.path.basename(pdf)}_p{c['page']}_{i}"
               for i, c in enumerate(chunks)]
        metas = [{"doc_name": c["doc_name"], "page": c["page"]}
                 for c in chunks]

        collection.add(
            documents=texts,
            embeddings=embeds,
            metadatas=metas,
            ids=ids
        )
        total_chunks += len(chunks)
        print(f"  Ingested {len(chunks)} chunks\n")

    print(f"Done!")
    print(f"Total chunks indexed: {total_chunks}")
    print(f"PDFs skipped: {skipped}")

if __name__ == "__main__":
    build_vectorstore()