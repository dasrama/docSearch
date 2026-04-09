import uuid
import pdfplumber

from embeddings import Embedder
from vectordb import ChromaDB
from utils import chunk_text


import chromadb
client = chromadb.CloudClient(
    api_key='ck-98kJau5obWdcFuGCvXLGjUxJx9XWxYBNbT9JcfKxp9xL',
    tenant='741f601e-4753-4299-89e6-b2cfe868832b',
    database='rag_pipeline'
)

def ingest_pdf(path: str, persist_dir: str = "./chroma_db", collection_name: str = "default") -> int:
    """Extract text from `path`, chunk it, embed and upsert to Chroma. Returns number of chunks."""
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((i, text))

    chunks = []
    for page_num, text in pages:
        for chunk in chunk_text(text):
            chunks.append({"id": str(uuid.uuid4()), "text": chunk, "page": page_num, "source": path})

    if not chunks:
        return 0

    embedder = Embedder()
    texts = [c["text"] for c in chunks]
    vectors = embedder.embed_texts(texts)

    db = ChromaDB(persist_dir=persist_dir, collection_name=collection_name)
    docs = []
    for c, v in zip(chunks, vectors):
        docs.append({
            "id": c["id"],
            "document": c["text"],
            "metadata": {"page": c["page"], "source": c["source"]},
            "embedding": v,
        })

    db.upsert(docs)
    return len(docs)
