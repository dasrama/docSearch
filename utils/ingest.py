import uuid
import requests
import pdfplumber
import chromadb

from utils.helper import chunk_text


OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL_NAME = "nomic-embed-text"


def ingest_pdf(path: str, persist_dir: str = "./chroma_db", collection_name: str = "default") -> int:
    
    # 1. Extract text
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((i, text))

    # 2. Chunk text
    chunks = []
    for page_num, text in pages:
        for chunk in chunk_text(text):
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk,
                "page": page_num,
                "source": path
            })

    if not chunks:
        return 0

    # 3. Prepare texts
    texts = [c["text"] for c in chunks]

    # 4. Call Ollama directly (NO WRAPPER)
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "input": texts
        }
    )
    response.raise_for_status()
    data = response.json()

    # 5. Extract embeddings (STRICT)
    if "embeddings" not in data:
        raise ValueError(f"Invalid embedding response: {data}")

    vectors = data["embeddings"]

    print("Chunks:", len(texts), "Embeddings:", len(vectors))

    # 6. Safety check
    if len(texts) != len(vectors):
        raise ValueError(f"Mismatch: {len(texts)} texts vs {len(vectors)} embeddings")

    # 7. Store in ChromaDB (DIRECT)
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name=collection_name)

    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=texts,
        metadatas=[{"page": c["page"], "source": c["source"]} for c in chunks],
        embeddings=vectors,
    )

    return len(chunks)