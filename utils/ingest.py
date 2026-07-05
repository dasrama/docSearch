import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pdfplumber
import redis
import numpy as np
from redisvl.index import SearchIndex
from redisvl.utils.vectorize import HFTextVectorizer

def ingest_pdf(path: str, redis_url: str = "redis://localhost:6379"):
    # 1. Extract and Chunk text
    chunks = []
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=0)
            page_chunks = splitter.split_text(text)
            for chunk in page_chunks:
                chunks.append({
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "page": page_num,
                    "source": path
                })
    
    if not chunks:
        return 0
    
    # 2. Vectorize
    hf = HFTextVectorizer(model="sentence-transformers/all-MiniLM-L6-v2")
    texts = [c["text"] for c in chunks]
    embeddings = hf.embed_many(texts)

    print(type(embeddings))
    print(type(embeddings[0]))
    print(type(embeddings[0][0]))
    # print("embeddings shape", embeddings.shape)
    # print("embeddings", embeddings)

    # 3. Store in Redis
    redis_client = redis.Redis.from_url(redis_url)
    index_name = "pdf_index"
    
    schema = {
        "index": {"name": index_name, "prefix": "chunk", "storage_type": "hash"},
        "fields": [
            {"name": "content", "type": "text"},
            {"name": "text_embedding", "type": "vector", "attrs": {
                "dims": 384, "distance_metric": "cosine", "algorithm": "hnsw", "datatype": "float32"
            }},
            {"name": "source", "type": "tag"},
            {"name": "page", "type": "numeric"}
        ]
    } 
    
    index = SearchIndex.from_dict(schema)
    index.set_client(redis_client)
    index.create(overwrite=True)
    
    data = []
    for chunk, embedding in zip(chunks, embeddings):
        data.append({
            "id": chunk["id"],
            "content": chunk["text"],
            # "text_embedding": embedding,
            "text_embedding": np.array(
                embedding,
                dtype=np.float32
            ).tobytes(),
            "source": "pdf",
            "page": chunk["page"]
        })
    print("data", data)
    print(type(data[0]["text_embedding"]))
    print(len(data[0]["text_embedding"]))
    import redisvl
    print(redisvl.__version__)

    # for k, v in row.items():
    #     print(f"{k}: {type(v)}")

    #     if isinstance(v, list):
    #         print("  first item type:", type(v[0]))

    for i, row in enumerate(data):
        try:
            index.load([row])
            print(f"Loaded {i}")
        except Exception as e:
            print(f"Failed at {i}")
            # print(row)
            for k, v in row.items():
                print(f"{k}: {type(v)}")

                if isinstance(v, list):
                    print("  first item type:", type(v[0]))
            raise
    # index.load(data)
    return len(chunks)