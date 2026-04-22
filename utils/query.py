import requests
import chromadb

from config.settings import settings


OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
OLLAMA_GEN_URL = "http://localhost:11434/api/generate"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "gemma3:270m"


def answer_with_rag(query: str, persist_dir: str = settings.persist_dir, top_k: int = 4):
    
    # 1. Embed query (DIRECT)
    embed_resp = requests.post(
        OLLAMA_EMBED_URL,
        json={
            "model": EMBED_MODEL,
            "input": [query]
        }
    )
    embed_resp.raise_for_status()
    embed_data = embed_resp.json()

    if "embeddings" not in embed_data:
        raise ValueError(f"Invalid embedding response: {embed_data}")

    qvec = embed_data["embeddings"][0]

    # 2. Query ChromaDB (DIRECT)
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name=settings.collection_name)

    results = collection.query(
        query_embeddings=[qvec],
        n_results=top_k
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    # 3. Build context
    context_parts = []
    sources = []

    for d, m in zip(docs, metas):
        context_parts.append(d)
        src = m.get("source") if isinstance(m, dict) else m
        sources.append(src)

    context = "\n\n---\n\n".join(context_parts)

    # 4. Build prompt
    prompt = f"""
Use the following context to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {query}

Answer concisely and cite sources.
"""

    # 5. Call LLM (DIRECT)
    gen_resp = requests.post(
        OLLAMA_GEN_URL,
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )
    print(gen_resp)
    gen_resp.raise_for_status()
    gen_data = gen_resp.json()
    print(gen_data)

    answer = gen_data.get("response", "")

    return {
        "answer": answer,
        "sources": sources
    }