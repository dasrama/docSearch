import requests
import redis
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.utils.vectorize import HFTextVectorizer
from config.settings import settings

OLLAMA_GEN_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "gemma3:270m"

def answer_with_rag(query: str, redis_url: str = "redis://localhost:6379", top_k: int = 4):
    # 1. Embed query using same model as ingestion
    hf = HFTextVectorizer(model="sentence-transformers/all-MiniLM-L6-v2")
    qvec = hf.embed(query)

    # 2. Query Redis
    redis_client = redis.Redis.from_url(redis_url)
    index = SearchIndex.from_existing("pdf_index", redis_client=redis_client)
    
    vquery = VectorQuery(
        vector=qvec,
        vector_field_name="text_embedding",
        return_fields=["content", "source", "page"],
        num_results=top_k
    )
    results = index.query(vquery)

    # 3. Build context
    context_parts = [r["content"] for r in results]
    print("context_parts", context_parts)
    sources = [r.get("source") for r in results]
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
    gen_resp.raise_for_status()
    gen_data = gen_resp.json()
    answer = gen_data.get("response", "")

    return {
        "answer": answer,
        "sources": sources
    }
