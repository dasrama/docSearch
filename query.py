from embeddings import Embedder
from vectordb import ChromaDB
from llm import LocalLLM


def answer_with_rag(query: str, persist_dir: str = "./chroma_db", top_k: int = 4, model_path: str = None):
    """Embed the query, retrieve top_k documents, call local LLM with context, and return structured answer."""
    embedder = Embedder()
    qvec = embedder.embed_texts([query])[0]

    db = ChromaDB(persist_dir=persist_dir)
    res = db.query(qvec, k=top_k)
    docs = res.get("documents", [])
    metas = res.get("metadatas", [])

    context_parts = []
    sources = []
    for d, m in zip(docs, metas):
        context_parts.append(d)
        src = m.get("source") if isinstance(m, dict) else m
        sources.append(src)

    context = "\n\n---\n\n".join(context_parts)
    prompt = f"Use the following extracted document context to answer the question. If the answer is not contained, say 'I don't know'.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer concisely and cite sources."

    llm = LocalLLM(model_path=model_path)
    answer = llm.generate(prompt)

    return {"answer": answer, "sources": sources}
