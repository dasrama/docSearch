import chromadb
from config.settings import settings


class ChromaDB:
    def __init__(self, persist_dir: str | None = None, collection_name: str | None = None):
        """Initialize a persistent Chroma client using provided values or settings.

        Args:
            persist_dir: optional path for Chroma persistence (overrides settings.persist_dir)
            collection_name: optional collection name (overrides settings.collection_name)
        """
        persist_dir = persist_dir or settings.persist_dir
        collection_name = collection_name or settings.collection_name

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def upsert(self, docs: list):
        self.collection.upsert(
            ids=[(d["id"]) for d in docs[0]],
            documents=[d.get("document", "") for d in docs[0]],
            metadatas=[d.get("metadata", {}) for d in docs[0]],
            embeddings=[d.get("embedding") for d in docs[0]],
        )

    def query(self, query_vector):
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=settings.top_k
        )
        return {
            "documents": results.get("documents", [[]])[0],
            "metadatas": results.get("metadatas", [[]])[0],
            "distances": results.get("distances", [[]])[0],
        }