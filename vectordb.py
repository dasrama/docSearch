try:
    import chromadb
    from chromadb.config import Settings
except Exception:
    chromadb = None


class ChromaDB:
    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "default"):
        if chromadb is None:
            raise ImportError("chromadb is required for ChromaDB wrapper")
        self.collection_name = collection_name

        # Try multiple ways to instantiate the client to be compatible with
        # different chromadb versions / migration states.
        settings = None
        try:
            settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir)
        except Exception:
            settings = None

        client = None
        # Try keyword-arg style
        try:
            if settings is not None:
                client = chromadb.Client(settings=settings)
        except Exception:
            client = None

        # Try positional (legacy) style
        if client is None:
            try:
                if settings is not None:
                    client = chromadb.Client(settings)
            except Exception:
                client = None

        # Fallback to default client (no settings); this will use env defaults
        if client is None:
            try:
                client = chromadb.Client()
            except Exception as e:
                raise RuntimeError(f"Failed to create chromadb client: {e}")

        self.client = client

        # Get or create collection
        try:
            # Newer versions provide get_collection/create_collection
            try:
                self.col = self.client.get_collection(name=collection_name)
            except Exception:
                self.col = self.client.create_collection(name=collection_name)
        except Exception:
            # As a last resort, try collection via attribute access
            try:
                self.col = getattr(self.client, "get_collection")(collection_name)
            except Exception as e:
                raise RuntimeError(f"Failed to access or create collection: {e}")

    def upsert(self, docs: list):
        """Docs: list of {id, document, metadata, embedding}"""
        ids = [d["id"] for d in docs]
        documents = [d.get("document", "") for d in docs]
        metadatas = [d.get("metadata", {}) for d in docs]
        embeddings = [d.get("embedding") for d in docs]
        # chroma add will upsert duplicates
        # Some chroma versions use `add`, others `upsert` — try both.
        try:
            self.col.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        except Exception:
            try:
                self.col.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
            except Exception as e:
                raise RuntimeError(f"Failed to add/upsert documents to Chroma: {e}")

    def query(self, query_vector, k: int = 4):
        # Support different query signatures across chroma versions
        try:
            res = self.col.query(query_embeddings=[query_vector], n_results=k, include=["documents", "metadatas", "distances"])
            # result lists are nested per query; return first
            if res and isinstance(res, dict) and "documents" in res and len(res["documents"])>0:
                return {"documents": res["documents"][0], "metadatas": res["metadatas"][0], "distances": res["distances"][0]}
            # Some versions return lists directly
            if isinstance(res, dict):
                return {"documents": res.get("documents", []), "metadatas": res.get("metadatas", []), "distances": res.get("distances", [])}
        except Exception:
            pass

        # Fallback: try a different call signature
        try:
            res = self.col.query(query_vector=query_vector, top_k=k)
            # Attempt to normalize response
            docs = res.get("documents") or []
            metas = res.get("metadatas") or []
            dists = res.get("distances") or []
            return {"documents": docs, "metadatas": metas, "distances": dists}
        except Exception as e:
            raise RuntimeError(f"Chroma query failed: {e}")

