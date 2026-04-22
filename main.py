from fastapi import FastAPI, UploadFile, File, Form
import os

from utils.ingest import ingest_pdf
from utils.query import answer_with_rag
from utils.helper import ensure_dir

app = FastAPI(title="RAG Backend (CPU local)")


@app.post("/ingest")
async def ingest_endpoint(file: UploadFile = File(...), persist_dir: str = Form("./chroma_db")):
	"""Upload a PDF file and ingest into the vector store (synchronous ingestion)."""
	ensure_dir(persist_dir)
	file_path = os.path.join(persist_dir, file.filename)
	with open(file_path, "wb") as f:
		f.write(await file.read())
	count = ingest_pdf(file_path, persist_dir=persist_dir)
	return {"status": "ingested", "file": file.filename, "chunks": count}


@app.post("/ask")
async def ask_endpoint(query: str = Form(...), k: int = Form(4), persist_dir: str = Form("./chroma_db")):
	"""Ask a question against the ingested corpus."""
	result = answer_with_rag(query, persist_dir=persist_dir, top_k=k)
	return result
