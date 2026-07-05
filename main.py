from fastapi import FastAPI, UploadFile, File, Form
import os

from utils.ingest import ingest_pdf
from utils.query import answer_with_rag
from config.settings import settings

app = FastAPI(title="DocSearch API")


@app.post("/ingest")
async def ingest_endpoint(file: UploadFile = File(...)):
	"""Upload a PDF file and ingest into the vector store (synchronous ingestion)."""
	with open(file.filename, "wb") as f:
		f.write(await file.read())
	count = ingest_pdf(file.filename, redis_url=settings.REDIS_URL)
	return {"status": "ingested", "file": file.filename, "chunks": count}


@app.post("/ask")
async def ask_endpoint(query: str = Form(...), k: int = Form(4)):
	"""Ask a question against the ingested corpus."""
	result = answer_with_rag(query, redis_url=settings.REDIS_URL, top_k=k)
	return result
