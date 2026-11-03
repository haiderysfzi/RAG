from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from app.embeddings import get_embedding
from app.ingest import ingest_text, ingest_documents
from app.llm import generate_answer
from app.vectorstore import ensure_collection, search_vectors


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_collection()
    yield


app = FastAPI(title="RAG API", version="0.1.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


class IngestResponse(BaseModel):
    chunks_ingested: int


class BulkIngestRequest(BaseModel):
    documents: list[dict]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    query_vector = await get_embedding(request.query)
    results = search_vectors(query_vector, top_k=request.top_k)

    context_chunks = [hit.payload["text"] for hit in results]
    sources = [
        {
            "text": hit.payload["text"],
            "source": hit.payload.get("source", "unknown"),
            "score": hit.score,
        }
        for hit in results
    ]

    answer = await generate_answer(request.query, context_chunks)
    return QueryResponse(answer=answer, sources=sources)


@app.post("/ingest/text", response_model=IngestResponse)
async def ingest_raw_text(body: dict):
    text = body.get("text", "")
    source = body.get("source", "api")
    count = await ingest_text(text, source=source)
    return IngestResponse(chunks_ingested=count)


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    count = await ingest_text(text, source=file.filename or "file")
    return IngestResponse(chunks_ingested=count)


@app.post("/ingest/bulk", response_model=IngestResponse)
async def ingest_bulk(request: BulkIngestRequest):
    count = await ingest_documents(request.documents)
    return IngestResponse(chunks_ingested=count)
