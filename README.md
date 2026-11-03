# RAG Pipeline

Retrieval-Augmented Generation system using Qdrant, Gemini, and FastAPI.

## Stack

| Service    | Purpose                  | Port |
|------------|--------------------------|------|
| Qdrant     | Vector database          | 6333 |
| PostgreSQL | Metadata storage         | 5432 |
| Redis      | Caching                  | 6379 |
| FastAPI    | RAG API                  | 8000 |

Embeddings and generation powered by **Google Gemini** (`text-embedding-004` / `gemini-2.0-flash`).

## Quick Start

```bash
# 1. Configure
cp .env .env.example   # edit .env and add your GEMINI_API_KEY

# 2. Start services
docker compose up -d

# 3. Load test data (BeIR/hotpotqa — 500 Wikipedia passages)
pip install pandas pyarrow huggingface_hub httpx
python scripts/load_dataset.py

# 4. Query
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "Who directed Jaws?", "top_k": 5}'
```

## API

### `POST /query`

Ask a question. Retrieves relevant chunks from Qdrant and generates an answer via Gemini.

```json
{ "query": "your question", "top_k": 5 }
```

### `POST /ingest/text`

Ingest raw text.

```json
{ "text": "document content here", "source": "my-doc" }
```

### `POST /ingest/file`

Upload a file (multipart form).

```bash
curl -F "file=@document.txt" http://localhost:8000/ingest/file
```

### `POST /ingest/bulk`

Batch ingest multiple documents.

```json
{ "documents": [{ "text": "...", "source": "..." }] }
```

### `GET /health`

Returns `{"status": "ok"}`.

## Dataset

The included loader script pulls the [BeIR/hotpotqa](https://huggingface.co/datasets/BeIR/hotpotqa) corpus from HuggingFace:

```bash
python scripts/load_dataset.py --limit 1000 --batch-size 100
```

## Configuration

All settings are in `.env`:

| Variable               | Default                  |
|------------------------|--------------------------|
| `GEMINI_API_KEY`       | (required)               |
| `EMBEDDING_MODEL`      | `models/text-embedding-004` |
| `EMBEDDING_DIMENSIONS` | `768`                    |
| `LLM_MODEL`           | `gemini-2.0-flash`       |
| `CHUNK_SIZE`           | `512`                    |
| `CHUNK_OVERLAP`        | `50`                     |
| `QDRANT_COLLECTION`    | `documents`              |
