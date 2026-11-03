import hashlib
import uuid

from app.chunker import chunk_text
from app.embeddings import get_embeddings
from app.vectorstore import upsert_vectors


def _make_id(text: str, source: str) -> str:
    digest = hashlib.md5(f"{source}:{text}".encode()).hexdigest()
    return str(uuid.UUID(digest))


async def ingest_text(text: str, source: str = "upload") -> int:
    chunks = chunk_text(text, metadata={"source": source})
    if not chunks:
        return 0

    texts = [c["text"] for c in chunks]
    embeddings = await get_embeddings(texts)

    ids = [_make_id(c["text"], source) for c in chunks]
    payloads = [{"text": c["text"], **c["metadata"]} for c in chunks]

    upsert_vectors(ids, embeddings, payloads)
    return len(chunks)


async def ingest_documents(documents: list[dict[str, str]]) -> int:
    """Ingest a list of {"text": ..., "source": ...} dicts."""
    total = 0
    batch_size = 50
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        all_chunks = []
        for doc in batch:
            all_chunks.extend(
                chunk_text(doc["text"], metadata={"source": doc.get("source", "bulk")})
            )
        if not all_chunks:
            continue
        texts = [c["text"] for c in all_chunks]
        embeddings = await get_embeddings(texts)
        ids = [_make_id(c["text"], c["metadata"].get("source", "")) for c in all_chunks]
        payloads = [{"text": c["text"], **c["metadata"]} for c in all_chunks]
        upsert_vectors(ids, embeddings, payloads)
        total += len(all_chunks)
    return total
