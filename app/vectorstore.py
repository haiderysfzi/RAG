from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import settings

qdrant = QdrantClient(url=settings.qdrant_url)


def ensure_collection():
    try:
        qdrant.get_collection(settings.qdrant_collection)
    except (UnexpectedResponse, Exception):
        qdrant.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=models.VectorParams(
                size=settings.embedding_dimensions,
                distance=models.Distance.COSINE,
            ),
        )


def upsert_vectors(
    ids: list[str],
    vectors: list[list[float]],
    payloads: list[dict],
):
    points = [
        models.PointStruct(id=uid, vector=vector, payload=payload)
        for uid, vector, payload in zip(ids, vectors, payloads)
    ]
    qdrant.upsert(collection_name=settings.qdrant_collection, points=points)


def search_vectors(query_vector: list[float], top_k: int = 5) -> list[models.ScoredPoint]:
    return qdrant.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points
