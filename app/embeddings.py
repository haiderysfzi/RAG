import google.generativeai as genai

from app.config import settings

genai.configure(api_key=settings.gemini_api_key)


async def get_embeddings(texts: list[str]) -> list[list[float]]:
    results = []
    for text in texts:
        response = genai.embed_content(
            model=settings.embedding_model,
            content=text,
        )
        results.append(response["embedding"])
    return results


async def get_embedding(text: str) -> list[float]:
    response = genai.embed_content(
        model=settings.embedding_model,
        content=text,
    )
    return response["embedding"]
