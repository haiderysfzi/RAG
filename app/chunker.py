from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_text(text: str, metadata: dict | None = None) -> list[dict]:
    chunks = splitter.split_text(text)
    return [
        {
            "text": chunk,
            "metadata": {**(metadata or {}), "chunk_index": i},
        }
        for i, chunk in enumerate(chunks)
    ]
