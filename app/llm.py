import google.generativeai as genai

from app.config import settings

genai.configure(api_key=settings.gemini_api_key)

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on the provided context. "
    "If the context doesn't contain enough information to answer, say so clearly. "
    "Always cite which parts of the context you used."
)


async def generate_answer(query: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer based on the context above:"
    )
    model = genai.GenerativeModel(
        model_name=settings.llm_model,
        system_instruction=SYSTEM_PROMPT,
    )
    response = model.generate_content(prompt)
    return response.text
