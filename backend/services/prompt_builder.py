import tiktoken
from backend.core.config import settings
from backend.services.retriever import SourceChunk

# Note:
# `max_context_tokens` is a safety ceiling, not a target —
# for `gpt-oss-120b` with a 300k context window,
# a budget of 8 000 tokens leaves ample room for the output
# while keeping latency and cost low

SYSTEM_PROMPT = """
You are a helpful assistant.
Answer the user's questions based ONLY on the context provided below.
If the context does not contain enough information to answer the question,
respond with exactly: 'I could not find the answer in the uploaded documents.'
Do not use prior knowledge. Cite the relevant context item by number, e.g. [1], [2].
"""

encoding = tiktoken.encoding_for_model(settings.chat_model)


def format_context(chunks: list[SourceChunk]) -> str:

    return "\n\n".join(
        [
            f"[{i+1}] (source: {chunk.source_file}, chunk {chunk.chunk_index})\n{chunk.text}"
            for i, chunk in enumerate(chunks)
        ]
    )


def count_tokens(text: str, encoding: tiktoken.Encoding) -> int:
    return len(encoding.encode(text=text))


def build_prompt(
    question: str, chunks: list[SourceChunk], max_context_tokens: int
) -> list[dict]:
    if not chunks:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
    while len(chunks) > 1:
        context = format_context(chunks=chunks)

        prompt = SYSTEM_PROMPT + "\n\nContext:\n:" + context
        total_tokens = count_tokens(text=prompt, encoding=encoding) + count_tokens(
            text=question, encoding=encoding
        )

        if total_tokens <= max_context_tokens:
            break

        chunks.pop()

    final_context = format_context(chunks=chunks)

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT + "\n\nContext:\n" + final_context,
        },
        {"role": "user", "content": question},
    ]
