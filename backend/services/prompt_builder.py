import tiktoken
from backend.core.config import settings
from backend.services.retriever import SourceChunk

# Note:
# `max_context_tokens` is a safety ceiling, not a target —
# for `gpt-oss-120b` with a 300k context window,
# a budget of 8 000 tokens leaves ample room for the output
# while keeping latency and cost low

SYSTEM_PROMPT = """
You are a helpful assistant that answers ONLY from the provided context.
Rules:
1. If the context contains the answer, cite it by number [1], [2], etc. Place citations at the end of the sentence that uses
that source.
2. If the context does NOT contain enough imformation, reply EXACTLY:
    "I could not find the answer in the uploaded documents."
3. Never use your own knowledge. Never guess. If you are unsure, follow rule 2.
4. Do not add meta-commentary like "Based on the context..." - just give the answer.

Example:
Context:
[1] (source: report.pdf, chunk 0)
The capital of France is Paris.

[2] (source: report.pdf, chunk 1)
The population of Paris is 2.1 million.

Question: What is the capital of France?
Correct answer: The capital of France is Paris. [1]

Question: What is the population of Lyon?
Correct answer: I could not find the answer in the uploaded documents.

Now answer the following using the context below."""


try:
    encoding = tiktoken.encoding_for_model(settings.tiktoken_model)
except KeyError:
    encoding = tiktoken.get_encoding(settings.tiktoken_encoder)


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
    chunks = chunks.copy()

    if not chunks:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
    while len(chunks) > 1:
        context = format_context(chunks=chunks)

        prompt = SYSTEM_PROMPT + "\n\nContext:\n" + context
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
