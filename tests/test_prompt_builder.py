from backend.services.prompt_builder import build_prompt, SYSTEM_PROMPT
from backend.services.retriever import SourceChunk


def test_build_prompt_empty_chunks_has_no_context():
    messages = build_prompt("What is RAG?", [], max_context_tokens=8000)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "What is RAG?"
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert "Context:\n\n[1]" not in messages[0]["content"]


def test_build_prompt_with_three_chunks_formats_numbers():
    chunks = [
        SourceChunk(
            text="Paris is the capital of France.",
            source_file="report.pdf",
            chunk_index=0,
            similarity_score=0.9,
        ),
        SourceChunk(
            text="The population of Paris is 2.1 million.",
            source_file="report.pdf",
            chunk_index=1,
            similarity_score=0.8,
        ),
        SourceChunk(
            text="France is in Western Europe.",
            source_file="report.pdf",
            chunk_index=2,
            similarity_score=0.7,
        ),
    ]

    messages = build_prompt("What is the capital?", chunks, max_context_tokens=8000)

    system_content = messages[0]["content"]
    assert "[1]" in system_content
    assert "[2]" in system_content
    assert "[3]" in system_content
    assert "report.pdf" in system_content


def test_build_prompt_drops_lowest_similarity_chunk_when_over_budget():
    chunks = [
        SourceChunk(
            text="Short text A.",
            source_file="a.pdf",
            chunk_index=0,
            similarity_score=0.9,
        ),
        SourceChunk(
            text="Short text B.",
            source_file="a.pdf",
            chunk_index=1,
            similarity_score=0.8,
        ),
        SourceChunk(
            text="Short text C.",
            source_file="a.pdf",
            chunk_index=2,
            similarity_score=0.1,
        ),
    ]

    messages = build_prompt("Hi", chunks, max_context_tokens=1)

    system_content = messages[0]["content"]
    assert "[1]" in system_content
    assert "Short text A." in system_content
    assert "Short text C." not in system_content or "[3]" not in system_content


def test_build_prompt_returns_exactly_two_messages():
    chunks = [
        SourceChunk(
            text="Some context.",
            source_file="doc.pdf",
            chunk_index=0,
            similarity_score=0.9,
        ),
    ]

    messages = build_prompt("Question?", chunks, max_context_tokens=8000)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
