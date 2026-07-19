import uuid
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import pytest
import chromadb
from fastapi.testclient import TestClient

from backend.services.chunker import TextChunk
from backend.services.doc_registry import initialize_registry
from backend.main import app

EMBEDDING_DIM = 1536
TEST_PDF_CONTENT = """
This is a test PDF file.
It contains some sample text for testing purposes.
The quick brown fox jumps over the lazy dog.
Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
"""


@pytest.fixture
def mock_openai_client():
    client = MagicMock(name="OpenAIClient")

    embed_response = MagicMock(name="EmbedResponse")
    embed_response.data = [MagicMock(name="EmbedData", embedding=[0.1] * EMBEDDING_DIM)]
    client.embeddings = MagicMock()
    client.embeddings.create = AsyncMock(return_value=embed_response)

    chat_response = MagicMock(name="ChatResponse")
    chat_response.choices = [MagicMock(message=MagicMock(content="Mock answer."))]
    chat_response.usage.prompt_tokens = 100
    chat_response.usage.completion_tokens = 50
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=chat_response)

    return client


@pytest.fixture
def chroma_client():
    return chromadb.EphemeralClient()


@pytest.fixture
def in_memory_chroma(chroma_client):
    name = f"test_collection_{uuid.uuid4().hex}"
    collection = chroma_client.create_collection(name)
    return collection


@pytest.fixture
def sample_chunks() -> list[TextChunk]:
    return [
        TextChunk(
            text="This is the first chunk of test content.",
            source_file="test_doc.pdf",
            chunk_index=0,
            char_start=0,
            char_end=40,
            total_chunks=3,
        ),
        TextChunk(
            text="Here is the second chunk for testing.",
            source_file="test_doc.pdf",
            chunk_index=1,
            char_start=35,
            char_end=72,
            total_chunks=3,
        ),
        TextChunk(
            text="Finally, the third chunk completes the set.",
            source_file="test_doc.pdf",
            chunk_index=2,
            char_start=68,
            char_end=110,
            total_chunks=3,
        ),
    ]


@pytest.fixture
def test_client(mock_openai_client, in_memory_chroma, tmp_path):
    registry_db = tmp_path / "test_registry.db"

    from backend.core.config import settings as global_settings

    asyncio.run(initialize_registry(str(registry_db)))

    with (
        patch("backend.dependencies._openai_client", mock_openai_client),
        patch("backend.dependencies._collection", in_memory_chroma),
        patch.object(global_settings, "registry_db_path", str(registry_db)),
    ):
        client = TestClient(app)
        yield client


@pytest.fixture
def tmp_pdf(tmp_path):
    tmp_pdf_path = tmp_path / "test_doc.pdf"
    text = TEST_PDF_CONTENT.strip()
    text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    stream_data = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    stream_len = len(stream_data)

    header = b"%PDF-1.4\n"
    obj1 = b"1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n"
    obj2 = b"2 0 obj\n<</Type/Pages/Kids[3 0 R]/Count 1>>\nendobj\n"
    obj3 = (
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources <</Font<</F1 5 0 R>>>> >>\n"
        b"endobj\n"
    )
    stream_obj = (
        b"4 0 obj\n"
        b"<< /Length " + str(stream_len).encode() + b" >>\n"
        b"stream\n" + stream_data + b"\nendstream\nendobj\n"
    )
    obj5 = b"5 0 obj\n<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>\nendobj\n"

    parts = [header, obj1, obj2, obj3, stream_obj, obj5]
    body = b"".join(parts)

    num_objects = 6
    pos = len(header)
    xref_entries = [b"0000000000 65535 f \n"]
    for part in parts[1:]:
        xref_entries.append(f"{pos:010d} 00000 n \n".encode())
        pos += len(part)

    xref = b"xref\n" + f"0 {num_objects}\n".encode() + b"".join(xref_entries)
    trailer = (
        f"trailer\n<< /Size {num_objects} /Root 1 0 R >>\n"
        f"startxref\n{len(body)}\n%%EOF".encode()
    )

    tmp_pdf_path.write_bytes(body + xref + trailer)
    yield tmp_pdf_path
