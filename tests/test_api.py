from unittest.mock import AsyncMock, patch
from io import BytesIO

from backend.services.retriever import SourceChunk
from backend.services.llm_client import LLMResponse


def test_health_returns_healthy(test_client):
    response = test_client.get("/health/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "vector_store_connected" in data
    assert "registry_connected" in data


def test_ask_with_sources_returns_grounded(test_client):
    chunks = [
        SourceChunk(
            text="RAG stands for Retrieval Augmented Generation.",
            source_file="doc.pdf",
            chunk_index=0,
            similarity_score=0.95,
        ),
        SourceChunk(
            text="It combines retrieval with text generation.",
            source_file="doc.pdf",
            chunk_index=1,
            similarity_score=0.85,
        ),
    ]

    with (
        patch("backend.routes.ask.retrieve", AsyncMock(return_value=chunks)),
        patch("backend.routes.ask.build_prompt") as mock_build,
        patch("backend.routes.ask.generate_answer", AsyncMock()) as mock_generate,
    ):
        mock_build.return_value = [
            {"role": "system", "content": "Context: ..."},
            {"role": "user", "content": "What is RAG?"},
        ]
        mock_generate.return_value = LLMResponse(
            answer="RAG is Retrieval Augmented Generation.",
            prompt_tokens=50,
            completion_tokens=20,
        )

        response = test_client.post(
            "/ask/", json={"question": "What is RAG?"}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["grounded"] is True
    assert len(data["answer"]) > 0
    assert len(data["sources"]) == 2
    assert data["sources"][0]["source_file"] == "doc.pdf"
    assert data["prompt_tokens"] == 50
    assert data["completion_tokens"] == 20


def test_ask_without_sources_returns_not_grounded(test_client):
    with (
        patch("backend.routes.ask.retrieve", AsyncMock(return_value=[])),
        patch("backend.routes.ask.build_prompt") as mock_build,
        patch("backend.routes.ask.generate_answer", AsyncMock()) as mock_generate,
    ):
        mock_build.return_value = [
            {"role": "system", "content": "No context."},
            {"role": "user", "content": "Unknown topic?"},
        ]
        mock_generate.return_value = LLMResponse(
            answer="I could not find the answer in the uploaded documents.",
            prompt_tokens=30,
            completion_tokens=15,
        )

        response = test_client.post(
            "/ask/", json={"question": "Unknown topic?"}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["grounded"] is False
    assert "could not find" in data["answer"]


def test_ask_with_short_question_returns_422(test_client):
    response = test_client.post("/ask/", json={"question": "ab"})
    assert response.status_code == 422


def test_ingest_unsupported_file_type_returns_422(test_client):
    response = test_client.post(
        "/ingest/",
        files={
            "file": (
                "test.docx",
                BytesIO(b"fake content"),
                "application/octet-stream",
            )
        },
    )
    assert response.status_code == 422


def test_ingest_when_source_exists_and_no_force_returns_409(test_client):
    with patch(
        "backend.routes.ingest.source_exists_async", AsyncMock(return_value=True)
    ):
        response = test_client.post(
            "/ingest/",
            files={"file": ("test.txt", BytesIO(b"Hello world"), "text/plain")},
        )
    assert response.status_code == 409


def test_get_documents_returns_200(test_client):
    with patch(
        "backend.routes.documents.list_documents", AsyncMock(return_value=[])
    ):
        response = test_client.get("/documents/")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert isinstance(data["documents"], list)


def test_delete_document_not_found_returns_404(test_client):
    with patch(
        "backend.routes.documents.get_document_by_id", AsyncMock(return_value=None)
    ):
        response = test_client.delete("/documents/999")
    assert response.status_code == 404
