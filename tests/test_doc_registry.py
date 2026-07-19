import pytest
import aiosqlite
from backend.services.doc_registry import (
    initialize_registry,
    register_document,
    list_documents,
    get_document_by_id,
    delete_document,
)


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_registry.db")


async def test_initialize_registry_creates_table(db_path):
    await initialize_registry(db_path)

    async with aiosqlite.connect(db_path) as conn:
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='documents'"
        )
        table = await cursor.fetchone()
    assert table is not None


async def test_initialize_registry_is_idempotent(db_path):
    await initialize_registry(db_path)
    await initialize_registry(db_path)

    async with aiosqlite.connect(db_path) as conn:
        cursor = await conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='documents'"
        )
        count = await cursor.fetchone()
    assert count[0] == 1


async def test_register_and_list_documents(db_path):
    await initialize_registry(db_path)
    await register_document(
        db_path,
        original_filename="My Report.pdf",
        sanitized_filename="my_report.pdf",
        chunk_count=5,
    )

    docs = await list_documents(db_path)

    assert len(docs) == 1
    assert docs[0]["sanitized_filename"] == "my_report.pdf"
    assert docs[0]["chunk_count"] == 5
    assert docs[0]["original_filename"] == "My Report.pdf"
    assert "ingestion_timestamp" in docs[0]


async def test_duplicate_sanitized_filename_raises(db_path):
    await initialize_registry(db_path)
    await register_document(
        db_path, "doc1.pdf", "same_name.pdf", 3
    )

    with pytest.raises(Exception):
        await register_document(
            db_path, "doc2.pdf", "same_name.pdf", 5
        )


async def test_get_document_by_id_returns_none_for_missing(db_path):
    await initialize_registry(db_path)

    doc = await get_document_by_id(db_path, 999)

    assert doc is None


async def test_delete_document_returns_filename_on_success(db_path):
    await initialize_registry(db_path)
    await register_document(
        db_path, "test.pdf", "test.pdf", 3
    )

    result = await delete_document(db_path, 1)

    assert result == "test.pdf"
    remaining = await list_documents(db_path)
    assert len(remaining) == 0


async def test_delete_document_returns_none_for_missing_id(db_path):
    await initialize_registry(db_path)

    result = await delete_document(db_path, 999)

    assert result is None
