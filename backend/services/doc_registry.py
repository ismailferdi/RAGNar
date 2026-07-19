import aiosqlite
from datetime import datetime
from contextlib import asynccontextmanager


@asynccontextmanager
async def _get_connection(db_path: str):
    conn = await aiosqlite.connect(db_path, timeout=10)
    await conn.execute("PRAGMA journal_mode=WAL;")
    await conn.execute("PRAGMA synchronous=NORMAL;")
    try:
        yield conn
    finally:
        await conn.close()


async def initialize_registry(db_path: str):
    async with _get_connection(db_path) as conn:
        await conn.execute("""CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_filename TEXT,
                sanitized_filename TEXT UNIQUE,
                ingestion_timestamp TEXT,
                chunk_count INTEGER
            );""")
        await conn.commit()


async def register_document(
    db_path: str, original_filename: str, sanitized_filename: str, chunk_count: int
):
    ingestion_timestamp = datetime.now().isoformat()
    async with _get_connection(db_path) as conn:
        await conn.execute(
            """INSERT INTO documents (original_filename, sanitized_filename, ingestion_timestamp, chunk_count)
               VALUES (?, ?, ?, ?);""",
            (original_filename, sanitized_filename, ingestion_timestamp, chunk_count),
        )
        await conn.commit()


async def list_documents(db_path: str) -> list[dict]:
    async with _get_connection(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        cursor = await conn.execute("SELECT * FROM documents;")
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def get_document_by_id(db_path: str, doc_id: int) -> dict | None:
    async with _get_connection(db_path) as conn:
        cursor = await conn.execute("SELECT * FROM documents WHERE id = ?;", (doc_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "original_filename": row[1],
            "sanitized_filename": row[2],
            "ingestion_timestamp": row[3],
            "chunk_count": row[4],
        }


async def get_document_id_by_sanitized_filename(
    db_path: str, sanitized_filename: str
) -> int | None:
    async with _get_connection(db_path) as conn:
        cursor = await conn.execute(
            "SELECT id FROM documents WHERE sanitized_filename = ?;",
            (sanitized_filename,),
        )
        row = await cursor.fetchone()
        return row[0] if row else None


async def delete_document(db_path: str, doc_id: int) -> str | None:
    async with _get_connection(db_path) as conn:
        cursor = await conn.execute(
            "SELECT sanitized_filename FROM documents WHERE id = ?;", (doc_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        await conn.execute("DELETE FROM documents WHERE id = ?;", (doc_id,))
        await conn.commit()
        return row[0]
