import sqlite3
from datetime import datetime


def initialize_registry(db_path: str):

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""CREATE TABLE IF NOT EXISTS documents (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            original_filename TEXT,
                            sanitized_filename TEXT UNIQUE,
                            ingestion_timestamp TEXT,
                            chunk_count INTEGER);
                       """)
        conn.commit()


def register_document(
    db_path: str, original_filename: str, sanitized_filename: str, chunk_count: int
):
    ingestion_timestamp = datetime.now().isoformat()

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO documents (original_filename, sanitized_filename, ingestion_timestamp, chunk_count)
                          VALUES (?, ?, ?, ?);""",
            (original_filename, sanitized_filename, ingestion_timestamp, chunk_count),
        )


def list_documents(db_path: str) -> list[dict]:

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents;")
        conn.commit()
        rows = cursor.fetchall()

        documents = [dict(row) for row in rows]

    return documents


def get_document_by_id(db_path: str, doc_id: int) -> dict | None:

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE id = ?;", (doc_id,))
        row = cursor.fetchone()

        if row:
            document = {
                "id": row[0],
                "original_filename": row[1],
                "sanitized_filename": row[2],
                "ingestion_timestamp": row[3],
                "chunk_count": row[4],
            }
        else:
            document = None

    return document


def delete_document(db_path: str, doc_id: int) -> str | None:

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT sanitized_filename FROM documents WHERE id = ?;", (doc_id,)
        )
        row = cursor.fetchone()

        if row:
            cursor.execute("DELETE FROM documents WHERE id = ?;", (doc_id,))
            conn.commit()
            return row[0]
        else:
            return None
