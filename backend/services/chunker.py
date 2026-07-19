from langchain_text_splitters import RecursiveCharacterTextSplitter
from dataclasses import dataclass
from backend.services.document_loader import LoadedDocument


@dataclass
class TextChunk:
    text: str
    source_file: str
    chunk_index: int
    char_start: int
    char_end: int
    total_chunks: int


def chunk_document(
    doc: LoadedDocument, chunk_size: int, chunk_overlap: int
) -> list[TextChunk]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    raw_chunks = text_splitter.split_text(doc.text)

    chunks = [c for c in raw_chunks if c.strip()]

    result = []
    search_from = 0
    for i, chunk in enumerate(chunks):
        char_start = doc.text.find(chunk, search_from)
        if char_start == -1:
            char_start = 0
        char_end = char_start + len(chunk)
        search_from = char_end
        result.append(
            TextChunk(
                text=chunk,
                source_file=doc.source_file,
                chunk_index=i,
                char_start=char_start,
                char_end=char_end,
                total_chunks=len(chunks),
            )
        )

    return result
