from langchain_text_splitters import RecursiveCharacterTextSplitter
from dataclasses import dataclass
from backend.services.document_loader import LoadedDocument

# Note:
# chunk_size and chunk_overlap are passed in (not read from config directly)
# so chunker.py is independently testable without a settings object


@dataclass
class TextChunk:
    text: str
    source_file: str
    chunk_index: int
    char_start: int
    char_end: int
    total_chunks: int


def estimate_char_offsets(chunks: list[str], full_text: str) -> list[tuple[int, int]]:
    offsets = []
    search_from = 0

    for chunk in chunks:
        char_start = full_text.find(chunk, search_from)
        char_end = char_start + len(chunk)
        offsets.append((char_start, char_end))
        search_from = char_end + 1

    return offsets


def chunk_document(
    doc: LoadedDocument, chunk_size: int, chunk_overlap: int
) -> list[TextChunk]:

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = text_splitter.split_text(doc.text)

    offsets = estimate_char_offsets(chunks, doc.text)
    result = []

    for i, (char_start, char_end) in enumerate(offsets):
        chunk = doc.text[char_start:char_end]
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
