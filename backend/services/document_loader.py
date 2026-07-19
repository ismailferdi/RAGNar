from backend.core.errors import EmptyDocumentError, UnsupportedFileTypeError
from backend.core.config import MAX_UPLOAD_SIZE_MB
from pypdf import PdfReader
from dataclasses import dataclass
from typing import Literal
import re
import os


@dataclass
class LoadedDocument:
    text: str
    source_file: str
    file_type: Literal["pdf", "txt"]


def sanitize_filename(filename: str) -> str:

    file = filename.lower().strip()
    file = os.path.basename(file)
    name, ext = os.path.splitext(file)
    name = re.sub(r"[^_\w]+", "_", name)
    name = name.strip("_")

    return name + ext


def load_pdf(file_path: str, source_file: str) -> LoadedDocument:

    reader = PdfReader(file_path)

    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)

    text = "\n".join(pages_text)

    if not text.strip():
        raise EmptyDocumentError(
            f"The PDF file '{file_path}' is empty or contains no extractable text."
        )

    return LoadedDocument(text=text, source_file=source_file, file_type="pdf")


def load_txt(file_path: str, source_file: str) -> LoadedDocument:

    max_file_size = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if os.path.getsize(file_path) > max_file_size:
        raise ValueError(
            f"The file '{file_path}' exceeds the maximum allowed size of {MAX_UPLOAD_SIZE_MB} MB."
        )

    try:
        with open(
            file_path,
            "r",
            encoding="utf-8",
        ) as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            text = f.read()

    return LoadedDocument(text=text, source_file=source_file, file_type="txt")


def load_document(file_path: str, original_filename: str) -> LoadedDocument:

    source_file = sanitize_filename(original_filename)
    _, ext = os.path.splitext(original_filename)
    ext = ext.lower()

    if ext == ".pdf":
        return load_pdf(file_path, source_file)
    elif ext == ".txt":
        return load_txt(file_path, source_file)
    else:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{ext}' for file '{original_filename}'. Only PDF and TXT files are supported."
        )
