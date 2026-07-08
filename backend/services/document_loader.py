from backend.core.errors import EmptyDocumentError, UnsupportedFileTypeError
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

    pages_text = [
        page.extract_text() for page in reader.pages if page.extract_text() is not None
    ]
    text = "\n".join(pages_text)

    if not text.strip():
        raise EmptyDocumentError(
            f"The PDF file '{file_path}' is empty or contains no extractable text."
        )

    return LoadedDocument(text=text, source_file=source_file, file_type="pdf")


def load_txt(file_path: str, source_file: str) -> LoadedDocument:
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
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".pdf":
        return load_pdf(file_path, source_file)
    elif ext == ".txt":
        return load_txt(file_path, source_file)
    else:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{ext}' for file '{file_path}'. Only PDF and TXT files are supported."
        )
