from backend.services.document_loader import (
    sanitize_filename,
    LoadedDocument,
    load_txt,
    load_pdf,
    load_document,
)
from backend.core.errors import UnsupportedFileTypeError


def test_sanitize_filename_lowercase_and_special_chars():
    result = sanitize_filename("My Report (Final).PDF")
    assert result == "my_report_final.pdf"
    assert result.islower()


def test_sanitize_filename_strips_path_components():
    result = sanitize_filename("../../etc/passwd")
    assert result == "passwd"
    assert "/" not in result


def test_load_txt_valid_utf8(tmp_path):
    test_file = tmp_path / "test_document.txt"
    test_content = "This is a test document.\nIt contains multiple lines.\nFor testing purposes."
    test_file.write_text(test_content, encoding="utf-8")

    result = load_txt(str(test_file), "test_document.txt")

    assert isinstance(result, LoadedDocument)
    assert result.text == test_content
    assert result.source_file == "test_document.txt"


def test_load_document_unsupported_file_type():
    try:
        load_document("/fake/path/doc.docx", "document.docx")
    except UnsupportedFileTypeError:
        pass
    else:
        raise AssertionError("Expected UnsupportedFileTypeError")


def test_load_pdf_with_tmp_pdf(tmp_pdf):
    result = load_pdf(str(tmp_pdf), "test_doc.pdf")

    assert isinstance(result, LoadedDocument)
    assert result.text is not None
    assert len(result.text.strip()) > 0
    assert result.source_file == "test_doc.pdf"
