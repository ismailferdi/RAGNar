from fastapi import HTTPException, status


class EmptyDocumentError(Exception):
    pass


class UnsupportedFileTypeError(Exception):
    pass


class EmbeddingCallError(Exception):
    pass


class EmbeddingModelMismatchError(Exception):
    pass


class ChunkConfigMismatchError(Exception):
    pass


class LLMCallError(Exception):
    pass


class DocumentAlreadyIngestedError(Exception):
    pass


class DocumentNotFoundError(HTTPException):
    def __init__(self, status_code: int, detail: str):
        super().__init__(
            status_code=status_code,
            detail=detail,
        )


class FileTooLargeError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=413, detail=detail)
