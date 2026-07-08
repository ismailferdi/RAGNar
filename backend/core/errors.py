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


class DocumentNotFoundError(Exception):
    pass


class FileTooLargeError(Exception):
    pass
