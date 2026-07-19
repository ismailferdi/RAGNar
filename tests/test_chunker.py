from backend.services.chunker import chunk_document
from backend.services.document_loader import LoadedDocument

UNIQUE_TEXT = (
    "Apple Banana Cherry Date. "
    "Elderberry Fig Grape Honeydew. "
    "Iceplant Jackfruit Kiwi Lemon. "
    "Mango Nectarine Orange Papaya. "
    "Quince Raspberry Strawberry Tangerine. "
    "Ugli fruit Vanilla Watermelon Xigua. "
    "Yam Zucchini Apricot Bilberry. "
    "Cranberry Dragonfruit Elderflower. "
    "Feijoa Gooseberry Huckleberry. "
    "Ilama Jabuticaba Kaffir lime Lime. "
    "Mulberry Nance Olive Pineapple. "
    "Cantaloupe Date Endive Fennel. "
    "Ginger Horseradish Jicama Kale. "
    "Leek Mushroom Nopal Okra. "
    "Parsnip Radish Spinach Tomato. "
    "Ube Vegetable Wasabi Ximenia. "
    "Yam Zucchini Artichoke Broccoli. "
    "Cabbage Daikon Eggplant Fiddlehead. "
)


def test_short_text_returns_single_chunk():
    document = LoadedDocument(
        text="This is a test document with a short text.",
        source_file="test_doc.pdf",
        file_type="pdf",
    )
    chunks = chunk_document(document, chunk_size=100, chunk_overlap=20)
    assert len(chunks) == 1


def test_long_text_returns_multiple_chunks():
    document = LoadedDocument(
        text=UNIQUE_TEXT,
        source_file="test_doc.pdf",
        file_type="pdf",
    )
    chunks = chunk_document(document, chunk_size=50, chunk_overlap=10)
    assert len(chunks) > 1


def test_chunk_lengths_within_limit():
    document = LoadedDocument(
        text=UNIQUE_TEXT,
        source_file="test_doc.pdf",
        file_type="pdf",
    )
    chunk_size = 50
    chunk_overlap = 10
    chunks = chunk_document(document, chunk_size, chunk_overlap)
    for chunk in chunks:
        assert chunk.char_end - chunk.char_start <= chunk_size + chunk_overlap


def test_chunk_index_and_total_chunks():
    document = LoadedDocument(
        text=UNIQUE_TEXT,
        source_file="test_doc.pdf",
        file_type="pdf",
    )
    chunks = chunk_document(document, chunk_size=50, chunk_overlap=10)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i
        assert chunk.total_chunks == len(chunks)
    assert chunks[0].chunk_index == 0


def test_adjacent_chunks_share_overlap():
    text = (
        "Machine learning is a method of data analysis that automates analytical model building. "
        "It is a branch of artificial intelligence based on the idea that systems can learn from data. "
        "Deep learning uses neural networks with many layers to analyze complex patterns. "
        "Natural language processing enables computers to understand human language. "
        "Computer vision allows machines to interpret and process visual information. "
        "Reinforcement learning trains agents through rewards and punishments. "
        "Supervised learning uses labeled training data to learn mapping functions. "
        "Unsupervised learning finds hidden patterns in unlabeled data. "
        "Transfer learning applies knowledge from one task to another related task. "
        "Generative models can create new content that resembles training data. "
        "Neural networks are computing systems inspired by biological neural networks. "
        "Convolutional networks excel at processing grid-like data such as images. "
        "Recurrent networks are designed to work with sequence data like text. "
        "Attention mechanisms allow models to focus on relevant parts of input. "
        "Transformers have become the dominant architecture for NLP tasks. "
        "Pre-training and fine-tuning have revolutionized how models are developed. "
        "Large language models can perform a wide variety of language tasks. "
        "Multimodal models process and integrate multiple types of data simultaneously. "
    )

    document = LoadedDocument(
        text=text,
        source_file="test_doc.pdf",
        file_type="pdf",
    )
    chunk_size = 100
    chunk_overlap = 20
    chunks = chunk_document(document, chunk_size, chunk_overlap)
    if len(chunks) > 1:
        has_overlap = any(
            chunks[i].char_end > chunks[i + 1].char_start
            for i in range(len(chunks) - 1)
        )
        assert has_overlap, (
            "Expected at least one pair of adjacent chunks to overlap "
            f"but none found (chunks[0] ends at {chunks[0].char_end}, "
            f"chunks[1] starts at {chunks[1].char_start})"
        )
