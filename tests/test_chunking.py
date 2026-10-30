from complyai.rag import RAGEngine

def test_split_text_returns_chunks():
    text = " ".join(["security"] * 500)
    chunks = RAGEngine._split_text(text, chunk_size=200, overlap=30)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)

def test_split_empty_text():
    assert RAGEngine._split_text("   ") == []
