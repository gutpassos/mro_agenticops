"""Tests for retriever."""
import pytest
from pathlib import Path
from src.retriever import VectorRetriever
import numpy as np


def test_vector_retriever_init():
    """Test VectorRetriever initialization."""
    index_path = Path("data/processed/faiss.index")
    retriever = VectorRetriever(
        index_path=index_path,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    assert retriever.index_path == index_path
    assert retriever.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert retriever.index is None  # Not loaded yet


@pytest.mark.integration
def test_retriever_load_index():
    """Integration test for loading index (requires built index)."""
    pytest.skip("Requires pre-built FAISS index")
    
    index_path = Path("data/processed/faiss.index")
    retriever = VectorRetriever(index_path=index_path)
    retriever.load_index()
    
    assert retriever.index is not None
    assert retriever.metadata is not None
    assert retriever.encoder is not None


@pytest.mark.integration
def test_retriever_retrieve():
    """Integration test for retrieval (requires built index)."""
    pytest.skip("Requires pre-built FAISS index")
    
    index_path = Path("data/processed/faiss.index")
    retriever = VectorRetriever(index_path=index_path)
    retriever.load_index()
    
    results = retriever.retrieve("test query", top_k=3)
    
    assert isinstance(results, list)
    assert len(results) <= 3
    
    if results:
        assert "text" in results[0]
        assert "score" in results[0]
        assert "metadata" in results[0]
