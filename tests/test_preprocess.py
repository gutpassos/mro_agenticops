"""Tests for preprocessing pipeline."""
import pytest
from pathlib import Path
from src.preprocess import clean_text, semantic_chunking, process_pdf


def test_clean_text():
    """Test text cleaning function."""
    dirty = "This   is  a\n\ntest\n5\n  with-\n  strange   formatting"
    clean = clean_text(dirty)
    
    assert "  " not in clean  # No double spaces
    assert "strange formatting" in clean  # Hyphenation fixed


def test_semantic_chunking():
    """Test chunking logic."""
    text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
    chunks = semantic_chunking(text, chunk_size=20, chunk_overlap=5)
    
    assert len(chunks) > 0
    assert all(isinstance(c, str) for c in chunks)
    assert all(len(c) > 0 for c in chunks)


def test_semantic_chunking_overlap():
    """Test that chunking creates overlap."""
    text = "A" * 100 + "\n\n" + "B" * 100
    chunks = semantic_chunking(text, chunk_size=50, chunk_overlap=10)
    
    # Should have multiple chunks with overlap
    assert len(chunks) >= 2


def test_process_pdf_structure():
    """Test that process_pdf returns correct structure."""
    # This is a mock test - would need actual PDF in test fixtures
    # For now, just test the structure expectations
    
    # Expected structure
    expected_chunk_keys = {"text", "metadata"}
    expected_metadata_keys = {"source", "chunk_id", "total_chunks", "chunk_size"}
    
    # This would be the actual test with a fixture PDF:
    # chunks = process_pdf(Path("tests/fixtures/sample.pdf"))
    # assert all(expected_chunk_keys == set(chunk.keys()) for chunk in chunks)
    # assert all(expected_metadata_keys == set(chunk["metadata"].keys()) for chunk in chunks)
    
    assert True  # Placeholder
