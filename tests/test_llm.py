"""Tests for LLM client."""
import pytest
from src.llm import OllamaClient


def test_ollama_client_init():
    """Test OllamaClient initialization."""
    client = OllamaClient(
        base_url="http://localhost:11434",
        model="llama3",
        temperature=0.5
    )
    
    assert client.base_url == "http://localhost:11434"
    assert client.model == "llama3"
    assert client.temperature == 0.5


def test_ollama_is_available():
    """Test server availability check."""
    client = OllamaClient()
    
    # This will fail if Ollama is not running, but shouldn't crash
    result = client.is_available()
    assert isinstance(result, bool)


@pytest.mark.integration
def test_ollama_generate():
    """Integration test for generation (requires Ollama server)."""
    pytest.skip("Requires running Ollama server")
    
    client = OllamaClient(model="llama3")
    response = client.generate("Say 'test' and nothing else.")
    
    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.integration
def test_ollama_chat():
    """Integration test for chat (requires Ollama server)."""
    pytest.skip("Requires running Ollama server")
    
    client = OllamaClient(model="llama3")
    messages = [
        {"role": "user", "content": "Hello"}
    ]
    response = client.chat(messages)
    
    assert isinstance(response, str)
    assert len(response) > 0
