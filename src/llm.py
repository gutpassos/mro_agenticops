"""LLM client for Ollama local inference."""
from typing import Optional, Iterator, Dict, Any
import requests
import json


class OllamaClient:
    """Client for Ollama API (local LLM inference)."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3",
        temperature: float = 0.1,
        max_tokens: int = 2048
    ):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Ollama server URL
            model: Model name (e.g., llama3, mistral)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        stream: bool = False
    ) -> str | Iterator[str]:
        """
        Generate completion from prompt.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system context
            stream: Whether to stream response
            
        Returns:
            Generated text or iterator of chunks
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        url = f"{self.base_url}/api/generate"
        
        if stream:
            return self._stream_generate(url, payload)
        else:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()["response"]
    
    def _stream_generate(self, url: str, payload: Dict) -> Iterator[str]:
        """Stream generation chunks."""
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "response" in chunk:
                        yield chunk["response"]
    
    def chat(
        self,
        messages: list[Dict[str, str]],
        stream: bool = False
    ) -> str | Iterator[str]:
        """
        Chat completion with conversation history.
        
        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
            stream: Whether to stream response
            
        Returns:
            Generated response
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        url = f"{self.base_url}/api/chat"
        
        if stream:
            return self._stream_chat(url, payload)
        else:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()["message"]["content"]
    
    def _stream_chat(self, url: str, payload: Dict) -> Iterator[str]:
        """Stream chat chunks."""
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "message" in chunk and "content" in chunk["message"]:
                        yield chunk["message"]["content"]
    
    def list_models(self) -> list[str]:
        """List available models in Ollama."""
        url = f"{self.base_url}/api/tags"
        response = requests.get(url)
        response.raise_for_status()
        return [model["name"] for model in response.json().get("models", [])]
    
    def is_available(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            requests.get(self.base_url, timeout=5)
            return True
        except:
            return False
