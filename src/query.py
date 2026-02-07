"""RAG pipeline orchestration - end-to-end query processing."""
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

from .retriever import VectorRetriever
from .llm import OllamaClient


class RAGPipeline:
    """End-to-end RAG pipeline: retrieve + augment + generate."""
    
    def __init__(self, config_path: str = "configs/default.yaml"):
        """
        Initialize RAG pipeline from config.
        
        Args:
            config_path: Path to YAML configuration
        """
        self.config = self._load_config(config_path)
        self.retriever = self._init_retriever()
        self.llm = self._init_llm()
        self.system_prompt = self._load_prompt("prompts/system_prompt.txt")
        
    def _load_config(self, path: str) -> Dict:
        """Load YAML config."""
        with open(path) as f:
            return yaml.safe_load(f)
    
    def _load_prompt(self, path: str) -> Optional[str]:
        """Load prompt template from file."""
        prompt_path = Path(path)
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return None
    
    def _init_retriever(self) -> VectorRetriever:
        """Initialize retriever from config."""
        cfg = self.config.get("retriever", {})
        index_path = Path(self.config["paths"]["processed_data"]) / "faiss.index"
        return VectorRetriever(
            index_path=index_path,
            embedding_model=self.config["embeddings"]["model"]
        )
    
    def _init_llm(self) -> OllamaClient:
        """Initialize LLM client from config."""
        cfg = self.config["ollama"]
        return OllamaClient(
            base_url=cfg["base_url"],
            model=cfg["model"],
            temperature=cfg.get("temperature", 0.1),
            max_tokens=cfg.get("max_tokens", 2048)
        )
    
    def query(
        self,
        question: str,
        top_k: int = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Execute RAG query: retrieve context + generate answer.
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve (overrides config)
            stream: Stream LLM response
            
        Returns:
            Dict with 'answer', 'sources', 'metadata'
        """
        # 1. Retrieve relevant context
        top_k = top_k or self.config["retriever"]["top_k"]
        retrieved = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            similarity_threshold=self.config["retriever"]["similarity_threshold"]
        )
        
        # 2. Build augmented prompt
        context = self._format_context(retrieved)
        prompt = self._build_prompt(question, context)
        
        # 3. Generate answer
        answer = self.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            stream=stream
        )
        
        # 4. Return structured response
        return {
            "answer": answer,
            "sources": [
                {
                    "text": chunk["text"],
                    "score": chunk["score"],
                    "metadata": chunk.get("metadata", {})
                }
                for chunk in retrieved
            ],
            "metadata": {
                "model": self.llm.model,
                "num_sources": len(retrieved),
                "question": question
            }
        }
    
    def _format_context(self, retrieved: list) -> str:
        """Format retrieved chunks into context string."""
        if not retrieved:
            return "Nenhum contexto relevante encontrado."
        
        context_parts = []
        for i, chunk in enumerate(retrieved, 1):
            context_parts.append(f"[Contexto {i}]\n{chunk['text']}\n")
        
        return "\n".join(context_parts)
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Build final prompt with context and question."""
        # Load template if exists
        template_path = Path("prompts/rag_template.txt")
        if template_path.exists():
            template = template_path.read_text(encoding="utf-8")
            return template.format(context=context, question=question)
        
        # Default template
        return f"""Com base no contexto abaixo, responda a pergunta de forma precisa e concisa.

{context}

Pergunta: {question}

Resposta:"""
    
    def stream_query(self, question: str, **kwargs):
        """Stream RAG response chunks."""
        result = self.query(question, stream=True, **kwargs)
        for chunk in result["answer"]:
            yield chunk


def main():
    """CLI para testar RAG pipeline."""
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python -m src.query 'Sua pergunta aqui'")
        sys.exit(1)
    
    question = sys.argv[1]
    rag = RAGPipeline()
    
    print(f"Pergunta: {question}\n")
    result = rag.query(question)
    
    print(f"Resposta: {result['answer']}\n")
    print(f"Fontes ({len(result['sources'])}):")
    for i, source in enumerate(result['sources'], 1):
        print(f"  {i}. Score: {source['score']:.3f}")


if __name__ == "__main__":
    main()
