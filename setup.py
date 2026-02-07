#!/usr/bin/env python
"""
MRO AgenticOps - Setup and Quick Start Script

This script helps you get started with the MRO RAG system.
"""
import sys
from pathlib import Path
import subprocess
import argparse


def check_ollama():
    """Check if Ollama server is running."""
    try:
        import requests
        response = requests.get("http://localhost:11434", timeout=5)
        return response.status_code == 200
    except:
        return False


def check_dependencies():
    """Check if required packages are installed."""
    required = [
        "pandas",
        "numpy",
        "sentence_transformers",
        "faiss",
        "pypdf",
        "pydantic",
        "yaml"
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    return missing


def install_dependencies():
    """Install dependencies from requirements.txt."""
    print("\n📦 Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✓ Dependencies installed!")


def run_pipeline():
    """Run the full ingestion pipeline."""
    print("\n🔄 Running ingestion pipeline...")
    subprocess.run([sys.executable, "pipelines/ingest_pipeline.py"])


def interactive_query():
    """Start interactive query mode."""
    try:
        from src.query import RAGPipeline
        
        print("\n💬 Starting interactive query mode...")
        print("Type 'exit' or 'quit' to stop.\n")
        
        rag = RAGPipeline()
        
        while True:
            try:
                question = input("❓ Pergunta: ").strip()
                
                if question.lower() in ['exit', 'quit', 'sair']:
                    print("Até logo!")
                    break
                
                if not question:
                    continue
                
                print("\n🤔 Processando...\n")
                result = rag.query(question)
                
                print(f"💡 Resposta:\n{result['answer']}\n")
                
                if result['sources']:
                    print(f"📚 Fontes ({len(result['sources'])}):")
                    for i, source in enumerate(result['sources'][:3], 1):
                        print(f"  {i}. {source['metadata']['source']} (score: {source['score']:.3f})")
                    print()
                
            except KeyboardInterrupt:
                print("\n\nAté logo!")
                break
            except Exception as e:
                print(f"❌ Erro: {e}\n")
    
    except ImportError as e:
        print(f"❌ Erro ao importar módulos: {e}")
        print("Execute primeiro: python setup.py --setup")


def main():
    parser = argparse.ArgumentParser(description="MRO AgenticOps Setup")
    parser.add_argument("--setup", action="store_true", help="Install dependencies and run initial setup")
    parser.add_argument("--pipeline", action="store_true", help="Run ingestion pipeline")
    parser.add_argument("--query", action="store_true", help="Start interactive query mode")
    parser.add_argument("--check", action="store_true", help="Check system status")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  MRO AgenticOps - LLMOps + RAG Local")
    print("=" * 60)
    
    # Check mode
    if args.check or not any([args.setup, args.pipeline, args.query]):
        print("\n🔍 System Status Check\n")
        
        # Check Ollama
        ollama_status = check_ollama()
        print(f"  Ollama Server: {'✓ Running' if ollama_status else '✗ Not running'}")
        
        # Check dependencies
        missing = check_dependencies()
        if missing:
            print(f"  Dependencies: ✗ Missing: {', '.join(missing)}")
        else:
            print("  Dependencies: ✓ All installed")
        
        # Check data
        raw_pdfs = list(Path("data/raw").glob("*.pdf")) if Path("data/raw").exists() else []
        print(f"  Raw PDFs: {len(raw_pdfs)} found")
        
        index_exists = Path("data/processed/faiss.index").exists()
        print(f"  FAISS Index: {'✓ Built' if index_exists else '✗ Not built'}")
        
        print("\n📝 Next Steps:")
        if not ollama_status:
            print("  1. Start Ollama: ollama serve")
        if missing:
            print("  2. Install dependencies: python setup.py --setup")
        if not index_exists:
            print("  3. Run pipeline: python setup.py --pipeline")
        if ollama_status and not missing and index_exists:
            print("  ✓ System ready! Run: python setup.py --query")
        
        return
    
    # Setup mode
    if args.setup:
        print("\n🚀 Running setup...\n")
        
        missing = check_dependencies()
        if missing:
            install_dependencies()
        else:
            print("✓ Dependencies already installed")
        
        print("\n✓ Setup complete!")
        print("\nNext: Run pipeline with --pipeline")
        return
    
    # Pipeline mode
    if args.pipeline:
        if not check_ollama():
            print("\n⚠️  Warning: Ollama server not detected")
            print("   Start with: ollama serve")
        
        run_pipeline()
        print("\n✓ Pipeline complete!")
        print("\nNext: Try queries with --query")
        return
    
    # Query mode
    if args.query:
        if not check_ollama():
            print("\n❌ Error: Ollama server not running")
            print("   Start with: ollama serve")
            sys.exit(1)
        
        if not Path("data/processed/faiss.index").exists():
            print("\n❌ Error: FAISS index not found")
            print("   Run pipeline first: python setup.py --pipeline")
            sys.exit(1)
        
        interactive_query()


if __name__ == "__main__":
    main()
