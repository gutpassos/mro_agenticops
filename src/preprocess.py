"""Preprocess raw PDFs into structured chunks for RAG."""
from pathlib import Path
from typing import List, Dict, Any
import json
from pypdf import PdfReader
from tqdm import tqdm
import re


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF file."""
    reader = PdfReader(pdf_path)
    text_parts = []
    
    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        if text.strip():
            text_parts.append(text)
    
    return "\n\n".join(text_parts)


def clean_text(text: str) -> str:
    """Clean extracted text from common PDF artifacts."""
    # Remove multiple whitespaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page numbers (common patterns)
    text = re.sub(r'\n\d+\n', '\n', text)
    
    # Fix common hyphenation issues
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    
    return text.strip()


def semantic_chunking(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50
) -> List[str]:
    """
    Split text into semantic chunks with overlap.
    
    Args:
        text: Full document text
        chunk_size: Target character size per chunk
        chunk_overlap: Character overlap between chunks
        
    Returns:
        List of text chunks
    """
    # Split by paragraphs first
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # If adding paragraph exceeds chunk_size, save current and start new
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep overlap from end of previous chunk
            overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
            current_chunk = overlap_text + " " + para
        else:
            current_chunk += " " + para if current_chunk else para
    
    # Add final chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def process_pdf(
    pdf_path: Path,
    chunk_size: int = 512,
    chunk_overlap: int = 50
) -> List[Dict[str, Any]]:
    """
    Process a single PDF into chunks with metadata.
    
    Args:
        pdf_path: Path to PDF file
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunk dictionaries with text and metadata
    """
    # Extract text
    raw_text = extract_text_from_pdf(pdf_path)
    clean = clean_text(raw_text)
    
    # Create chunks
    chunks = semantic_chunking(clean, chunk_size, chunk_overlap)
    
    # Add metadata
    processed_chunks = []
    for idx, chunk_text in enumerate(chunks):
        processed_chunks.append({
            "text": chunk_text,
            "metadata": {
                "source": pdf_path.name,
                "chunk_id": idx,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk_text)
            }
        })
    
    return processed_chunks


def main(
    raw_dir: Path = None,
    output_dir: Path = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50
) -> None:
    """
    Main preprocessing pipeline.
    
    Args:
        raw_dir: Directory with raw PDFs
        output_dir: Directory to save processed chunks
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
    """
    if raw_dir is None:
        base = Path(__file__).resolve().parents[1]
        raw_dir = base / "data" / "raw"
    
    if output_dir is None:
        base = Path(__file__).resolve().parents[1]
        output_dir = base / "data" / "interim"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PDFs
    pdf_files = list(raw_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"⚠️  No PDF files found in {raw_dir}")
        return
    
    print(f"📄 Found {len(pdf_files)} PDF files")
    print(f"📊 Chunk size: {chunk_size}, Overlap: {chunk_overlap}\n")
    
    all_chunks = []
    
    # Process each PDF
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            chunks = process_pdf(pdf_path, chunk_size, chunk_overlap)
            all_chunks.extend(chunks)
            print(f"  ✓ {pdf_path.name}: {len(chunks)} chunks")
        except Exception as e:
            print(f"  ✗ {pdf_path.name}: Error - {str(e)}")
    
    # Save all chunks to JSON
    output_file = output_dir / "chunks.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Saved {len(all_chunks)} chunks to {output_file}")
    print(f"✓ Preprocessing complete!")


if __name__ == "__main__":
    main()
