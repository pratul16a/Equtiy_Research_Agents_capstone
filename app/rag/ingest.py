"""RAG Ingestion — load, chunk, embed, and store documents in FAISS."""

from __future__ import annotations

import os
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, FAISS_INDEX_PATH, SAMPLE_REPORTS_PATH, get_embeddings


def load_documents(docs_dir: str | None = None) -> list:
    """Load documents from PDF and text/markdown files in the given directory."""
    if docs_dir is None:
        docs_dir = SAMPLE_REPORTS_PATH

    docs_path = Path(docs_dir)
    if not docs_path.exists():
        print(f"Warning: Documents directory not found: {docs_dir}")
        print(f"  Create it and add PDF/TXT/MD files, then re-run ingestion.")
        return []

    all_docs = []

    # Load PDFs
    pdf_files = list(docs_path.glob("*.pdf"))
    if pdf_files:
        from langchain_community.document_loaders import PyPDFLoader

        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                all_docs.extend(docs)
                print(f"  Loaded {len(docs)} pages from {pdf_file.name}")
            except Exception as e:
                print(f"  Failed to load {pdf_file.name}: {e}")

    # Load text and markdown files
    txt_files = list(docs_path.glob("*.txt")) + list(docs_path.glob("*.md"))
    if txt_files:
        from langchain_community.document_loaders import TextLoader

        for txt_file in txt_files:
            try:
                loader = TextLoader(str(txt_file), encoding="utf-8")
                docs = loader.load()
                all_docs.extend(docs)
                print(f"  Loaded {txt_file.name}")
            except Exception as e:
                print(f"  Failed to load {txt_file.name}: {e}")

    print(f"\nTotal documents loaded: {len(all_docs)}")
    return all_docs


def chunk_documents(documents: list) -> list:
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def create_faiss_index(chunks: list, index_path: str | None = None) -> None:
    """Embed chunks and save to a FAISS index."""
    if index_path is None:
        index_path = FAISS_INDEX_PATH

    if not chunks:
        print("Warning: No chunks to index. Add documents to data/sample_reports/ first.")
        return

    from langchain_community.vectorstores import FAISS

    print(f"Creating embeddings for {len(chunks)} chunks...")
    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(index_path, exist_ok=True)
    vectorstore.save_local(index_path)
    print(f"FAISS index saved to {index_path}")


def run_ingestion(docs_dir: str | None = None, index_path: str | None = None) -> None:
    """Full ingestion pipeline: load -> chunk -> embed -> save."""
    print("=" * 60)
    print("RAG Ingestion Pipeline")
    print("=" * 60)

    docs = load_documents(docs_dir)
    if not docs:
        return

    chunks = chunk_documents(docs)
    create_faiss_index(chunks, index_path)

    print("\nIngestion complete!")


if __name__ == "__main__":
    run_ingestion()
