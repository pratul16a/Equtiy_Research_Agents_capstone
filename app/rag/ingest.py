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


def get_local_embeddings():
    """Get local HuggingFace embeddings (free, no API key needed)."""
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except ImportError:
        print("HuggingFace embeddings not available, falling back to API embeddings")
        return get_embeddings()


def ingest_screener_data(
    symbols: list[str],
    index_path: str | None = None,
    use_local_embeddings: bool = True,
) -> None:
    """Ingest Screener.in data for given symbols into FAISS.

    Creates Document objects with symbol metadata for filtered retrieval.
    Uses local embeddings by default (free, fast).
    """
    from langchain_core.documents import Document
    from app.rag.screener_scraper import scrape_company_page

    if index_path is None:
        index_path = FAISS_INDEX_PATH

    all_docs: list[Document] = []

    for symbol in symbols:
        clean_symbol = symbol.upper().replace(".NS", "").replace(".BO", "")
        try:
            data = scrape_company_page(clean_symbol)
            if "error" in data:
                print(f"  Skipping {clean_symbol}: {data.get('error')}")
                continue

            raw_text = data.get("raw_text", "")
            if not raw_text:
                continue

            # Create documents with metadata for symbol-based filtering
            section_keys = ["quarterly_results", "annual_pl", "balance_sheet",
                          "cash_flows", "ratios", "shareholding"]

            for section_key in section_keys:
                section = data.get(section_key)
                if section and isinstance(section, dict) and section.get("text"):
                    doc = Document(
                        page_content=section["text"],
                        metadata={
                            "symbol": clean_symbol,
                            "section": section_key,
                            "source": "screener.in",
                        },
                    )
                    all_docs.append(doc)

            # Pros/cons as a single document
            pros_cons = data.get("pros_cons", {})
            if pros_cons.get("pros") or pros_cons.get("cons"):
                pros_text = "\n".join(f"PRO: {p}" for p in pros_cons.get("pros", []))
                cons_text = "\n".join(f"CON: {c}" for c in pros_cons.get("cons", []))
                doc = Document(
                    page_content=f"## {clean_symbol} Pros and Cons\n{pros_text}\n{cons_text}",
                    metadata={"symbol": clean_symbol, "section": "pros_cons", "source": "screener.in"},
                )
                all_docs.append(doc)

            print(f"  Prepared {clean_symbol}: {len([d for d in all_docs if d.metadata['symbol'] == clean_symbol])} sections")

        except Exception as e:
            print(f"  Error processing {clean_symbol}: {e}")

    if not all_docs:
        print("No documents to ingest")
        return

    # Chunk documents
    chunks = chunk_documents(all_docs)

    # Preserve metadata through chunking
    print(f"Total chunks: {len(chunks)} from {len(symbols)} symbols")

    # Create or merge into FAISS index
    embeddings = get_local_embeddings() if use_local_embeddings else get_embeddings()

    from langchain_community.vectorstores import FAISS

    index_dir = Path(index_path)
    if index_dir.exists() and (index_dir / "index.faiss").exists():
        # Merge into existing index
        print("Merging into existing FAISS index...")
        existing = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
        new_store = FAISS.from_documents(chunks, embeddings)
        existing.merge_from(new_store)
        existing.save_local(str(index_path))
    else:
        # Create new index
        print("Creating new FAISS index...")
        os.makedirs(index_path, exist_ok=True)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(str(index_path))

    print(f"Screener.in RAG ingestion complete: {len(chunks)} chunks indexed")


if __name__ == "__main__":
    run_ingestion()
