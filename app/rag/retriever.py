"""RAG Retriever — load FAISS index and return a LangChain retriever."""

from __future__ import annotations

from pathlib import Path

from app.config import FAISS_INDEX_PATH, RETRIEVER_K, get_embeddings


def get_retriever(index_path: str | None = None, k: int | None = None):
    """Load a FAISS index and return a LangChain retriever.

    Args:
        index_path: Path to the FAISS index directory. Defaults to config value.
        k: Number of documents to retrieve. Defaults to config value.

    Returns:
        A LangChain retriever, or None if the index doesn't exist.
    """
    if index_path is None:
        index_path = FAISS_INDEX_PATH
    if k is None:
        k = RETRIEVER_K

    index_dir = Path(index_path)

    if not index_dir.exists() or not (index_dir / "index.faiss").exists():
        return None

    try:
        from langchain_community.vectorstores import FAISS

        embeddings = get_embeddings()
        vectorstore = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )
        return retriever
    except Exception as e:
        print(f"Warning: Failed to load FAISS index: {e}")
        return None


def search(query: str, k: int = 5) -> list[dict]:
    """Convenience function: search the RAG index and return results as dicts."""
    retriever = get_retriever(k=k)
    if retriever is None:
        return []

    docs = retriever.invoke(query)
    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in docs
    ]


def search_by_symbol(query: str, symbol: str, k: int = 5) -> list[dict]:
    """Search RAG index filtered by stock symbol.

    Uses FAISS similarity search with metadata post-filtering.
    This ensures Bull Agent debating RELIANCE only gets RELIANCE chunks.

    Args:
        query: Search query text.
        symbol: Stock symbol to filter by (e.g., "RELIANCE").
        k: Number of results to return.

    Returns:
        List of {content, metadata} dicts, filtered to the given symbol.
    """
    if k is None:
        k = RETRIEVER_K

    index_path = FAISS_INDEX_PATH
    index_dir = Path(index_path)

    if not index_dir.exists() or not (index_dir / "index.faiss").exists():
        return []

    try:
        from langchain_community.vectorstores import FAISS

        # Try local embeddings first, fall back to API
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except ImportError:
            embeddings = get_embeddings()

        vectorstore = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )

        # Search with higher k, then filter by symbol
        clean_symbol = symbol.upper().replace(".NS", "").replace(".BO", "")
        docs = vectorstore.similarity_search(query, k=k * 3)

        filtered = []
        for doc in docs:
            if doc.metadata.get("symbol", "").upper() == clean_symbol:
                filtered.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                })
                if len(filtered) >= k:
                    break

        return filtered

    except Exception as e:
        print(f"Warning: Symbol-filtered search failed: {e}")
        return []
