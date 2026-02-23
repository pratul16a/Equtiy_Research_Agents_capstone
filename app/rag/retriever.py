"""RAG Retriever — load FAISS index and return a LangChain retriever."""

from __future__ import annotations

import os
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

    # Check if the index exists
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
        print(f"⚠️  Failed to load FAISS index: {e}")
        return None


def search(query: str, k: int = 5) -> list[dict]:
    """Convenience function: search the RAG index and return results as dicts.

    Args:
        query: The search query
        k: Number of results to return

    Returns:
        List of dicts with 'content' and 'metadata' keys, or empty list if no index.
    """
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
