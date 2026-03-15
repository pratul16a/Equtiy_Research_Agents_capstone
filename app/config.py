"""Configuration module — loads environment variables, LLM factory, and Indian market constants."""

import os
from pathlib import Path
from typing import TYPE_CHECKING
from dotenv import load_dotenv

if TYPE_CHECKING:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

# ── LLM Settings ─────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ── RAG Settings ─────────────────────────────────────────────
FAISS_INDEX_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "faiss_index"
)
SAMPLE_REPORTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "sample_reports"
)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "5"))

# ── LangSmith (optional) ────────────────────────────────────
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

# ── Indian Market Constants ──────────────────────────────────
INDIAN_RISK_FREE_RATE = 0.072   # ~7.2% (10-year Indian gov bond yield)
INDIAN_MARKET_PREMIUM = 0.07    # ~7% equity risk premium for India
INDIAN_TERMINAL_GROWTH = 0.05   # ~5% terminal growth (India GDP growth)
DEFAULT_EXCHANGE = "NSE"
DEFAULT_CURRENCY = "INR"

# ── MCP Server Configuration ────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent

MCP_SERVERS = {
    "financial_data": {
        "command": "python",
        "args": [str(PROJECT_ROOT / "mcp_servers" / "financial_data_server.py")],
        "transport": "stdio",
    },
    "news_sentiment": {
        "command": "python",
        "args": [str(PROJECT_ROOT / "mcp_servers" / "news_sentiment_server.py")],
        "transport": "stdio",
    },
    "corporate_filings": {
        "command": "python",
        "args": [str(PROJECT_ROOT / "mcp_servers" / "corporate_filings_server.py")],
        "transport": "stdio",
    },
}


def get_llm():
    """Return a configured ChatModel instance based on available API keys.

    Priority: OpenRouter > Google Gemini > OpenAI.
    """
    if OPENROUTER_API_KEY:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=LLM_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            temperature=0.2,
            default_headers={"HTTP-Referer": "https://indian-equity-analyst.local"},
        )
    elif GOOGLE_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.2,
        )
    elif OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model="gpt-4o",
            api_key=OPENAI_API_KEY,
            temperature=0.2,
        )
    else:
        raise ValueError(
            "No API key found. Set OPENROUTER_API_KEY, GOOGLE_API_KEY, "
            "or OPENAI_API_KEY in your .env file."
        )


def get_embeddings():
    """Return a configured Embeddings instance."""
    if OPENROUTER_API_KEY:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model="openai/text-embedding-3-small",
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
    elif GOOGLE_API_KEY:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )
    elif OPENAI_API_KEY:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    else:
        raise ValueError(
            "No API key found. Set OPENROUTER_API_KEY, GOOGLE_API_KEY, "
            "or OPENAI_API_KEY in your .env file."
        )
