import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    # --- GEMINI (LLM + embeddings) ---
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    # Model IDs are placeholders: confirm the current free-tier IDs in Google AI Studio.
    GEMINI_MODEL_PRIMARY = os.getenv("GEMINI_MODEL_PRIMARY", "gemini-3.8-flash")
    GEMINI_MODEL_FALLBACK = os.getenv("GEMINI_MODEL_FALLBACK", "gemini-3.5-flash-lite")

    # --- DOMAIN ---
    # What the assistant is about; used by the planner prompt and the guardrail messages.
    DOMAIN_TOPICS = os.getenv("DOMAIN_TOPICS", "Unity 3D engine and game development")

    # --- VECTOR DB (QDRANT) ---
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = "advanced_rag"

    # --- GUARDRAILS ---
    # Opt-in: match intents with embeddings only (NeMo `embeddings_only`). Test before enabling.
    GUARDRAILS_EMBEDDINGS_ONLY = os.getenv("GUARDRAILS_EMBEDDINGS_ONLY", "false").lower() == "true"

    # --- OBSERVABILITY ---
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "AdvancedRAG")
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

settings = Settings()

# LangSmith tracing is switched on only when a key exists; otherwise LangChain tries (and fails) to upload every trace.
_langsmith_on = bool(settings.LANGSMITH_API_KEY) and str(settings.LANGSMITH_TRACING).lower() == "true"
os.environ["LANGCHAIN_TRACING_V2"] = "true" if _langsmith_on else "false"
os.environ["LANGSMITH_TRACING"] = "true" if _langsmith_on else "false"
if _langsmith_on:
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
