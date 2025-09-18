# === LLM Provider (Gemini only now) ===
LLM_PROVIDER = "gemini"

# === Gemini Settings ===
GEMINI_API_KEY = "AIzaSyAxe-3d67qUKphipqFeuxDUHxsZNZUALa0"
GEMINI_MODEL_ID = "gemini-2.5-pro"
GEMINI_MODEL_NAME = GEMINI_MODEL_ID
# === Vertex AI Embeddings ===
GCP_PROJECT = "startup-analyst-472410"
GCP_LOCATION = "us-central1"
VERTEX_EMBED_MODEL = "text-embedding-004"

DOCAI_PROJECT   = "startup-analyst-472410"   # your GCP project ID
DOCAI_LOCATION  = "us"                       # region shown in console
DOCAI_PROCESSOR = "27defb4a6a1a1ab1"         # processor ID from console

# Tavily (used by tools/search_adapters.py). Leave blank if using --no-web.
TAVILY_API_KEY = "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"
SERPER_API_KEY = "72fce82cf4db0c45b962d906dffe265eaec9199d"        # https://serper.dev
EXA_API_KEY = "6a24a2f9-e7a7-4b67-9367-5fa3094f8397"           # https://exa.ai

# knobs
SEARCH_TIMEOUT = 20        # seconds per request
SEARCH_TOPK_PER_BACKEND = 6
SEARCH_MERGED_TOPK = 20
