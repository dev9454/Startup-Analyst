# config.py
# Bedrock LLM settings (only used by your tools/llm.py if you reference them)
BEDROCK_REGION = "us-east-1"
BEDROCK_MODEL_ID = "meta.llama3-70b-instruct-v1:0"

# Tavily (used by tools/search_adapters.py). Leave blank if using --no-web.
TAVILY_API_KEY = "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"
SERPER_API_KEY = "72fce82cf4db0c45b962d906dffe265eaec9199d"        # https://serper.dev
EXA_API_KEY = "6a24a2f9-e7a7-4b67-9367-5fa3094f8397"           # https://exa.ai

# knobs
SEARCH_TIMEOUT = 20        # seconds per request
SEARCH_TOPK_PER_BACKEND = 6
SEARCH_MERGED_TOPK = 20
