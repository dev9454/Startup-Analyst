
# Startup Analyst — Multi-Agent AI for Startup Evaluation

This project is a hackathon prototype of an **AI Analyst** that reviews startup materials (pitch decks, websites, bios, market reports) and produces investor-ready deal notes.

It uses a **multi-agent system built on LangChain + AWS Bedrock (Llama 3-70B)** with modular agents for ingestion, verification, peer benchmarking, scoring, risk flagging, sector insights, and more.

---

## 🔧 Setup

### 1. Clone & install requirements
```bash
git clone <your_repo_url>
cd Startup-analyst
pip install -r requirements.txt
```
### Edit config.py in the project root:
# config.py
```bash
# Bedrock (used in tools/llm.py)
BEDROCK_REGION = "us-east-1"
BEDROCK_MODEL_ID = "meta.llama3-70b-instruct-v1:0"

# Search backends (optional: leave blank to disable)
TAVILY_API_KEY = "your_tavily_api_key"
SERPER_API_KEY = "your_serper_api_key"
EXA_API_KEY = "your_exa_api_key"

# Search settings
SEARCH_TIMEOUT = 20
SEARCH_TOPK_PER_BACKEND = 6
SEARCH_MERGED_TOPK = 20
```

### Running the system
``` bash 
python main.py --company "Multipli" \
  --sector fintech \
  --inputs "data/Investment_Template.pdf" "data/Multipl_Pitch.pdf" "https://multipl.in/our_story"
```
