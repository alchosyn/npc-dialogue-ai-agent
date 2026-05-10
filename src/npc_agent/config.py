import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "knowledge_base.json"
HISTORY_FILE = PROJECT_ROOT / "chat_history.json"
TRACE_DIR = PROJECT_ROOT / "data" / "traces"

MAX_MESSAGES = 20
MAX_STEPS = 6
TOP_K = 3
WEB_SEARCH_MAX_RESULTS = 3


def get_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing environment variable {name}. "
            f"Set it in your shell or a .env file before running."
        )
    return value
