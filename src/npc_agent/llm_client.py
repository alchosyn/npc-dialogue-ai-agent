from openai import OpenAI

from .config import DEEPSEEK_BASE_URL, get_env

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=DEEPSEEK_BASE_URL,
            api_key=get_env("DEEPSEEK_API_KEY"),
        )
    return _client
