from tavily import TavilyClient

from ..config import WEB_SEARCH_MAX_RESULTS, get_env

_client: TavilyClient | None = None


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        _client = TavilyClient(api_key=get_env("TAVILY_API_KEY"))
    return _client


def web_search(query: str) -> str:
    try:
        response = _get_client().search(query, max_results=WEB_SEARCH_MAX_RESULTS)
        output = ""
        for r in response["results"]:
            output += f"标题：{r['title']}\n内容：{r['content']}\n\n"
        return output.strip()
    except Exception as e:
        return f"搜索失败：{e}"
