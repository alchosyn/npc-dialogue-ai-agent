import json

from .calculator import calculator
from .knowledge import search_knowledge
from .time_tool import get_current_time
from .web_search import web_search

tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "搜索互联网获取实时信息，如天气、新闻、最新事件等",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式，如加减乘除、幂运算等",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，例如 '1832*772' 或 '2+3*5'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "搜索信噪的记忆档案和世界资料。当用户问到信噪的过去、她认识的人、这个世界的规则、或任何需要查阅背景知识的问题时，调用这个工具",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，用自然语言描述要查的内容",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

tool_map = {
    "web_search": lambda args: web_search(args["query"]),
    "get_current_time": lambda args: get_current_time(),
    "calculator": lambda args: calculator(args["expression"]),
    "search_knowledge": lambda args: json.dumps(
        search_knowledge(args["query"]), ensure_ascii=False
    ),
}
