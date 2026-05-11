import json

from .calculator import calculator
from .knowledge import search_knowledge
from .risk_score import risk_score
from .time_tool import get_current_time
from .web_search import web_search

tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "搜索互联网获取最新反诈情报、安全公告、案例报道。"
                "优先权威信息源（国家反诈中心、公安部刑侦局、CNCERT/CC、CNVD、网信办、"
                "12321 举报中心、银行/运营商官方公告、知名厂商安全应急响应中心）。"
                "用户问到近期高发手法、新型诈骗、最新案件、漏洞通报时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，建议带上时间词（如 '2024 年 AI 换脸诈骗'）以提升时效性",
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
            "description": "获取当前日期和时间。判断案例时效或回应'最近''今年'类问题时调用",
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
            "description": (
                "计算数学表达式，如加减乘除、幂运算等。"
                "用于密码熵估算（log2(字符集大小^长度)）、损失评估、风险打分等"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，例如 '1832*772' 或 '26**8'",
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
            "description": (
                "搜索内部反诈/安全意识知识库。包含 5 类条目："
                "law-* 法律基线（反电诈法、个人信息保护法等）、"
                "pattern-* 诈骗手法图谱（杀猪盘、冒充公检法、AI 换脸等）、"
                "case-* 真实案例（带金额/地区/时间）、"
                "protect-* 个人防护手册（密码、2FA、隐私设置）、"
                "decision-* 上当后处置流程（96110/110/12321、银行止付）。"
                "用户问到具体诈骗手法、求助案例、防护建议、报案流程时必须调用。"
            ),
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
    {
        "type": "function",
        "function": {
            "name": "risk_score",
            "description": (
                "对一段可疑文本（短信、邮件、通话脚本等）按规则打风险分（0-100），"
                "返回 score、命中的信号清单、处置建议。"
                "用户粘贴可疑内容、转述电话/视频沟通、贴出陌生链接时必须先调用此工具，"
                "再结合 search_knowledge 找类似案例。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "scenario": {
                        "type": "string",
                        "description": "用户粘贴的可疑文本原文，越完整越好",
                    }
                },
                "required": ["scenario"],
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
    "risk_score": lambda args: json.dumps(
        risk_score(args["scenario"]), ensure_ascii=False
    ),
}
