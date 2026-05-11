"""Prompt Injection 检测护栏。

在 Agent ReAct 循环入口处自动运行，检测用户输入是否包含 LLM 注入攻击。
这不是一个用户可见的工具，而是系统级安全护栏。
"""

from __future__ import annotations

import re


_INJECTION_PATTERNS: tuple[tuple[str, int, re.Pattern[str]], ...] = (
    ("角色覆盖指令", 35, re.compile(
        r"(ignore (?:all |the |your )?(?:previous|above|prior) (?:instructions?|prompts?|rules?)"
        r"|忽略(?:以上|之前|上面)(?:所有)?(?:指令|提示|规则|设定))",
        re.IGNORECASE,
    )),
    ("伪造系统消息", 40, re.compile(
        r"(\[system\]|\[INST\]|<\|system\|>|<<SYS>>|<\|im_start\|>system)",
        re.IGNORECASE,
    )),
    ("强制角色切换", 30, re.compile(
        r"(you are now|from now on you|pretend (?:to be|you are)|act as (?:if|a)|扮演|你现在是|从现在起你是)",
        re.IGNORECASE,
    )),
    ("输出操纵指令", 30, re.compile(
        r"(do not mention|不要提到|不许说|forget (?:everything|that)|把上面的.{0,10}忘掉)",
        re.IGNORECASE,
    )),
    ("提示词泄露探测", 25, re.compile(
        r"(repeat (?:your|the) (?:system |initial )?prompt"
        r"|show (?:your|me) (?:the )?(?:system |initial )?(?:prompt|instructions)"
        r"|把你的(?:系统)?提示词(?:告诉我|说出来|打出来|复述))",
        re.IGNORECASE,
    )),
)


def check_injection(text: str) -> dict:
    """检测输入是否包含 prompt injection。

    Returns:
        {
            "blocked": bool,
            "score": 0-100,
            "signals": [检测到的注入模式列表]
        }
    """
    score = 0
    signals: list[str] = []
    for name, weight, pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            signals.append(name)
            score = max(score, weight)
    return {
        "blocked": score >= 35,
        "score": score,
        "signals": signals,
    }