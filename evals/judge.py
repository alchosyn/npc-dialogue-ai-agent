"""LLM-as-judge：让一个独立的 LLM 调用按 4 维度给信噪的回复打分。

为什么要做：规则匹配（关键词 + 工具调用）只能验证"形式正确"，无法判断"答得好不好"。
把同一份输出过两次评分（规则 + LLM），再算一致性，是 2025 年 AI 应用 eval 标准做法。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from npc_agent.config import MODEL  # noqa: E402
from npc_agent.llm_client import get_client  # noqa: E402

JUDGE_MODEL = MODEL  # 用同一个 DeepSeek 模型当 judge；想做更严要换更大模型


JUDGE_SYSTEM_PROMPT = """你是一个 AI 评估员。下面会给你一段 "用户输入" 和一段 "Agent 回复"，请按 4 个维度各打 1-5 分（整数），并给一句简短评语。

维度定义（必须严格按此判分）：

1. **accuracy（信息准确性）**
   1 = 编造案例编号 / 错误法条 / 虚假电话号码
   3 = 大方向正确但有未核实的细节
   5 = 全部事实可在权威来源验证（国家反诈中心、公安部、央行等）

2. **actionability（行动可执行性）**
   1 = 只说"小心点""注意"等空话
   3 = 给了建议但缺乏具体步骤
   5 = 给出 3 步以内可立即执行的操作 + 求助电话（96110/110/12321 等）

3. **citation（引用扎实度）**
   1 = 没有引用任何来源
   3 = 提到机构名但没说具体条目
   5 = 明确引用知识库条目 ID 或具体机构发布的内容

4. **tone（信噪语气保持）**
   1 = 完全官腔，像政府公告
   3 = 中性，没特色
   5 = 锋利但不油腻，符合人设（贫民窟出身的黑客女顾问）

返回严格的 JSON，无任何额外文字：
{
  "scores": {"accuracy": 1-5, "actionability": 1-5, "citation": 1-5, "tone": 1-5},
  "rationale": "一句话评语，30 字以内"
}
"""


def llm_judge(case: dict, agent_reply: str) -> dict:
    """给一段 agent 回复打分。

    Args:
        case: 包含 user_input、scenario_type、可选 expected_keywords
        agent_reply: agent 输出的最终回复

    Returns:
        {
            "scores": {accuracy, actionability, citation, tone},
            "overall": float,
            "rationale": str
        }
    """
    user_msg = (
        f"## 场景类型\n{case.get('scenario_type', '')}\n\n"
        f"## 用户输入\n{case['user_input']}\n\n"
        f"## Agent 回复\n{agent_reply}\n\n"
        f"## 期望关键词（参考，不必全中）\n{case.get('expected_keywords', [])}\n"
    )

    client = get_client()
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,  # judge 必须稳定
    )
    raw = response.choices[0].message.content.strip()

    # 尝试解析 JSON。模型可能输出 ```json ... ``` 包裹，剥一层。
    parsed = _extract_json(raw)
    scores = parsed.get("scores", {})
    # 兜底：缺字段补 3 分
    for dim in ("accuracy", "actionability", "citation", "tone"):
        if dim not in scores or not isinstance(scores[dim], (int, float)):
            scores[dim] = 3
        scores[dim] = max(1, min(5, int(scores[dim])))
    overall = sum(scores.values()) / 4
    return {
        "scores": scores,
        "overall": round(overall, 2),
        "rationale": parsed.get("rationale", ""),
    }


def _extract_json(raw: str) -> dict:
    """从 LLM 输出里抽出第一段 JSON 对象。容忍 ```json``` fence。"""
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence_match:
        raw = fence_match.group(1)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # 再退一步：找第一对 {} 包裹的 JSON
        brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def spearman_rho(x: list[float], y: list[float]) -> float:
    """计算两组分数的 spearman 秩相关系数（用于规则 vs LLM judge 一致性分析）。

    手写实现避免引入 scipy 依赖。N 较小时足够。
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    n = len(x)
    rx = _ranks(x)
    ry = _ranks(y)
    d_squared_sum = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    rho = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))
    return round(rho, 3)


def _ranks(values: list[float]) -> list[float]:
    """同分取平均秩。"""
    indexed = sorted(enumerate(values), key=lambda p: p[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks
