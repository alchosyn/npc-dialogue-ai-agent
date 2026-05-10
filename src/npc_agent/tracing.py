import datetime
import json
import os
import uuid

from .config import TRACE_DIR


def new_trace(user_input: str) -> dict:
    return {
        "trace_id": str(uuid.uuid4())[:8],
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_input": user_input,
        "steps": [],
        "total_tokens": 0,
        "total_latency_ms": 0,
    }


def log_llm_call(trace: dict, response, latency_ms: int) -> None:
    usage = response.usage
    tokens_used = usage.prompt_tokens + usage.completion_tokens if usage else 0
    trace["steps"].append({
        "type": "llm_call",
        "latency_ms": latency_ms,
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        "has_tool_calls": bool(response.choices[0].message.tool_calls),
    })
    trace["total_tokens"] += tokens_used
    trace["total_latency_ms"] += latency_ms


def log_tool_call(trace: dict, name: str, args: dict, result: str, latency_ms: int) -> None:
    trace["steps"].append({
        "type": "tool_call",
        "tool": name,
        "input": args,
        "output": result[:200],
        "latency_ms": latency_ms,
    })
    trace["total_latency_ms"] += latency_ms


def save_trace(trace: dict, reply: str) -> None:
    trace["agent_reply"] = reply
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    day_dir = os.path.join(TRACE_DIR, today)
    os.makedirs(day_dir, exist_ok=True)
    path = os.path.join(day_dir, f"{trace['trace_id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)
