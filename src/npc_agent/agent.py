import json
import time

from .config import MAX_STEPS, MODEL
from .llm_client import get_client
from .memory import load_messages, save_messages
from .tools import tool_map, tools
from .tracing import log_llm_call, log_tool_call, new_trace, save_trace
from .utils import clean_reply


def step(messages: list[dict], user_input: str) -> tuple[str, list[dict]]:
    """Run one turn: append user input, drive the ReAct loop, return (reply, messages)."""
    client = get_client()
    messages.append({"role": "user", "content": user_input})
    trace = new_trace(user_input)
    reply: str | None = None

    for _ in range(MAX_STEPS):
        try:
            t0 = time.time()
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
            )
            log_llm_call(trace, response, int((time.time() - t0) * 1000))
        except Exception:
            reply = "……信号不太好 你再说一遍"
            break

        msg = response.choices[0].message

        if not msg.tool_calls:
            reply = clean_reply(msg.content)
            break

        messages.append(msg.to_dict())

        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                t0 = time.time()
                result = tool_map[name](args)
                log_tool_call(trace, name, args, result, int((time.time() - t0) * 1000))
            except Exception as e:
                result = f"工具执行出错：{e}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    if reply is None:
        reply = "想了半天没想明白 你换个方式问问"

    messages.append({"role": "assistant", "content": reply})
    save_trace(trace, reply)
    messages = save_messages(messages)
    return reply, messages


def run_chat_loop() -> None:
    messages = load_messages()
    while True:
        user_input = input("你：")
        if user_input == "quit":
            break
        reply, messages = step(messages, user_input)
        print(f"信噪：{reply}")
