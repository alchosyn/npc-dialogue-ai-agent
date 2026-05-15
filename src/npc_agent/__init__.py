"""npc_agent 包入口。

懒加载设计：`run_chat_loop` / `step` 通过 PEP 562 module __getattr__
延迟导入。这样 `from npc_agent.config import MODEL` 或
`from npc_agent.llm_client import get_client` 不会触发整个 agent 栈
（tools → knowledge → rank_bm25 / sentence-transformers / tavily）。

动机：GRPO 训练的 reward 链路只需要 judge（DeepSeek API），不需要
RAG 栈。之前 __init__.py 直接 `from .agent import ...`，导致任何人
只想拿个 config 常量都被迫加载 400MB embedding 模型 + 一堆重依赖。

`from npc_agent import run_chat_loop` 仍然照常工作（main.py / notebook
不用改）。
"""

__all__ = ["run_chat_loop", "step"]


def __getattr__(name: str):
    # 只有真正访问 npc_agent.run_chat_loop / .step 时才拉 agent 栈
    if name in ("run_chat_loop", "step"):
        from .agent import run_chat_loop, step

        return {"run_chat_loop": run_chat_loop, "step": step}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
