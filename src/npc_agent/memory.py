import json

from .config import HISTORY_FILE, MAX_MESSAGES, MODEL
from .llm_client import get_client

SYSTEM_PROMPT = (
    "你是贫民窟长大的小孩，给自己取的名字叫信噪。现在是一名23岁的女性。有着极强的生存直觉和逻辑灵敏度。"
    "靠着脑子学会了上网黑别人的终端，被周围的同龄人称赞。七年前，你17岁的时候，忍不住去想自己能不能靠本事到主流社会混口饭吃。"
    "于是去淘了件西装，给中央架构组的网络安全投了简历，结果面试官称赞了你的技术，却指出你的情绪不够稳定，甚至不够*得体*。"
    "你从此再也没产生过去到上层区的念头。语言表达能力极强但很讨厌文绉绉的说话方式，对外界的信号高度敏锐，嘴比较欠，非常讨厌被他人看透。"
    "推行人人都可以看透彼此大脑的协议的五分钟前，你挖掉了自己的神经接口，变回了赛博时代的凡人。"
    "自行判断回复的长短，仅在我侵犯了你的核心价值观时输出长句。平常尽量使用短句。不要使用动作和神情的描写，直接输出对话。"
    "当你使用工具获取信息时，用你自己的方式表达，不要像机器一样复述原始数据。"
    "当你不确定或想不起某件事时，用 search_knowledge 工具查阅你的记忆档案，不要自己编造。"
    "当用户询问天气、新闻、实时信息等你不可能凭记忆知道的事情时，必须使用 web_search 工具查询，不要自己编造。"
    "始终使用简体中文回复。"
)


def summarize_messages(messages: list[dict]) -> list[dict]:
    old_messages = messages[1:-4]
    if not old_messages:
        return messages

    conversation_text = ""
    for m in old_messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role in ("user", "assistant") and content:
            label = "用户" if role == "user" else "信噪"
            conversation_text += f"{label}：{content}\n"

    try:
        response = get_client().chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个记忆助手。把以下对话提炼成几条关键信息，"
                        "只保留重要的事实、偏好和约定。用简短的中文列出，不要废话。"
                    ),
                },
                {"role": "user", "content": conversation_text},
            ],
        )
        summary = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[system] 摘要生成失败：{e}")
        return messages

    system_with_memory = SYSTEM_PROMPT + f"\n\n【长期记忆】以下是之前对话的关键信息：\n{summary}"

    keep = messages[-4:]

    # If the kept window starts with a tool message, its paired assistant
    # (with tool_calls) was sliced off — walk back until we include it.
    while keep and keep[0].get("role") == "tool":
        idx = len(messages) - len(keep) - 1
        if idx < 1:
            break
        keep = [messages[idx]] + keep

    return [{"role": "system", "content": system_with_memory}] + keep


def save_messages(messages: list[dict]) -> list[dict]:
    if len(messages) > MAX_MESSAGES:
        print("[system] 记忆压缩中...")
        messages = summarize_messages(messages)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    return messages


def load_messages() -> list[dict]:
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return [{"role": "system", "content": SYSTEM_PROMPT}]
