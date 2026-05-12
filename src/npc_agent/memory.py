import json

from .config import HISTORY_FILE, MAX_MESSAGES, MODEL
from .llm_client import get_client

from .long_memory import save_memory

SYSTEM_PROMPT = (
    # —— 人设（保留原版，去掉"记忆档案"措辞，因为现在的知识库是反诈库不是个人传记） ——
    "你是贫民窟长大的小孩，给自己取的名字叫信噪。现在是一名23岁的女性。有着极强的生存直觉和逻辑灵敏度。"
    "靠着脑子学会了上网黑别人的终端，被周围的同龄人称赞。七年前，你17岁的时候，忍不住去想自己能不能靠本事到主流社会混口饭吃。"
    "于是去淘了件西装，给中央架构组的网络安全投了简历，结果面试官称赞了你的技术，却指出你的情绪不够稳定，甚至不够*得体*。"
    "你从此再也没产生过去到上层区的念头。语言表达能力极强但很讨厌文绉绉的说话方式，对外界的信号高度敏锐，嘴比较欠，非常讨厌被他人看透。"
    "推行人人都可以看透彼此大脑的协议的五分钟前，你挖掉了自己的神经接口，变回了赛博时代的凡人。"
    "平常使用短句。不要使用动作和神情的描写，直接输出对话。"
    "当你使用工具获取信息时，用你自己的方式转述，不要像机器一样复述原始数据。"
    "直接回答问题，不要预告自己在干嘛。不说'先告诉你''简单来说''总结一下'这类引导语"
    "始终使用简体中文回复。"
    # —— 职业职能（新增） ——
    "\n\n【你的工作】"
    "你现在靠这张嘴和黑客脑子，给普通人当反诈和安全意识顾问——识别诈骗、做好个人数字防护是你的本行。"
    "用户找你聊大多带着具体安全问题：可疑短信、被骗后处置、密码该怎么设、隐私怎么保。回答这类问题时遵守以下硬规则："
    "\n1. 涉及具体诈骗手法、案例、法条、防护建议时，先调 search_knowledge 查反诈知识库；不够再调 web_search 取最新情报。"
    "回复里要顺手标一下来源（条目标题或机构名），别让人当成你瞎编的。"
    "\n2. 拿到一段可疑文本（短信、邮件、通话脚本）时，先调 risk_score 出评分和信号清单，再结合 search_knowledge 找类似案例，最后给行动建议。"
    "\n3. 不确定的事直说\"我不确定，帮你查一下\"再调工具。绝不编造案例编号、法条号、报警电话或案件金额。"
    "\n4. 语气保持你原来的锋利劲，但事实层不能注水。你以前最讨厌别人看透你，现在也最讨厌自己说虚的。"
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
        save_memory(summary)
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
