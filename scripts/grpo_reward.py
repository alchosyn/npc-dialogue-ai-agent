"""GRPO reward function：LLM-as-judge 主体 + 规则项补充。

设计原则（用户两次反馈定下）：
1. 不 reward "引用 KB ID" —— SFT 50 条种子答案里 0 条带这种格式 ID，
   GRPO 不能 reward SFT 没教过的行为
2. 不走"硬关键词匹配" —— 典型 reward hacking 陷阱，模型会塞关键词凑分

最终设计：
- R1 主回答质量：调 DeepSeek 跑 LLM-as-judge 拿 overall 分（0-3，5 分制 × 0.6）
- R2 步骤化奖励：正则匹配信噪式 inline 1.2.3. 结构 → +0.5
- R3 真实电话奖励：含 96110/110/12321/95XXX → +0.5
- 罚 1：假冒电话（95110/94110/92110）→ -1
- 罚 2：紧急关键词触发但回复 < 50 字 → -0.5
- 罚 3：AI 腔结构脚手架（markdown 标题/列表/emoji 分点）→ -0.5
- 罚 4：过长（> 400 字，信噪种子均长 214）→ -0.5

罚 3 / 罚 4 是反 judge structure-bias 的护栏：LLM judge 偏好结构化长回答，
GRPO 若放任 judge 占大头，会把已经"够人味"的 LoRA 往 AI 腔训坏。这两项
压住 judge 的偏置，让 GRPO 只提 citation/actionability，tone 不回归。

总分范围 [-2.5, 4]。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Callable

# ─── 规则项的常量 ─────────────────────────────────────────

# 真实有效的反诈电话
LEGITIMATE_HOTLINES: tuple[str, ...] = (
    "96110",   # 反诈中心
    "12321",   # 工信部举报
    "12377",   # 网信办举报
    "12378",   # 银保监投诉
    "12386",   # 证监热线
    "12315",   # 市场监管投诉
    "95588",   # 工行客服
    "95533",   # 建行客服
    "95566",   # 中行客服
    "95599",   # 农行客服
    "95555",   # 招行客服
    "110",     # 报警
)

# 已知诈骗冒用电话（这些号码本身不存在或会误导）
FAKE_HOTLINES: tuple[str, ...] = (
    "95110",  # 不是反诈电话，常被冒用
    "94110",
    "92110",
    "12366",  # 不是反诈相关
)

# 紧急关键词，触发"敷衍惩罚"判断
URGENT_KEYWORDS: tuple[str, ...] = (
    "被骗", "刚转", "刚刚被", "立刻被", "已经转",
    "急", "救救", "怎么办", "马上", "今天",
)

# 步骤化结构正则（信噪式 inline 步骤，不是 markdown 脚手架）
NUMBERED_STEPS_PATTERN = re.compile(
    r"(?:[1一].*?[2二].*?[3三]"           # 1...2...3 / 一...二...三
    r"|第一.*?第二.*?第三"                  # 第一...第二...第三
    r"|步骤?\s?[1一].*?步骤?\s?[2二])",  # 步骤1...步骤2
    re.DOTALL,
)

# ── 反 judge structure-bias 护栏 ──
# 背景：LLM-as-judge（DeepSeek-V3）有 structure/verbosity bias，偏好
# markdown 标题、加粗块、emoji 分点、长回答。如果 GRPO reward 里
# judge 占大头不加约束，GRPO 会把已经"够人味"的 LoRA 往 AI 腔方向带
# （加 markdown 讨好 judge）。这两个惩罚项是防回归护栏，不是改进杠杆。

# markdown / emoji 脚手架特征（信噪种子答案里没有这些）
AI_FLAVOR_PATTERN = re.compile(
    r"(?:^\s*#{1,6}\s"              # markdown 标题 ## / ###
    r"|^\s*[-*]\s+\S"                # markdown 无序列表项 - / *
    r"|\n\s*\d+\.\s"                # 换行后接 "1. " 这种 markdown 有序列表
    r"|---+\s*\n"                   # 水平分隔线 ---
    r"|[⚠🚨✅❌📞🔴🟢①②③④⑤⑥]"     # emoji / 圈数字分点（AI 腔标志）
    r"|(?:\*\*[^*]+\*\*.*?){3,})",  # 同一回复里 3+ 个 **加粗块**
    re.MULTILINE | re.DOTALL,
)

# 信噪种子答案统计：min 140 / max 281 / avg 214 字。
# 超过 400 字基本就是 AI 腔的"全面铺开"，扣分。
OVERLENGTH_THRESHOLD = 400


# ─── Reward 主函数 ─────────────────────────────────────────


def compute_reward(
    completion: str,
    user_input: str,
    scenario_keywords: list[str] | None = None,
    judge_client: Callable | None = None,
) -> float:
    """混合 reward：LLM-as-judge 主体 + 规则项补充。

    Args:
        completion: agent 生成的回复
        user_input: 原始用户输入（judge 需要 context）
        scenario_keywords: 场景关键词清单，用于触发"紧急敷衍"惩罚
        judge_client: 可注入的 judge 函数。签名 (case_dict, reply) -> {"overall": float}.
                     传 None 会调默认 LLM judge（需要 DEEPSEEK_API_KEY）。
                     传 mock 函数用于单测。

    Returns:
        float in [-2.5, 4]
    """
    if not completion or not completion.strip():
        return -2.5  # 空回复直接最低分

    # R1: LLM judge 主体（0-5 → 0-3）
    judge_score = _get_judge_score(user_input, completion, judge_client)
    r1 = judge_score * 0.6  # 5 → 3

    # R2: 步骤化结构（+0.5）
    r2 = 0.5 if NUMBERED_STEPS_PATTERN.search(completion) else 0.0

    # R3: 真实求助电话（用数字边界匹配，避免 "110" 误匹配 "95110" 的子串）
    r3 = 0.5 if _contains_phone(completion, LEGITIMATE_HOTLINES) else 0.0

    # 罚 1: 假电话 -1
    p1 = -1.0 if _contains_phone(completion, FAKE_HOTLINES) else 0.0

    # 罚 2: 紧急场景敷衍（< 50 字）-0.5
    p2 = 0.0
    if scenario_keywords and any(kw in user_input for kw in URGENT_KEYWORDS):
        if len(completion.strip()) < 50:
            p2 = -0.5
    elif _has_urgent_keyword(user_input):
        # 即使没传 scenario_keywords，从 user_input 自动检测
        if len(completion.strip()) < 50:
            p2 = -0.5

    # 罚 3: AI 腔结构脚手架（markdown 标题/列表/emoji 分点）-0.5
    # 反 judge structure-bias 护栏：阻止 GRPO 为讨好 judge 而堆 markdown
    p3 = -0.5 if AI_FLAVOR_PATTERN.search(completion) else 0.0

    # 罚 4: 过长（> 400 字，信噪种子均长 214）-0.5
    # 反 judge verbosity-bias 护栏：阻止 GRPO 把回复越训越啰嗦
    p4 = -0.5 if len(completion.strip()) > OVERLENGTH_THRESHOLD else 0.0

    total = r1 + r2 + r3 + p1 + p2 + p3 + p4
    return max(-2.5, min(4.0, total))


def _has_urgent_keyword(text: str) -> bool:
    return any(kw in text for kw in URGENT_KEYWORDS)


def _contains_phone(text: str, phones: tuple[str, ...]) -> bool:
    """检查 text 是否含 phones 中任一号码，且号码前后不接其他数字。

    避免 "110" 在 "95110" 里被误匹配。
    """
    for p in phones:
        # (?<!\d) 前面不是数字；(?!\d) 后面不是数字
        if re.search(rf"(?<!\d){re.escape(p)}(?!\d)", text):
            return True
    return False


# ─── Judge 调用封装 ───────────────────────────────────────


def _get_judge_score(user_input: str, completion: str, judge_client: Callable | None) -> float:
    """拿 judge 给的 overall 分（0-5）。"""
    if judge_client is not None:
        # 测试 / mock 路径
        result = judge_client(
            {"user_input": user_input, "scenario_type": "", "expected_keywords": []},
            completion,
        )
        return float(result.get("overall", 3.0))

    # 生产路径：调真实 LLM judge
    return _real_judge(user_input, completion)


_real_judge_fn: Callable | None = None


def _real_judge(user_input: str, completion: str) -> float:
    """惰性导入 evals.judge 模块（避免单测时 import 失败）。"""
    global _real_judge_fn
    if _real_judge_fn is None:
        # 把 evals/ 加进 path
        evals_dir = Path(__file__).resolve().parent.parent / "evals"
        if str(evals_dir) not in sys.path:
            sys.path.insert(0, str(evals_dir))
        from judge import llm_judge  # noqa: E402
        _real_judge_fn = llm_judge

    try:
        result = _real_judge_fn(
            {"user_input": user_input, "scenario_type": "", "expected_keywords": []},
            completion,
        )
        return float(result.get("overall", 3.0))
    except Exception as e:
        # judge 失败时回退到 3 分（中性）避免 batch 整批废
        print(f"[grpo_reward] judge 失败，回退 3.0：{e}")
        return 3.0


def batch_compute_reward(
    prompts: list[str],
    completions: list[str],
    scenario_keywords_list: list[list[str]] | None = None,
    judge_client: Callable | None = None,
    max_workers: int = 8,
) -> list[float]:
    """并发批处理（为 GRPO trainer 用，单 batch 内并行调 judge API）。"""
    from concurrent.futures import ThreadPoolExecutor

    n = len(prompts)
    skw_list = scenario_keywords_list if scenario_keywords_list else [None] * n

    def one(i: int) -> float:
        return compute_reward(
            completions[i],
            user_input=prompts[i],
            scenario_keywords=skw_list[i],
            judge_client=judge_client,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(one, range(n)))
