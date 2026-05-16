"""把 evals/cases.json 的 15 条 + data/sft_seeds.json 的 50 条合并成 GRPO 训练数据。

输出 data/grpo_train.jsonl，每行是**对话式 prompt**（套 SFT 同款 system 人设，
TRL GRPOTrainer 会自动 apply chat template）：
    {"prompt": [{"role": "system", "content": "<信噪人设>"},
                {"role": "user", "content": "..."}],
     "scenario_keywords": ["被骗", "刚转"], "_source": "..."}

为什么对话式：SFT LoRA 在带 system 人设 + Qwen chat 模板的数据上训。TRL 只对
list-of-messages 的 prompt 套模板，对裸字符串原样 tokenize —— 喂裸串会让 GRPO
在和 warm-start / 评测都不一致的分布上优化，把人设训歪。

GRPO 不需要 expected_reply（它靠 reward 函数引导）。
也不需要 expected_keywords（R1 走 LLM-as-judge，不需要关键词清单）。

scenario_keywords 仅用于触发 reward 函数里的"紧急-短回复罚"——
通过 grep 紧急词从 user_input 自动提取。

用法：
    python scripts/build_grpo_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Windows 控制台默认 cp936/cp932，print 中文会 UnicodeEncodeError。
# 文件写出本身已显式 encoding="utf-8"，这里只修 std 流让统计打印不崩。
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = PROJECT_ROOT / "evals" / "cases.json"
SEEDS_PATH = PROJECT_ROOT / "data" / "sft_seeds.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "grpo_train.jsonl"

# 复用 SFT 训练用的同款 system 人设（单一真源），保证 GRPO 输入分布与 SFT warm-start 一致
sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_for_qwen import SFT_SYSTEM_PROMPT  # noqa: E402

# 与 grpo_reward.py 里 URGENT_KEYWORDS 保持一致
URGENT_KEYWORDS = (
    "被骗", "刚转", "刚刚被", "立刻被", "已经转",
    "急", "救救", "怎么办", "马上", "今天",
)


def extract_scenario_keywords(user_input: str) -> list[str]:
    """从用户输入里 grep 出触发紧急罚的关键词。"""
    return [kw for kw in URGENT_KEYWORDS if kw in user_input]


def make_row(user_input: str, source: str) -> dict:
    """构造对话式训练行：注入 SFT 同款 system，user 放真实问题。

    TRL GRPOTrainer 对 list-of-messages 的 prompt 会自动 apply chat template
    （加 system / <|im_start|> 脚手架 / generation prompt），与 SFT warm-start
    输入分布一致。
    """
    return {
        "prompt": [
            {"role": "system", "content": SFT_SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
        "scenario_keywords": extract_scenario_keywords(user_input),
        "_source": source,
    }


def main() -> None:
    rows: list[dict] = []

    # 1. 从 cases.json 提取 15 条 prompt
    with open(CASES_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)
    for c in cases:
        prompt = c.get("user_input", "").strip()
        if not prompt:
            continue
        rows.append(make_row(prompt, f"cases.json::{c.get('id', '?')}"))

    # 2. 从 sft_seeds.json 提取 50 条 user_input
    with open(SEEDS_PATH, "r", encoding="utf-8") as f:
        seeds = json.load(f)
    for s in seeds:
        prompt = s.get("user_input", "").strip()
        if not prompt:
            continue
        rows.append(make_row(prompt, f"sft_seeds.json::{s.get('id', '?')}"))

    # 去重（cases 里的某些场景可能跟 seeds 重复）。
    # prompt 现在是 list（unhashable），按 user 文本去重。
    seen = set()
    deduped = []
    for r in rows:
        key = r["prompt"][-1]["content"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    # 写出
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_emergency = sum(1 for r in deduped if r["scenario_keywords"])
    print(f"写出 {len(deduped)} 条到 {OUTPUT_PATH}")
    print(f"  来自 cases.json: {sum(1 for r in deduped if r['_source'].startswith('cases'))}")
    print(f"  来自 sft_seeds.json: {sum(1 for r in deduped if r['_source'].startswith('sft_seeds'))}")
    print(f"  含紧急关键词的 prompt: {n_emergency}")
    print(f"  示例（前 3 条）:")
    for r in deduped[:3]:
        print(f"    [{r['_source']}] {r['prompt'][-1]['content'][:50]}...")
        if r["scenario_keywords"]:
            print(f"      → urgent: {r['scenario_keywords']}")


if __name__ == "__main__":
    main()
