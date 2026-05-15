"""把 evals/cases.json 的 15 条 + data/sft_seeds.json 的 50 条合并成 GRPO 训练数据。

输出 data/grpo_train.jsonl，每行：
    {"prompt": "...", "scenario_keywords": ["被骗", "刚转"]}

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = PROJECT_ROOT / "evals" / "cases.json"
SEEDS_PATH = PROJECT_ROOT / "data" / "sft_seeds.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "grpo_train.jsonl"

# 与 grpo_reward.py 里 URGENT_KEYWORDS 保持一致
URGENT_KEYWORDS = (
    "被骗", "刚转", "刚刚被", "立刻被", "已经转",
    "急", "救救", "怎么办", "马上", "今天",
)


def extract_scenario_keywords(user_input: str) -> list[str]:
    """从用户输入里 grep 出触发紧急罚的关键词。"""
    return [kw for kw in URGENT_KEYWORDS if kw in user_input]


def main() -> None:
    rows: list[dict] = []

    # 1. 从 cases.json 提取 15 条 prompt
    with open(CASES_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)
    for c in cases:
        prompt = c.get("user_input", "").strip()
        if not prompt:
            continue
        rows.append({
            "prompt": prompt,
            "scenario_keywords": extract_scenario_keywords(prompt),
            "_source": f"cases.json::{c.get('id', '?')}",
        })

    # 2. 从 sft_seeds.json 提取 50 条 user_input
    with open(SEEDS_PATH, "r", encoding="utf-8") as f:
        seeds = json.load(f)
    for s in seeds:
        prompt = s.get("user_input", "").strip()
        if not prompt:
            continue
        rows.append({
            "prompt": prompt,
            "scenario_keywords": extract_scenario_keywords(prompt),
            "_source": f"sft_seeds.json::{s.get('id', '?')}",
        })

    # 去重（cases 里的某些场景可能跟 seeds 重复）
    seen = set()
    deduped = []
    for r in rows:
        if r["prompt"] in seen:
            continue
        seen.add(r["prompt"])
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
        print(f"    [{r['_source']}] {r['prompt'][:50]}...")
        if r["scenario_keywords"]:
            print(f"      → urgent: {r['scenario_keywords']}")


if __name__ == "__main__":
    main()
