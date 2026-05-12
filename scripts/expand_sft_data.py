"""把 data/sft_seeds.json 里 50 条种子用 LLM 扩写成 ~200 条 SFT 训练数据。

每条种子保留原 expected_reply 作为 ground truth answer，
让 LLM 生成 4 个语义等价但表达不同的 user_input 变体。
原种子也保留进结果，所以 50 × 5 = 250 条。

用法：
    python scripts/expand_sft_data.py                    # 跑全量
    python scripts/expand_sft_data.py --variants 3       # 每条扩 3 变体
    python scripts/expand_sft_data.py --limit 5          # 调试，只扩前 5 条
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from npc_agent.config import MODEL  # noqa: E402
from npc_agent.llm_client import get_client  # noqa: E402

SEEDS_PATH = PROJECT_ROOT / "data" / "sft_seeds.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "sft_expanded.jsonl"


REWRITE_SYSTEM = """你是一个数据生成助手。下面会给你一段「原始用户提问」，请改写成 N 个语义完全等价、但表达方式不同的版本。

改写维度（每个变体至少切换 1-2 个）：
1. **长短**：原句长则缩成 1 句；原句短则扩成 2-3 句的描述
2. **语气**：正式 / 口语 / 紧急 / 困惑 / 沮丧 / 询问
3. **人设**：年轻人 / 中年 / 老年代问 / 家长帮孩子问 / 帮父母问 / 企业员工
4. **场景细节**：换具体金额、地点、平台名（但保持核心套路一致）
5. **省略 vs 详尽**：有的去掉部分细节，有的加无关背景

硬规则：
- 不能改变核心问题（同样的诈骗手法、同样的求助类型）
- 不能让答案变化——所有变体应该都能用原始答案回答
- 保持口语化中文，不要书面化
- 一行一个变体，不要编号，不要解释"""


def rewrite_one(seed: dict, n_variants: int, client) -> list[str]:
    """让 LLM 把一条种子的 user_input 改写成 n 个变体。"""
    prompt = (
        f"原始用户提问：\n{seed['user_input']}\n\n"
        f"请生成 {n_variants} 个变体。"
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.85,  # 多样性优先
    )
    raw = response.choices[0].message.content.strip()
    # 切分多行，每行可能带各种前缀（数字、点、横杠），清掉
    lines = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        # 去掉常见列表前缀
        line = re.sub(r"^[\d]+[\.\)、]\s*", "", line)
        line = re.sub(r"^[-•·]\s*", "", line)
        line = line.strip()
        if line:
            lines.append(line)
    return lines[:n_variants]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=int, default=4, help="每条种子扩多少个变体")
    parser.add_argument("--limit", type=int, default=None, help="只扩前 N 条种子（调试）")
    args = parser.parse_args()

    with open(SEEDS_PATH, "r", encoding="utf-8") as f:
        seeds = json.load(f)
    if args.limit:
        seeds = seeds[: args.limit]

    client = get_client()
    expanded: list[dict] = []

    for i, seed in enumerate(seeds, 1):
        print(f"[{i}/{len(seeds)}] 扩写 {seed['id']} ({seed['category']})...")

        # 原种子保留
        expanded.append({
            "source_id": seed["id"],
            "category": seed["category"],
            "kb_grounding": seed.get("kb_grounding", []),
            "user_input": seed["user_input"],
            "expected_reply": seed["expected_reply"],
            "is_variant": False,
        })

        # 扩写
        try:
            variants = rewrite_one(seed, args.variants, client)
            for j, v in enumerate(variants, 1):
                expanded.append({
                    "source_id": f"{seed['id']}-v{j}",
                    "category": seed["category"],
                    "kb_grounding": seed.get("kb_grounding", []),
                    "user_input": v,
                    "expected_reply": seed["expected_reply"],
                    "is_variant": True,
                })
            print(f"  -> {len(variants)} variants")
        except Exception as e:
            print(f"  -> error: {e}")

        # 节流防 rate limit
        time.sleep(0.5)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in expanded:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n共生成 {len(expanded)} 条数据 → {OUTPUT_PATH}")
    n_original = sum(1 for r in expanded if not r["is_variant"])
    n_variant = sum(1 for r in expanded if r["is_variant"])
    print(f"  原种子: {n_original}")
    print(f"  变体:   {n_variant}")


if __name__ == "__main__":
    main()
