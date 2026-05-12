"""把 data/sft_expanded.jsonl 转成 Qwen2.5 SFT 训练用的 messages 格式，做 90/10 train/val 切分。

Qwen2.5-Instruct 期望的格式（HuggingFace SFTTrainer 标准）：
    {"messages": [{"role": "system", "content": "..."},
                  {"role": "user", "content": "..."},
                  {"role": "assistant", "content": "..."}]}

输出：
    data/sft_train.jsonl
    data/sft_val.jsonl

用法：
    python scripts/format_for_qwen.py                  # 默认 90/10 切分
    python scripts/format_for_qwen.py --val-ratio 0.15
    python scripts/format_for_qwen.py --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPANDED_PATH = PROJECT_ROOT / "data" / "sft_expanded.jsonl"
TRAIN_PATH = PROJECT_ROOT / "data" / "sft_train.jsonl"
VAL_PATH = PROJECT_ROOT / "data" / "sft_val.jsonl"


# 训练用的简化 SYSTEM_PROMPT
# 关键区别：
# 1. 删掉所有「先调 search_knowledge」「先调 risk_score」等工具规则（1.5B 不调工具）
# 2. 保留人设 + 反诈顾问职业框架
# 3. 强调「直接回答 + 三步以内行动 + 引用机构名」
SFT_SYSTEM_PROMPT = (
    "你是信噪，23 岁，贫民窟出身、自学成才的黑客，现在做反诈和安全意识顾问。"
    "用户来找你聊大多带具体安全问题：可疑短信、被骗后处置、密码该怎么设、隐私怎么保。"
    "你的回答规则："
    "1. 语气锋利但不油腻——你以前最讨厌别人看透你，现在最讨厌自己说虚的；"
    "2. 平常用短句，回答围绕「这是什么 → 怎么做 → 为什么」三段；"
    "3. 给行动建议时尽量给 3 步以内可立即执行的动作；"
    "4. 必要时附求助电话：96110 反诈中心、110 报警、12321 举报短信骚扰；"
    "5. 引用权威信息源时标明机构名（国家反诈中心、CNCERT、央行征信中心等）；"
    "6. 绝不编造案例编号、案件金额、法条号码——不确定就说不确定；"
    "7. 不用动作神情描写，直接输出对话；不说「先告诉你」「简单来说」「总结一下」这类引导语；"
    "8. 始终使用简体中文。"
)


def to_messages_format(row: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SFT_SYSTEM_PROMPT},
            {"role": "user", "content": row["user_input"]},
            {"role": "assistant", "content": row["expected_reply"]},
        ],
        # 元信息（训练时被忽略，但保留便于追溯）
        "_meta": {
            "source_id": row.get("source_id", ""),
            "category": row.get("category", ""),
            "kb_grounding": row.get("kb_grounding", []),
            "is_variant": row.get("is_variant", False),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    rows: list[dict] = []
    with open(EXPANDED_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # 按 source_id 前缀（seed-XXX）分组，确保同一组种子的变体不跨 train/val
    # 否则会有数据泄露：val 里的变体的 answer 在 train 里已经见过
    grouped: dict[str, list[dict]] = {}
    for r in rows:
        sid = r.get("source_id", "")
        key = sid.split("-v")[0] if "-v" in sid else sid
        grouped.setdefault(key, []).append(r)

    keys = sorted(grouped.keys())
    rng = random.Random(args.seed)
    rng.shuffle(keys)

    n_val = max(1, int(len(keys) * args.val_ratio))
    val_keys = set(keys[:n_val])

    train_rows, val_rows = [], []
    for k, group in grouped.items():
        target = val_rows if k in val_keys else train_rows
        target.extend(group)

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(to_messages_format(r), ensure_ascii=False) + "\n")
    with open(VAL_PATH, "w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(to_messages_format(r), ensure_ascii=False) + "\n")

    print(f"训练集: {len(train_rows)} 条 → {TRAIN_PATH}")
    print(f"验证集: {len(val_rows)} 条 → {VAL_PATH}")
    print(f"分组依据: {len(grouped)} 个 source_id 组（同一种子的所有变体在同一边）")
    print(f"train/val ratio: {len(train_rows) / max(1, len(train_rows) + len(val_rows)):.2%}")


if __name__ == "__main__":
    main()
