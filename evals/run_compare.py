"""四路对比评估：DeepSeek base / DeepSeek+Agent / Qwen base / Qwen+LoRA。

每路对所有 case 跑一次，用 LLM-as-judge 给所有结果打 4 维度分（accuracy /
actionability / citation / tone），输出对比报告。

用法：
    # 只跑 DeepSeek 两路（本地无 GPU 时）
    python evals/run_compare.py --strategies deepseek-base deepseek-agent

    # 在 Kaggle 上跑全四路（需 GPU + adapter 路径）
    python evals/run_compare.py --strategies all \\
        --qwen-base-name Qwen/Qwen2.5-1.5B-Instruct \\
        --qwen-lora-path /kaggle/working/qwen-1.5b-xinzao-lora

    # 限制 case 数量调试
    python evals/run_compare.py --strategies deepseek-base --limit 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "evals"))

CASES_PATH = PROJECT_ROOT / "evals" / "cases.json"
REPORT_PATH = PROJECT_ROOT / "evals" / "compare_report.md"
RESULTS_PATH = PROJECT_ROOT / "evals" / "compare_results.json"


# ─── 策略 1: DeepSeek base (无 RAG，纯 LLM) ─────────────

_SIMPLE_SYSTEM = (
    "你是信噪，反诈和安全意识顾问。"
    "回答用户的安全问题时给出可执行的 3 步以内行动建议。"
    "必要时附求助电话：96110 反诈中心、110 报警、12321 举报短信。"
    "始终使用简体中文。"
)


def infer_deepseek_base(user_input: str) -> str:
    from npc_agent.config import MODEL
    from npc_agent.llm_client import get_client

    client = get_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _SIMPLE_SYSTEM},
            {"role": "user", "content": user_input},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


# ─── 策略 2: DeepSeek + Agent (有 RAG/工具/护栏) ─────────────


def infer_deepseek_agent(user_input: str) -> str:
    from npc_agent.agent import step
    from npc_agent.memory import SYSTEM_PROMPT

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    reply, _ = step(messages, user_input)
    return reply


# ─── 策略 3-5: Qwen (base / +LoRA / +GRPO) ─────────────
#
# 注意：之前用单一全局 _qwen_model 缓存——同一进程里跑 qwen-base 然后 qwen-lora 时，
# qwen-lora 会返回 cache 里的 base 模型，根本没装 adapter。这是 silent bug，
# 让之前 4 路对比的 qwen-lora 实际等于 qwen-base。修法：用 dict cache，
# key 包含 lora_path。


_qwen_cache: dict[tuple[str, str | None], tuple] = {}


def _load_qwen(base_name: str, lora_path: str | None = None):
    """惰性加载 Qwen，按 (base, adapter) 组合缓存。"""
    key = (base_name, lora_path)
    if key in _qwen_cache:
        return _qwen_cache[key]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[qwen] loading base: {base_name} (adapter: {lora_path or 'none'})")
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if lora_path:
        from peft import PeftModel
        print(f"[qwen] loading LoRA adapter: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()  # 合并 LoRA 加速推理

    model.eval()
    _qwen_cache[key] = (model, tokenizer)
    return model, tokenizer


def _make_qwen_inferer(base_name: str, lora_path: str | None) -> Callable[[str], str]:
    def infer(user_input: str) -> str:
        import torch

        model, tokenizer = _load_qwen(base_name, lora_path)
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": _SIMPLE_SYSTEM},
                {"role": "user", "content": user_input},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=350,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_only = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(gen_only, skip_special_tokens=True).strip()

    return infer


# ─── 主流程 ─────────────


def load_cases(limit: int | None = None) -> list[dict]:
    with open(CASES_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)
    if limit:
        cases = cases[:limit]
    return cases


def run_strategy(name: str, inferer: Callable[[str], str], cases: list[dict]) -> list[dict]:
    results = []
    for i, c in enumerate(cases, 1):
        print(f"  [{name}] {i}/{len(cases)} {c['id']}")
        try:
            reply = inferer(c["user_input"])
            err = None
        except Exception as e:
            reply = ""
            err = str(e)
            print(f"    error: {e}")
        results.append({
            "case_id": c["id"],
            "strategy": name,
            "user_input": c["user_input"],
            "reply": reply,
            "error": err,
        })
    return results


def judge_all(results: list[dict], cases_by_id: dict[str, dict]) -> list[dict]:
    from judge import llm_judge  # noqa: E402

    judged = []
    for i, r in enumerate(results, 1):
        print(f"  [judge] {i}/{len(results)} {r['strategy']}/{r['case_id']}")
        if not r["reply"]:
            r["judge"] = None
            judged.append(r)
            continue
        case = cases_by_id[r["case_id"]]
        try:
            r["judge"] = llm_judge(
                {
                    "user_input": r["user_input"],
                    "expected_keywords": case.get("expected_keywords", []),
                    "scenario_type": case.get("scenario_type", ""),
                },
                r["reply"],
            )
        except Exception as e:
            print(f"    judge error: {e}")
            r["judge"] = None
        judged.append(r)
    return judged


def aggregate(all_results: dict[str, list[dict]]) -> dict[str, dict]:
    """每个 strategy 计算平均 overall + 4 维度平均。"""
    summary = {}
    for strategy, results in all_results.items():
        judged = [r for r in results if r.get("judge")]
        if not judged:
            summary[strategy] = {"n": 0, "overall": None}
            continue
        overall = statistics.mean(r["judge"]["overall"] for r in judged)
        dims = {}
        for dim in ("accuracy", "actionability", "citation", "tone"):
            dims[dim] = statistics.mean(r["judge"]["scores"][dim] for r in judged)
        summary[strategy] = {
            "n": len(judged),
            "overall": round(overall, 2),
            **{k: round(v, 2) for k, v in dims.items()},
        }
    return summary


def write_report(summary: dict, all_results: dict[str, list[dict]]) -> None:
    strategies = list(summary.keys())

    lines = [
        "# 信噪 Agent · 四路对比评估",
        "",
        "## 总体得分（LLM-as-Judge，5 分制）",
        "",
        "| 策略 | n | overall | accuracy | actionability | citation | tone |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in strategies:
        st = summary[s]
        if st["overall"] is None:
            lines.append(f"| {s} | {st['n']} | - | - | - | - | - |")
        else:
            lines.append(
                f"| {s} | {st['n']} | **{st['overall']}** | "
                f"{st['accuracy']} | {st['actionability']} | {st['citation']} | {st['tone']} |"
            )
    lines.append("")

    # 加分析段落
    if "qwen-base" in summary and "qwen-lora" in summary:
        qb = summary["qwen-base"]
        ql = summary["qwen-lora"]
        if qb["overall"] and ql["overall"]:
            delta = ql["overall"] - qb["overall"]
            sign = "+" if delta >= 0 else ""
            lines.extend([
                f"## LoRA 微调效果",
                f"",
                f"Qwen2.5-1.5B 微调后 overall {qb['overall']} → {ql['overall']} ({sign}{delta:.2f})",
                f"",
            ])

    if "deepseek-agent" in summary and "qwen-lora" in summary:
        da = summary["deepseek-agent"]
        ql = summary["qwen-lora"]
        if da["overall"] and ql["overall"]:
            gap = da["overall"] - ql["overall"]
            lines.extend([
                f"## 小模型逼近大模型程度",
                f"",
                f"Qwen2.5-1.5B+LoRA 离 DeepSeek+Agent 还差 {gap:.2f} 分（{ql['overall']} vs {da['overall']}）",
                f"",
            ])

    # 每个 case 的并列对比
    lines.extend(["---", "", "## 逐 case 对比", ""])
    case_ids = sorted({r["case_id"] for results in all_results.values() for r in results})
    for cid in case_ids:
        lines.append(f"### `{cid}`")
        lines.append("")
        for s in strategies:
            row = next((r for r in all_results.get(s, []) if r["case_id"] == cid), None)
            if not row:
                continue
            j = row.get("judge")
            score_str = f"{j['overall']}/5" if j else "(no judge)"
            lines.append(f"**{s}** [{score_str}]:")
            lines.append(f"> {row['reply'][:300]}{'...' if len(row['reply']) > 300 else ''}")
            lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告写入: {REPORT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["deepseek-base", "deepseek-agent"],
        choices=["deepseek-base", "deepseek-agent", "qwen-base", "qwen-lora", "qwen-grpo", "all"],
        help="跑哪几路。'all' 包括 5 路（含 qwen-grpo）",
    )
    parser.add_argument("--qwen-base-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--qwen-lora-path", default=None, help="SFT LoRA adapter 目录")
    parser.add_argument("--qwen-grpo-path", default=None, help="GRPO 后的 adapter 目录")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-judge", action="store_true", help="跳过 LLM-as-judge")
    args = parser.parse_args()

    strategies = args.strategies
    if "all" in strategies:
        strategies = ["deepseek-base", "deepseek-agent", "qwen-base", "qwen-lora", "qwen-grpo"]

    # 注册策略 inferers
    inferers: dict[str, Callable[[str], str]] = {}
    if "deepseek-base" in strategies:
        inferers["deepseek-base"] = infer_deepseek_base
    if "deepseek-agent" in strategies:
        inferers["deepseek-agent"] = infer_deepseek_agent
    if "qwen-base" in strategies:
        inferers["qwen-base"] = _make_qwen_inferer(args.qwen_base_name, lora_path=None)
    if "qwen-lora" in strategies:
        if not args.qwen_lora_path:
            print("ERROR: --qwen-lora-path 必填（指向训练好的 SFT adapter 目录）")
            sys.exit(1)
        inferers["qwen-lora"] = _make_qwen_inferer(args.qwen_base_name, lora_path=args.qwen_lora_path)
    if "qwen-grpo" in strategies:
        if not args.qwen_grpo_path:
            # all 模式下如果没传 grpo path，跳过这一路而不是崩
            if "all" in args.strategies:
                print("[skip] --qwen-grpo-path 未提供，跳过 qwen-grpo 策略")
            else:
                print("ERROR: --qwen-grpo-path 必填（指向 GRPO 后 adapter 目录）")
                sys.exit(1)
        else:
            inferers["qwen-grpo"] = _make_qwen_inferer(args.qwen_base_name, lora_path=args.qwen_grpo_path)

    cases = load_cases(args.limit)
    cases_by_id = {c["id"]: c for c in cases}
    print(f"加载 {len(cases)} 个 case，跑 {len(inferers)} 个策略\n")

    all_results: dict[str, list[dict]] = {}
    for name, inferer in inferers.items():
        print(f"=== 跑 {name} ===")
        all_results[name] = run_strategy(name, inferer, cases)

    if not args.no_judge:
        print(f"\n=== LLM-as-judge 评分 ===")
        for name in all_results:
            all_results[name] = judge_all(all_results[name], cases_by_id)

    summary = aggregate(all_results)
    print("\n=== 总结 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    RESULTS_PATH.write_text(
        json.dumps({"summary": summary, "details": all_results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(summary, all_results)


if __name__ == "__main__":
    main()
