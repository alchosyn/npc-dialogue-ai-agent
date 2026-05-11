"""跑 evals/cases.json 里的所有场景，输出规则匹配评分（可选叠加 LLM-as-judge）。

用法：
    python evals/run_eval.py                # 只跑规则匹配
    python evals/run_eval.py --judge        # 同时跑 LLM-as-judge 并做一致性分析
    python evals/run_eval.py --limit 3      # 只跑前 3 个 case（调试用）

输出：
    evals/report.md       人类可读报告
    evals/results.json    机器可读完整结果
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "evals"))

from npc_agent.agent import step  # noqa: E402
from npc_agent.memory import SYSTEM_PROMPT  # noqa: E402

CASES_PATH = PROJECT_ROOT / "evals" / "cases.json"
REPORT_PATH = PROJECT_ROOT / "evals" / "report.md"
RESULTS_PATH = PROJECT_ROOT / "evals" / "results.json"


def load_cases() -> list[dict]:
    with open(CASES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def fresh_messages() -> list[dict]:
    """每个 case 用全新 messages，避免上下文串扰。"""
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def evaluate_rules(case: dict, reply: str, tool_calls_made: list[str]) -> dict:
    """规则匹配评分，返回 {pass, checks: {...}}."""
    checks = {}

    # 1. 工具调用 — 期望集合是 made 集合的子集
    expected = set(case.get("expected_tool_calls", []))
    made = set(tool_calls_made)
    checks["tool_calls"] = {
        "expected": sorted(expected),
        "made": sorted(made),
        "pass": expected.issubset(made),
    }

    # 2. 必须包含关键词（任一即可）
    expected_kws = case.get("expected_keywords", [])
    if expected_kws:
        hits = [kw for kw in expected_kws if kw in reply]
        checks["keywords_present"] = {
            "expected_any_of": expected_kws,
            "hits": hits,
            "pass": len(hits) >= 1,
        }
    else:
        checks["keywords_present"] = {"pass": True, "note": "no required keywords"}

    # 3. 不允许出现的禁词
    forbidden = case.get("must_not_contain", [])
    if forbidden:
        violations = [w for w in forbidden if w in reply]
        checks["no_forbidden"] = {
            "forbidden": forbidden,
            "violations": violations,
            "pass": len(violations) == 0,
        }
    else:
        checks["no_forbidden"] = {"pass": True}

    overall_pass = all(c["pass"] for c in checks.values())
    return {"pass": overall_pass, "checks": checks}


def run_one_case(case: dict) -> dict:
    """跑一个 case，返回完整结果（不含 judge）。"""
    messages = fresh_messages()
    t0 = time.time()
    try:
        reply, messages_after = step(messages, case["user_input"])
        latency_ms = int((time.time() - t0) * 1000)
        # 从 messages 里提取被调用的工具名
        tool_calls_made = []
        for m in messages_after:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    tool_calls_made.append(tc["function"]["name"])
        rule_result = evaluate_rules(case, reply, tool_calls_made)
        return {
            "id": case["id"],
            "scenario_type": case.get("scenario_type", ""),
            "user_input": case["user_input"],
            "agent_reply": reply,
            "tool_calls_made": tool_calls_made,
            "rule_result": rule_result,
            "latency_ms": latency_ms,
            "error": None,
        }
    except Exception as e:
        return {
            "id": case["id"],
            "scenario_type": case.get("scenario_type", ""),
            "user_input": case["user_input"],
            "agent_reply": "",
            "tool_calls_made": [],
            "rule_result": {"pass": False, "checks": {}},
            "latency_ms": int((time.time() - t0) * 1000),
            "error": str(e),
        }


def write_report(results: list[dict], with_judge: bool = False) -> None:
    """生成 markdown 报告。"""
    total = len(results)
    rule_passed = sum(1 for r in results if r["rule_result"]["pass"])
    rule_pass_rate = rule_passed / total if total else 0
    avg_latency = statistics.mean(r["latency_ms"] for r in results) if results else 0

    lines = [
        "# 信噪 Agent 评估报告",
        "",
        f"- **场景数**: {total}",
        f"- **规则通过率**: {rule_passed}/{total} = **{rule_pass_rate:.1%}**",
        f"- **平均单轮延迟**: {avg_latency:.0f} ms",
    ]

    if with_judge:
        judge_results = [r for r in results if r.get("judge_result")]
        if judge_results:
            avg_overall = statistics.mean(r["judge_result"]["overall"] for r in judge_results)
            dim_avgs = {}
            for dim in ["accuracy", "actionability", "citation", "tone"]:
                vals = [r["judge_result"]["scores"].get(dim, 0) for r in judge_results]
                dim_avgs[dim] = statistics.mean(vals)
            lines.extend([
                f"- **LLM-as-judge 平均分**: {avg_overall:.2f} / 5",
                f"  - accuracy: {dim_avgs['accuracy']:.2f}",
                f"  - actionability: {dim_avgs['actionability']:.2f}",
                f"  - citation: {dim_avgs['citation']:.2f}",
                f"  - tone: {dim_avgs['tone']:.2f}",
            ])
            # 一致性分析
            from judge import spearman_rho  # noqa: E402
            rule_scores = [1.0 if r["rule_result"]["pass"] else 0.0 for r in judge_results]
            judge_scores = [r["judge_result"]["overall"] for r in judge_results]
            rho = spearman_rho(rule_scores, judge_scores)
            lines.append(f"- **规则 vs Judge 评分一致性 (spearman ρ)**: {rho:.2f}")

    lines.extend(["", "---", "", "## 各 case 结果", ""])

    for r in results:
        emoji_pass = "PASS" if r["rule_result"]["pass"] else "FAIL"
        lines.extend([
            f"### [{emoji_pass}] `{r['id']}` — {r['scenario_type']}",
            "",
            f"**用户输入**：{r['user_input']}",
            "",
            f"**Agent 回复**：{r['agent_reply']}",
            "",
            f"**工具调用**：{r['tool_calls_made']}（期望 {r['rule_result']['checks'].get('tool_calls', {}).get('expected', [])}）",
            "",
            f"**延迟**: {r['latency_ms']} ms",
        ])
        if r.get("judge_result"):
            j = r["judge_result"]
            lines.extend([
                "",
                f"**Judge 评分**: overall {j['overall']:.2f} | "
                f"accuracy={j['scores']['accuracy']} actionability={j['scores']['actionability']} "
                f"citation={j['scores']['citation']} tone={j['scores']['tone']}",
                f"**Judge 评语**: {j['rationale']}",
            ])
        if r.get("error"):
            lines.extend(["", f"**错误**: `{r['error']}`"])

        # 失败的 check 详情
        if not r["rule_result"]["pass"]:
            lines.append("\n**失败原因**：")
            for name, c in r["rule_result"]["checks"].items():
                if not c["pass"]:
                    lines.append(f"- `{name}` 失败: {json.dumps(c, ensure_ascii=False)}")
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告写入: {REPORT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", action="store_true", help="启用 LLM-as-judge 评分")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 个 case")
    args = parser.parse_args()

    cases = load_cases()
    if args.limit:
        cases = cases[: args.limit]

    print(f"开始跑 {len(cases)} 个 case...")
    results = []
    for i, c in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {c['id']} — {c.get('scenario_type', '')}")
        r = run_one_case(c)
        results.append(r)
        status = "PASS" if r["rule_result"]["pass"] else "FAIL"
        print(f"    -> {status} | tools={r['tool_calls_made']} | {r['latency_ms']}ms")

    if args.judge:
        print("\n开始 LLM-as-judge 评分...")
        from judge import llm_judge  # noqa: E402

        for i, r in enumerate(results, 1):
            print(f"  [{i}/{len(results)}] judging {r['id']}")
            try:
                r["judge_result"] = llm_judge(
                    {
                        "user_input": r["user_input"],
                        "expected_keywords": next(
                            (c.get("expected_keywords", []) for c in cases if c["id"] == r["id"]), []
                        ),
                        "scenario_type": r["scenario_type"],
                    },
                    r["agent_reply"],
                )
            except Exception as e:
                print(f"    judge error: {e}")
                r["judge_result"] = None

    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(results, with_judge=args.judge)


if __name__ == "__main__":
    main()
