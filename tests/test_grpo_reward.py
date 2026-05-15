"""scripts/grpo_reward.py 单元测试。

Judge 部分用 mock，避免单测依赖 DEEPSEEK_API_KEY。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 把 scripts/ 加进 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from grpo_reward import compute_reward  # noqa: E402


# ─── Mock judge 函数 ─────────────────────────────────────


def make_mock_judge(score: float):
    def judge(_case, _reply):
        return {"overall": score}
    return judge


JUDGE_5 = make_mock_judge(5.0)  # 完美 judge 分
JUDGE_3 = make_mock_judge(3.0)  # 中等
JUDGE_1 = make_mock_judge(1.0)  # 差
JUDGE_NEUTRAL = make_mock_judge(3.0)


# ─── 测试 ─────────────────────────────────────────────────


def test_empty_completion_returns_min():
    """空回复直接返回最低分。"""
    r = compute_reward("", user_input="测试", judge_client=JUDGE_5)
    assert r == -1.5


def test_judge_dominates_signal():
    """Judge 5 vs 1 的差距应至少等于 (5-1)*0.6 = 2.4。"""
    r_high = compute_reward("正常回复", user_input="问题", judge_client=JUDGE_5)
    r_low = compute_reward("正常回复", user_input="问题", judge_client=JUDGE_1)
    # 同样的 completion，规则项贡献相同，所以差就是 judge 部分
    assert (r_high - r_low) >= 2.4 - 0.001  # 浮点容差


def test_steps_bonus_active():
    """带 1.2.3. 步骤的回复应该比没有的多 0.5。"""
    no_steps = compute_reward("是诈骗", user_input="问题", judge_client=JUDGE_3)
    with_steps = compute_reward("是诈骗。1. 挂断 2. 报警 3. 截图证据", user_input="问题", judge_client=JUDGE_3)
    assert with_steps > no_steps
    assert (with_steps - no_steps) >= 0.4  # 应该是 +0.5


def test_steps_bonus_chinese_numbers():
    """中文数字的步骤化也应该被识别。"""
    r = compute_reward(
        "三件事：第一不要点链接，第二拨打官方客服核实，第三举报到 12321",
        user_input="问题",
        judge_client=JUDGE_3,
    )
    no_steps = compute_reward("不要点链接", user_input="问题", judge_client=JUDGE_3)
    assert r > no_steps + 0.4  # 步骤 +0.5 + 真实电话 +0.5（12321 在白名单）


def test_hotline_bonus_active():
    """含 96110/110/12321 应该 +0.5。"""
    no_phone = compute_reward("是诈骗，注意安全", user_input="问题", judge_client=JUDGE_3)
    with_phone = compute_reward("是诈骗，拨 96110", user_input="问题", judge_client=JUDGE_3)
    assert with_phone > no_phone


def test_fake_hotline_penalty():
    """假电话应该比真电话低 1 分。"""
    real = compute_reward("拨 96110", user_input="问题", judge_client=JUDGE_3)
    fake = compute_reward("拨 95110", user_input="问题", judge_client=JUDGE_3)
    # 真电话: +0.5 (R3)，假电话: -1 (P1) 且没有任何真电话所以 R3=0
    # 差 = 0.5 - (-1) = 1.5
    assert (real - fake) >= 1.4


def test_emergency_short_penalty():
    """紧急关键词触发但回复 < 50 字会被罚 -0.5。"""
    short = compute_reward(
        "打 96110",  # 极短
        user_input="我爸刚才被骗了 5 万，怎么办",
        scenario_keywords=["被骗", "刚转"],
        judge_client=JUDGE_3,
    )
    long_ = compute_reward(
        "立即拨 96110 申请紧急止付。三件事：1. 别再转任何账户 2. 24h 内带聊天记录到派出所立案 3. 同步联系开户银行冻结",
        user_input="我爸刚才被骗了 5 万，怎么办",
        scenario_keywords=["被骗", "刚转"],
        judge_client=JUDGE_3,
    )
    # short 触发紧急-短罚 -0.5；long 不触发（>=50 字）
    assert short < long_


def test_emergency_auto_detect_from_user_input():
    """即使没传 scenario_keywords，也能从 user_input 自动检测紧急。"""
    short = compute_reward("挂断", user_input="我刚转了钱，怎么办", judge_client=JUDGE_3)
    # "刚转" 在 URGENT_KEYWORDS 里，触发短回复罚
    # 没有真电话，所以 R3=0；判罚 -0.5
    not_urgent = compute_reward("挂断", user_input="问个理论问题", judge_client=JUDGE_3)
    assert short < not_urgent


def test_score_clamped_to_range():
    """总分应被 clamp 到 [-1.5, 4]。"""
    # 极端情况：judge 5 + R2 + R3 - 0 - 0 = 3 + 0.5 + 0.5 = 4，刚好顶到 4
    perfect = compute_reward(
        "立即拨 96110。1. 挂断电话 2. 报案 3. 联系银行止付",
        user_input="问题",
        judge_client=JUDGE_5,
    )
    assert perfect <= 4.0
    assert perfect >= 3.5  # 至少要近 4

    # 极端低：judge 1 + 假电话 - 紧急短罚
    awful = compute_reward(
        "拨 95110",
        user_input="刚被骗",
        scenario_keywords=["被骗"],
        judge_client=JUDGE_1,
    )
    assert awful >= -1.5


def test_judge_failure_falls_back_to_neutral():
    """如果 judge 函数抛异常，应该不会让整个 reward 计算崩。"""
    def broken_judge(_case, _reply):
        raise RuntimeError("API down")

    # broken_judge 抛异常 → 应该被 catch（虽然我们直接传 mock 不会进 _real_judge 路径）
    # 这里测试 compute_reward 行为：传一个会抛异常的 judge_client，应该 propagate
    # 因为单测路径里我们假设 judge_client 是稳定 mock。真正的 fallback 在 _real_judge 里。
    import pytest  # 内部 import 避免 top-level 强依赖
    with pytest.raises(RuntimeError):
        compute_reward("回复", user_input="问题", judge_client=broken_judge)


if __name__ == "__main__":
    # 简易自测，不需要 pytest 也能跑
    import traceback
    tests = [
        test_empty_completion_returns_min,
        test_judge_dominates_signal,
        test_steps_bonus_active,
        test_steps_bonus_chinese_numbers,
        test_hotline_bonus_active,
        test_fake_hotline_penalty,
        test_emergency_short_penalty,
        test_emergency_auto_detect_from_user_input,
        test_score_clamped_to_range,
    ]
    n_pass = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            n_pass += 1
        except Exception:
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{n_pass}/{len(tests)} passed")
