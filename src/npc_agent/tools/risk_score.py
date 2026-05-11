"""确定性规则评分器：给一段可疑文本（短信/邮件/通话脚本）打 0-100 分。

这是一个故意做成 "纯规则、无 LLM" 的工具，价值在于：
1. 演示工具不必只是 LLM 转发器，可以是确定性逻辑
2. 给 agent 一个稳定的"信号清单"输出，便于后续案例匹配
3. 评分不依赖外部 API，离线可跑、可测、可复现
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Signal:
    name: str
    weight: int
    pattern: re.Pattern[str]


# 规则库：每条 (信号名, 加分, 匹配正则)
# 权重设计原则：单条触发 ≥ 70 分应对应"几乎一定是诈骗"的强信号
_SIGNALS: tuple[Signal, ...] = (
    Signal("索要短信验证码或银行卡密码", 30, re.compile(
        r"(短信验证码|动态码|银行卡密码|U盾|网银密码|交易密码|支付密码)"
    )),
    Signal("引导转账到所谓\"安全账户\"", 30, re.compile(
        r"(安全账户|保护账户|监管账户|资金清查账户|公检法账户)"
    )),
    Signal("自称公检法/客服/银行/医保等权威身份", 20, re.compile(
        r"(公安|警官|检察|法院|法官|银行经理|银行客服|官方客服|医保中心|社保局|税务局|海关)"
    )),
    Signal("可疑链接（短链/小众 TLD）", 25, re.compile(
        r"(t\.cn/|dwz\.cn/|suo\.im/|url\.cn/|w\.url\.cn|7toutiao\."
        r"|https?://[^\s]+\.(?:top|vip|xyz|click|link|info|cf|tk|ml|ga|live|site|online|store|shop)\b)"
    )),
    Signal("文本含外部链接（需独立核实）", 10, re.compile(
        r"https?://\S+"
    )),
    Signal("紧迫性话术（限时/立即/否则）", 15, re.compile(
        r"(立即|马上|尽快|限时|今天内|24小时内|否则|逾期|失效|冻结|封禁|拘留|追究)"
    )),
    Signal("索要银行卡号/身份证号/人脸识别", 25, re.compile(
        r"(身份证号|银行卡号|开户行|预留手机号|人脸识别|刷脸|手持身份证)"
    )),
    Signal("未知收款账户或扫码付款", 20, re.compile(
        r"(扫描下方二维码|扫码付款|私人账户|代收账户|境外账户|对公账户)"
    )),
    Signal("中奖/退款/补贴诱饵", 15, re.compile(
        r"(中奖|大奖|退税|退款|补贴|福利金|低保|疫情补贴|个税退还)"
    )),
    Signal("视频通话+索要资金", 25, re.compile(
        r"(视频通话.{0,20}(?:转账|汇款|打钱)"
        r"|视频会议.{0,20}(?:转账|汇款|打钱)"
        r"|开视频.{0,20}(?:转账|汇款|打钱))"
    )),
    Signal("贷款/刷单/兼职高回报", 15, re.compile(
        r"(无抵押贷款|秒批贷款|刷单返利|做任务返佣|日入\d+|轻松月入|动动手指)"
    )),
    Signal("涉及亲友身份冒充求助", 15, re.compile(
        r"(我是你(?:儿子|女儿|爸|妈|领导|老板).{0,30}(?:钱|转|汇|借))"
    )),
    Signal("AI 换脸/合成音视频特征描述", 20, re.compile(
        r"(换脸|AI合成|声音克隆|deepfake|视频是假的)"
    )),
)


def _matched_signals(scenario: str) -> list[Signal]:
    return [s for s in _SIGNALS if s.pattern.search(scenario)]


def _suggested_action(score: int, signals: Iterable[Signal]) -> str:
    sig_names = {s.name for s in signals}
    if score >= 70:
        base = "几乎可以确认是诈骗。立即停止任何回复或操作，不要点击链接、不要回拨号码。"
        if "引导转账到所谓\"安全账户\"" in sig_names or "自称公检法/客服/银行/医保等权威身份" in sig_names:
            base += "拨打 96110（反诈中心）核实，必要时报 110。"
        else:
            base += "可拨打 96110 咨询，短信类可转发到 12321 举报。"
        return base
    if score >= 40:
        return (
            "高度可疑。先别动作，用官方渠道独立核实（如银行官方 App、客服总机），"
            "不要使用对方提供的任何链接或电话。可拨 96110 咨询。"
        )
    if score >= 15:
        return "存在可疑信号，建议保持警惕、不要泄露任何个人信息或验证码，必要时拨 96110 咨询。"
    return "未检测到明显诈骗信号，但仍建议谨慎处理陌生信息。"


def risk_score(scenario: str) -> dict:
    """对一段可疑文本打分。

    Args:
        scenario: 可疑短信/邮件/通话脚本文本

    Returns:
        {
            "score": 0-100,
            "signals": [触发的信号名列表],
            "suggested_action": 处置建议
        }
    """
    if not scenario or not scenario.strip():
        return {
            "score": 0,
            "signals": [],
            "suggested_action": "请提供具体的可疑文本（短信、邮件、通话内容等）",
        }

    matched = _matched_signals(scenario)
    score = min(100, sum(s.weight for s in matched))
    return {
        "score": score,
        "signals": [s.name for s in matched],
        "suggested_action": _suggested_action(score, matched),
    }
