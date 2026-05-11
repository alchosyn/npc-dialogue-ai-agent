# 信噪 · NPC Dialogue AI Agent

一个中文反诈与安全意识对话 Agent。信噪——23 岁、贫民窟出身的黑客女顾问——用红队视角教普通人识别诈骗、做好个人数字防护。

DeepSeek + RAG + 5 工具 + 端到端 observability + 双轨 eval（规则匹配 + LLM-as-judge）。

## 它能做什么

- **粘条可疑短信进来 → 拆解话术 + 风险打分 + 引真实案例 + 给 3 步行动建议**
- 回答"我妈接到电话说医保卡涉嫌洗钱怎么办" / "我密码用 password123 安全吗" / "刚被骗 5 万怎么止付"
- 所有具体建议都从内部知识库或权威源取证，绝不编造案号、法条号、报警电话

## 技术栈

| 层 | 选型 |
|---|---|
| 模型 | DeepSeek (`deepseek-chat`，OpenAI 兼容接口) |
| RAG | sentence-transformers `paraphrase-multilingual-MiniLM-L12-v2` + 余弦相似度 |
| 工具 | 5 个：web_search (Tavily) / search_knowledge / risk_score / calculator / get_current_time |
| 观测 | 每步 trace JSON（token / latency / 工具调用） |
| 评估 | 双轨：规则匹配（关键词 + 工具调用断言） + LLM-as-judge（4 维度评分 + spearman ρ 一致性） |
| 部署 | CLI (`main.py`) + Colab notebook |

## 工具设计

| 工具 | 类型 | 价值 |
|---|---|---|
| `risk_score` | 确定性规则引擎（12 类信号 + 正则） | 演示工具不必只是 LLM 转发，可以是稳定的离线逻辑 |
| `search_knowledge` | 向量检索本地知识库 | RAG 主入口，55 条权威条目 |
| `web_search` | Tavily | 拿最新预警 / 案例 |
| `calculator` | 表达式求值 | 密码熵估算、损失计算 |
| `get_current_time` | 时间 | 案例时效判断 |

## 知识库

55 条，5 层结构，全部来源于公开权威源：

| 层 | 数量 | 内容 |
|---|---|---|
| `law-*` 法律基线 | 5 | 反电诈法 / 个人信息保护法 / 刑法 266 / 253 之一 |
| `pattern-*` 手法图谱 | 15 | 国家反诈中心 12 大类 + AI 换脸 / 虚拟币 / SIM Swap |
| `case-*` 真实案例 | 15 | 月度通报 + 媒体复核报道，含金额 / 地区 / 时间 |
| `protect-*` 防护手册 | 10 | 密码 / 2FA / 隐私 / 可疑信号识别 |
| `decision-*` 处置流程 | 10 | 96110 / 110 / 12321 决策树 + 银行止付 |

**数据来源（按权重）**：

- T1 国家级反诈权威（50%）：国家反诈中心、公安部刑侦局、国务院反诈办
- T2 技术与漏洞权威（15%）：CNCERT/CC、CNVD、网信办
- T3 监管与举报数据（10%）：12321、中国互联网协会反诈骗联盟
- T4 法律法规（10%）：反电诈法、个人信息保护法、网络安全法
- T5 技术标准 + 厂商情报（15%）：NIST SP 800-63B、OWASP Top 10、各 SRC

故意排除：自媒体反诈博主、B 站视频、知乎个人帖。

## Demo 脚本

**Demo A —— 仿冒银行短信识别**
```
你：我刚收到一条短信：【工商银行】您的账户存在风险，请立即登录
   http://icbc-secure.cn.vip 验证身份，否则 24 小时内冻结。

信噪：（调 risk_score → search_knowledge）
     钓鱼短信。两个破绽：1）icbc-secure.cn.vip 这域名是仿冒，
     工行只用 icbc.com.cn；2）真银行不会让你点链接验身份。
     国家反诈中心的「pattern-004 冒充电商物流客服」属同类套路。
     三件事：别点链接 / 工行客服只走 95588 / 已经点了立即拨 96110。
     这条短信整条转发到 12321 举报。
```

**Demo B —— 老年人接到"医保卡被盗用"电话**
```
你：我妈接到电话说医保卡被人异地盗用要她转账到安全账户怎么办？

信噪：（调 search_knowledge → 引 case-002）
     典型冒充公检法。卡点：警方、医保局永远不会让你转账到任何账户，
     更没有「安全账户」这种东西。让她做这三件：
     1）立刻挂断  2）用她自己手机拨当地医保局官方电话核实
     3）拨 96110。这套路是对老年人单笔损失最高的一类，去年通报过 80 万的案例。
```

**Demo C —— 个人安全卫生体检**
```
你：我所有账号都用 password123，安全吗？

信噪：（调 search_knowledge → protect-001）
     按 NIST SP 800-63B 标准这密码连暴力破解都不需要，字典攻击秒过。
     三件事：
     1）装个密码管理器（1Password、Bitwarden 都行），每个账号独立密码
     2）银行、邮箱、社交主账号开 Authenticator 2FA（别用短信 2FA）
     3）查一下 haveibeenpwned，看你邮箱有没有泄露过。
```

## 企业需求映射（B2B 落地想象）

| 客户 | 痛点 | 信噪能解决的部分 |
|---|---|---|
| 银行 / 运营商 | 老年客户被骗后投诉率高，监管压力 | 内嵌进 App 客服 / 短信预警 / 主动外呼前置教育 |
| 企业 IT / 安全部门 | 员工钓鱼邮件点击率高，培训内容老旧 | 持续生成新型钓鱼模板 + 个性化培训对话 |
| 政府反诈中心 | 案例库丰富但触达难 | 以对话形式分发知识，比图文宣传转化高 |

对标公司：**KnowBe4（Vista 46亿美元收购）、Hoxhunt、国家反诈中心 App、Singapore ScamShield**。注意：这是 Security Awareness / Anti-Fraud 赛道，不是 Microsoft Security Copilot 那种 SecOps Copilot。

## 快速开始

### 本地 CLI

```bash
git clone https://github.com/alchosyn/npc-dialogue-ai-agent.git
cd npc-dialogue-ai-agent
pip install -r requirements.txt

export DEEPSEEK_API_KEY=sk-...
export TAVILY_API_KEY=tvly-...

python main.py
```

### Colab

打开 `NPC_agent.ipynb`，在 Colab Secrets 里设好 `DEEPSEEK_API_KEY` 和 `TAVILY_API_KEY`，按顺序跑三个 cell。

## 跑评估

```bash
# 规则匹配（无需额外配置）
python evals/run_eval.py

# 叠加 LLM-as-judge（用 DeepSeek 当 judge，调用约 16 次）
python evals/run_eval.py --judge

# 只跑前 3 个 case（调试）
python evals/run_eval.py --limit 3
```

输出：
- `evals/results.json` —— 完整结果（机器可读）
- `evals/report.md` —— 人类可读报告，含规则通过率、judge 平均分、spearman ρ 一致性、逐 case trace

## 项目结构

```
src/npc_agent/
  agent.py         ReAct 主循环
  llm_client.py    DeepSeek 客户端
  config.py        路径、模型、阈值
  utils.py         clean_reply
  memory.py        SYSTEM_PROMPT + 摘要存档
  tracing.py       trace 记录
  tools/
    registry.py    工具注册表（5 个）
    web_search.py  Tavily
    knowledge.py   sentence-transformers RAG
    risk_score.py  确定性规则评分器（无 LLM）
    calculator.py
    time_tool.py

evals/
  cases.json       15 个测试场景
  run_eval.py      规则匹配 + 调度
  judge.py         LLM-as-judge + spearman ρ
  report.md        最新评估报告
  results.json     完整原始结果

scripts/
  inspect_traces.py     看最新 trace
  list_traces.py        列出全部 trace
  clear_history.py      清对话存档
  clean_notebook.py     修复 ipynb widget metadata

knowledge_base.json     55 条反诈知识
main.py                 本地命令行入口
NPC_agent.ipynb         Colab notebook（3 个 cell）
```

## Roadmap

1. 接入国家反诈中心 RSS / 公开 API 做实时知识更新
2. 加 OCR：直接传短信截图
3. 多 Agent：信噪 + 一个"法务助理 Agent"分工
4. 部署 hosted demo（Hugging Face Spaces / Vercel）
5. 写技术拆解博文
6. 双语（英文版可对标 Hoxhunt / KnowBe4 内容）

## 已知局限

- **不替代专业律师 / 警方 / 心理咨询**。涉及人身安全、大额损失、跨境纠纷请走正规渠道。
- 知识库是 MVP 静态库，案例时效有限。生产化应接入 RSS 月更或实时 API。
- LLM-as-judge 用同模型自评有偏，更严的 setup 应换 GPT-4 / Claude 当 judge。
- risk_score 是确定性规则，对绕过规则的话术（如黑话、谐音）漏检；属于 MVP 妥协。

## 延伸阅读

- 国家反诈中心：https://www.gjfzzx.com/
- CNCERT：https://www.cert.org.cn/
- 12321 举报中心：https://www.12321.cn/
- NIST SP 800-63B：https://pages.nist.gov/800-63-3/sp800-63b.html
- Kevin Mitnick《The Art of Deception》（社工经典，思路参考）

## License

MIT
