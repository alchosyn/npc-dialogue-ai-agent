# 信噪 · Phishing Detection Agent

基于 LLM 的钓鱼攻击识别 Agent，覆盖短信 / 邮件 / 电话场景。用户贴一段可疑消息进来，Agent协助判断是不是骗局，解释攻击手法并给出行动建议。

## 架构

```
用户输入
  │
  ├─ input_guard（Prompt Injection 护栏，自动拦截）
  │
  ▼
ReAct 循环（MAX_STEPS=6）
  │
  ├─ risk_score ──→ 确定性规则评分（话术层 + URL 分析层），零延迟
  ├─ search_knowledge ──→ 向量检索本地知识库（55 条反诈条目）
  ├─ web_search ──→ Tavily 实时搜索
  ├─ calculator ──→ 密码熵估算 / 损失计算
  └─ get_current_time ──→ 案例时效判断
  │
  ▼
结构化 trace（每步记录 token / latency / 工具调用）
```

## 技术选型

| 层 | 选型 |
|---|---|
| LLM | DeepSeek（deepseek-chat，OpenAI 兼容接口） |
| Embedding | paraphrase-multilingual-MiniLM-L12-v2（384 维） |
| Web 搜索 | Tavily |
| 评估 | 规则匹配 + LLM-as-Judge（角色一致性 + 工具调用准确率） |

## risk_score 设计

三层确定性分析，纯规则不依赖 LLM：

- 话术层：13 类信号正则匹配（冒充公检法、安全账户、索要验证码等）
- URL 分析层：域名仿冒（Levenshtein 编辑距离）、IP 直连、可疑 TLD 检测 
- 
## Prompt Injection Detection
input_guard：独立护栏，检测 Prompt Injection（角色覆盖、伪造系统消息、提示词泄露探测）

## 知识库

55 条，5 层结构，来源于国家反诈中心、CNCERT、反电诈法等公开数据源：

| 层 | 数量 | 内容 |
|---|---|---|
| law | 5 | 反电诈法 / 个人信息保护法 / 刑法相关条款 |
| pattern | 15 | 12 大类诈骗手法 + AI 换脸 / 虚拟币 / SIM Swap |
| case | 15 | 真实案例（含金额 / 地区 / 时间） |
| protect | 10 | 密码 / 2FA / 隐私 / 可疑信号识别 |
| decision | 10 | 96110 / 110 / 12321 处置决策树 |

## 快速开始

```bash
git clone https://github.com/alchosyn/npc-dialogue-ai-agent.git
cd npc-dialogue-ai-agent
pip install -r requirements.txt

# .env 文件放项目根目录
echo "DEEPSEEK_API_KEY=sk-..." > .env
echo "TAVILY_API_KEY=tvly-..." >> .env

python main.py
```

## 跑评估

```bash
python evals/run_eval.py           # 规则匹配
python evals/run_eval.py --judge   # + LLM-as-Judge
```

输出 `evals/report.md`（人类可读）和 `evals/results.json`（机器可读）。

## 项目结构

```
src/npc_agent/
  agent.py           ReAct 主循环
  llm_client.py      DeepSeek 客户端
  config.py          配置
  memory.py          对话记忆 + 摘要
  tracing.py         可观测性 trace
  tools/
    registry.py      工具注册表
    risk_score.py    确定性规则评分器（话术 + URL）
    input_guard.py   Prompt Injection 护栏
    knowledge.py     向量检索 RAG
    web_search.py    Tavily
    calculator.py
    time_tool.py

evals/
  cases.json         测试场景
  run_eval.py        规则匹配 + LLM-as-Judge 评估
  run_compare.py     4 路对比（DeepSeek base/Agent vs Qwen base/LoRA）
  judge.py           LLM-as-Judge

data/
  sft_seeds.json     50 条手工种子（覆盖 15 类诈骗手法 + 处置 + 防护）

scripts/
  expand_sft_data.py LLM 把种子扩成 ~250 条变体
  format_for_qwen.py 转 Qwen messages 格式 + 90/10 切分
  train_lora.py      Qwen2.5-1.5B + LoRA SFT 训练主脚本（CLI 参数化）

notebooks/
  train_qwen_lora_kaggle.ipynb   Kaggle 瘦壳，5 个 cell 调用 train_lora.py
  eval_compare_kaggle.ipynb      Kaggle 瘦壳，5 个 cell 调用 run_compare.py
```

## SFT / LoRA 工作流

所有训练 / 评估逻辑都在 `.py` 里。两种跑法二选一：

```bash
# ── 第 1 步：本地生成训练数据（共通） ──
python scripts/expand_sft_data.py    # 50 条种子 → ~250 条变体
python scripts/format_for_qwen.py    # 转 Qwen messages 格式 + 90/10 切分


# ── 第 2 步：训练，二选一 ──

# 选项 A: 本地或服务器有 GPU
python scripts/train_lora.py \
    --train-jsonl data/sft_train.jsonl \
    --val-jsonl   data/sft_val.jsonl \
    --output-dir  outputs/qwen-1.5b-xinzao-lora \
    --smoke-test

# 选项 B: Kaggle 上跑（T4 x2 / P100 都行）
# - 把 sft_train.jsonl + sft_val.jsonl 打包成 Kaggle Dataset
# - 把仓库代码打包成另一个 Kaggle Dataset（或让 notebook git clone）
# - 打开 notebooks/train_qwen_lora_kaggle.ipynb，run all
# 也可以把 scripts/train_lora.py 直接上传作 Kaggle Script Kernel


# ── 第 3 步：4 路对比评估 ──

# 本地（只跑 DeepSeek 两路，没 GPU 也行）
python evals/run_compare.py --strategies deepseek-base deepseek-agent

# Kaggle 跑全四路：notebooks/eval_compare_kaggle.ipynb
# 或 CLI：
python evals/run_compare.py --strategies all \
    --qwen-lora-path /path/to/qwen-1.5b-xinzao-lora
```

`scripts/train_lora.py` 的超参全部 CLI 化（rank / alpha / epochs / lr / batch / seq-len / seed），默认值已调好。Unsloth 自动启用，import 失败 fallback 到原生 transformers + peft + trl。

## Roadmap

- [x] Agentic RAG（Query Rewrite + BM25 混合检索 + quality_hint 自适应降级）
- [x] 长期向量记忆（跨会话）
- [x] SFT 数据构建 + Qwen2.5-1.5B LoRA 训练
- [ ] GRPO 后训练
- [ ] OCR 支持（直接传截图）
- [ ] 部署 hosted demo（Hugging Face Spaces）

## 已知局限

- 不替代专业律师 / 警方 / 心理咨询，涉及人身安全和大额损失请走正规渠道
- 知识库为静态 MVP，案例时效有限
- risk_score 为确定性规则，对谐音、黑话等变体覆盖有限

## License

MIT