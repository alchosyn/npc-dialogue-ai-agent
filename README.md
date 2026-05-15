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

## 评估结果

在 15 个反诈/钓鱼场景上做 4 路对比 + 4 维度 LLM-as-Judge 评分（DeepSeek-V3 当 judge），所有数字来自 `evals/run_compare.py` 的实际跑数。

### 总体得分（5 分制）

| 策略 | overall | accuracy | actionability | citation | tone |
|---|---:|---:|---:|---:|---:|
| **deepseek-agent**（完整 agent，RAG + 5 工具 + injection guard） | **4.70** | 5.00 | 5.00 | 3.80 | 5.00 |
| deepseek-base（DeepSeek-V3 纯 LLM，无 RAG） | 3.87 | 5.00 | 5.00 | 1.80 | 3.67 |
| qwen-lora（Qwen2.5-1.5B + LoRA SFT，220 样本） | 3.27 | 4.27 | 4.00 | 1.73 | 3.07 |
| qwen-base（Qwen2.5-1.5B-Instruct 原生） | 3.22 | 4.20 | 4.07 | 1.60 | 3.00 |

### Ablation：单项杠杆贡献

| 杠杆 | 操作 | Δ overall | 备注 |
|---|---|---:|---|
| **RAG + tools + agent loop** | deepseek-base → deepseek-agent | **+0.83** | 单维 citation 翻倍以上 (+2.00)、tone +1.33 |
| 换大模型 | qwen-base → deepseek-base（~100x 容量） | +0.65 | accuracy +0.80、actionability +0.93 |
| LoRA SFT | qwen-base → qwen-lora（220 样本，45 step） | +0.05 | 各维度都接近持平 |

### 三个能讲故事的发现

1. **工程层 (+0.83) 比换模型 (+0.65) 更划算。** 在垂直域应用里，RAG + 自研工具的回报率高于把基座从 1.5B 换到 V3 级模型。这量化了"为什么 RAG 是 LLM 落地刚需"。
2. **Citation 是 RAG 的护城河。** 没有 RAG 时，无论模型大小 citation 都低（qwen-base 1.60、deepseek-base 1.80）；接上 RAG agent 后跃升到 3.80。在受监管场景（医疗 / 金融 / 反诈），"能说出来源"决定能否落地，光靠 RLHF 教不会。
3. **220 条 SFT 不够撬动 RLHF 后的 1.5B 模型。** LoRA 给的 +0.05 接近 noise floor。结论：要让小模型学会"信噪"这种特定语气，至少需要 500-1000 条数据 + 100+ 优化器 step（当前 45 step），或者改用非 Instruct 基座降低 RLHF 惯性。

### 部署边界（trade-off 分析）

| 场景 | 推荐方案 | 理由 |
|---|---|---|
| 在线对话客服、企业培训内容生成 | **deepseek-agent** | 4.70/5，可控来源引用，调用成本可忽略 |
| 隐私敏感数据（银行 / 政务内网） | qwen-lora（再加训） | 大模型 API 出域受限，本地推理是硬约束 |
| 离线设备 / 边缘部署 | 不适合 1.5B | 1.5B 仍需 GPU，纯 CPU 推理太慢；要么上 ONNX/GGUF，要么换 0.5B 级 |

完整逐 case 报告：`evals/compare_report.md`（跑完 `python evals/run_compare.py --strategies all` 自动生成）。

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
- [x] 4 路 LLM-as-Judge 评估 + ablation 分析
- [ ] 扩种子到 500 样本 + 8 epoch 重训，目标 tone 突破 3.5
- [ ] GRPO 后训练
- [ ] OCR 支持（直接传截图）
- [ ] 部署 hosted demo（Hugging Face Spaces）

## 已知局限

- 不替代专业律师 / 警方 / 心理咨询，涉及人身安全和大额损失请走正规渠道
- 知识库为静态 MVP，案例时效有限
- risk_score 为确定性规则，对谐音、黑话等变体覆盖有限
- 1.5B + LoRA 在 tone / citation 上离 RAG agent 仍有 35%+ 差距（详见评估结果章节）

## 简历范本句（基于实测数据）

> **NPC Dialogue AI Agent —— 中文反诈安全意识助手**（个人项目 · [GitHub](https://github.com/alchosyn/npc-dialogue-ai-agent)）
> - 基于 DeepSeek-V3 + 自建 RAG（55 条权威知识库，BM25 + 向量混合检索 + query rewriting）构建中文反诈对话 Agent，自研 5 个工具（含确定性规则风险打分器 + prompt injection 护栏）
> - 设计 4 路对比评估管线（DeepSeek 裸调 / + RAG agent / Qwen2.5-1.5B base / + LoRA），使用 LLM-as-Judge 在 4 维度（accuracy / actionability / citation / tone）上量化打分
> - **关键 ablation 发现：RAG + tools 工程层贡献 (+0.83/5 overall) 超过将基座从 1.5B 换到 V3 级（~100× 容量，+0.65/5），证明垂直域 LLM 应用工程层 ROI 高于模型层；citation 维度 RAG 贡献 +2.00/5 是没有 RAG 时的两倍以上，量化了 RAG 在受监管场景的不可替代性**
> - 完整 SFT 管线：50 条手工种子 → LLM 扩展至 220 条 → Qwen2.5-1.5B + LoRA (rank 16) 在 Kaggle T4 训练（Unsloth 加速，bf16/fp16 自动适配），双轨 eval（规则匹配 + LLM-Judge）+ spearman ρ 一致性分析
> - 技术栈：Python 包结构 / DeepSeek API / sentence-transformers / rank-bm25 / transformers + peft + trl / Unsloth

## License

MIT