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

> **方法论说明**：早期跑数有个 silent bug——`run_compare.py` 用单一全局变量缓存 Qwen 模型，同一进程里跑 `qwen-base` 后再跑 `qwen-lora` 时，后者直接返回了前者缓存的 base 模型（没装 LoRA）。导致 qwen-lora 实际等于 qwen-base，第一版报告里 LoRA "只提升 +0.05" 是这个 bug 的假象。已修（cache key 改为 `(base, adapter)` 元组）并重跑。下表是**修复后**的数字。

### 总体得分（5 分制，修复 cache bug 后重跑）

| 策略 | overall | accuracy | actionability | citation | tone |
|---|---:|---:|---:|---:|---:|
| **deepseek-agent**（完整 agent，RAG + 5 工具 + injection guard） | **4.55** | 5.00 | 5.00 | 3.20 | 5.00 |
| deepseek-base（DeepSeek-V3 纯 LLM，无 RAG） | 3.97 | 5.00 | 4.93 | 2.07 | 3.87 |
| **qwen-lora**（Qwen2.5-1.5B + LoRA SFT，220 样本 45 step） | **3.83** | 4.53 | 4.60 | 1.80 | 4.40 |
| qwen-base（Qwen2.5-1.5B-Instruct 原生） | 3.20 | 4.00 | 4.20 | 1.67 | 2.93 |

### Ablation：单项杠杆贡献

| 杠杆 | 操作 | Δ overall | 关键维度 |
|---|---|---:|---|
| **LoRA SFT** | qwen-base → qwen-lora（220 样本，45 step） | **+0.63** | **tone +1.47**、accuracy +0.53、actionability +0.40 |
| 换大模型 | qwen-base → deepseek-base（~100× 容量） | +0.77 | citation +0.40、tone +0.94 |
| RAG + tools + agent loop | deepseek-base → deepseek-agent | +0.58 | **citation +1.13**、tone +1.13 |

### 三个能讲故事的发现

1. **LoRA 真的有效，尤其在 tone 维度。** 220 条蒸馏数据 + 45 step 把 overall 拉了 +0.63，其中 tone 维度 **2.93 → 4.40（+1.47，+50%）**——证明风格迁移是 SFT 在小数据上最容易吃到的红利。一开始判定"LoRA 没用"是 cache bug 假象，**修 bug + 重跑才暴露真相**——这本身是关于"eval 基础设施可信度"的硬教训。
2. **Citation 仍是 RAG 的护城河，SFT 撬不动。** qwen 不管有没有 LoRA，citation 都卡在 1.7-1.8（SFT 种子答案里压根没"引用来源"这个行为）；deepseek-agent 靠 RAG 拿到 3.20。在受监管场景（医疗 / 金融 / 反诈），"能说出来源"决定能否落地——这是 RL/RAG 该补的，不是 SFT。
3. **单次跑的 LLM-judge 有 ±0.15 抖动，单项 ablation 差值 < 0.2 不可信。** deepseek 两路没受 cache bug 影响，但两次跑 overall 仍变了 ±0.1~0.15（agent 回复 temperature=0.7 每次不同）。所以"RAG +0.58 vs 换模型 +0.77"这种 0.2 内的对比不能下强结论，要多 seed 平均才行。**唯一可信的强结论是 LoRA +0.63 这种大效应**。这条方法论自觉是简历里能体现 senior 的点。

### 部署边界（trade-off 分析）

| 场景 | 推荐方案 | 理由 |
|---|---|---|
| 在线对话客服、企业培训内容生成 | **deepseek-agent** | 4.55/5，可控来源引用，调用成本可忽略 |
| 隐私敏感数据（银行 / 政务内网） | **qwen-lora** | 3.83/5 已逼近 deepseek-base，gap 仅 0.72；大模型 API 出域受限时本地推理是硬约束 |
| 离线设备 / 边缘部署 | 不适合 1.5B | 1.5B 仍需 GPU，纯 CPU 推理太慢；要么上 ONNX/GGUF，要么换 0.5B 级 |

完整逐 case 报告：`evals/compare_report.md`（跑完 `python evals/run_compare.py --strategies all` 自动生成）。

## GRPO 后训练设计（脚手架就绪，待跑数）

在 SFT LoRA 之上加一层 GRPO（Group Relative Policy Optimization，DeepSeek-R1 同款 RL 算法）后训练，目标是用可程序化 reward 进一步压榨 actionability + citation 维度。

### Reward 设计：LLM-as-judge 主体 + 4 项规则补充

**为什么不走纯规则**（项目迭代过程的两次反馈定下）：
1. 不 reward "引用 KB ID" —— SFT 50 条种子答案里 0 条带 `pattern-XXX` 这种格式 ID，GRPO 不能 reward SFT 没教过的行为
2. 不走"硬关键词匹配" —— 典型 reward hacking 陷阱，模型会塞关键词凑分而不真理解

**先记一个关键发现：LLM-judge 有 structure-bias。** 人工 spot check 对比报告样本发现：judge 给 deepseek-agent 的 tone 打 5.0、给 qwen-lora 打 4.40，但**人眼读下来 qwen-lora 更像真人、deepseek-agent 一股 AI 味**。原因是 judge（DeepSeek-V3）有 length / structure / self-preference 三重偏好——它奖励 markdown 标题、加粗块、emoji 分点、长回答，而这些正是"AI 腔"的来源。这是论文反复证实的 LLM-judge 系统性偏差，不是噪音。

**这对 GRPO reward 设计是个陷阱**：如果 reward 里 judge 占大头不加约束，GRPO 会为了讨好 judge 把已经"够人味"的 LoRA 往 AI 腔训坏（加 markdown、变长）——经典的 reward misspecification。所以 reward 里加了**反 judge-bias 护栏**（罚 3 / 罚 4）。

**最终设计**（`scripts/grpo_reward.py`）：

| 层 | 来源 | 范围 | 备注 |
|---|---|---|---|
| **R1 主回答质量** | 调 DeepSeek API 跑 LLM-as-judge，复用 `evals/judge.py` 拿 overall 分 × 0.6 缩放 | 0-3 | reward 大头，**直接对齐 eval 指标** |
| R2 步骤化奖励 | 正则匹配信噪式 **inline** 1.2.3.（非 markdown 列表） | +0.5 | 小 bonus |
| R3 真实电话奖励 | 含 96110/110/12321/95XXX（数字边界检查防 95110 误匹配 110） | +0.5 | 小 bonus |
| 罚 1 假电话 | 出现 95110/94110/92110 等冒用号码 | -1 | hard penalty |
| 罚 2 紧急敷衍 | 用户输入含紧急词 ("被骗"/"刚转") 但回复 < 50 字 | -0.5 | 防废话 |
| **罚 3 AI 腔结构** | markdown 标题/列表/分隔线/emoji 分点/3+ 加粗块 | -0.5 | **反 judge structure-bias 护栏** |
| **罚 4 过长** | 回复 > 400 字（信噪种子均长 214） | -0.5 | **反 judge verbosity-bias 护栏** |

**总分范围 [-2.5, 4]**。设计逻辑：judge 提供语义质量信号（避免硬关键词 reward hacking），R2/R3 引导关键形式，罚 3/罚 4 压住 judge 的 structure-bias——让 GRPO 只提 citation/actionability，**tone 不许从 LoRA 的 4.40 回归**。R2 特意只认 inline 步骤（信噪那种"三件事：1.…2.…3."），而罚 3 专打换行后的 markdown 列表，两者用"数字是否在行首"区分，不冲突。

### 工程价值（写进简历）

- 这是 **RLAIF (RL from AI Feedback)** 风格而不是纯规则 RL，跟 Constitutional AI / R1 思路一致
- **发现 LLM-judge structure-bias 并在 reward 里工程化对冲**——从"会用 judge"到"知道 judge 何时不可信、并设计去偏置 reward"，是 reward shaping 的核心 senior 能力
- Reward 跟 eval 指标用的是**同一个 LLM judge**，训练直接优化最终评估目标——闭环，但有意加护栏防止 judge 偏置被 RL 放大
- 单元测试覆盖 13 个 reward 分支（judge 部分用 mock），**纯逻辑可独立验证**

### 训练参数

- **Warm-start**: 从 SFT 训出来的 LoRA adapter 继续训（不是从 base 开始）
- **Prompt 数据**: 65 条 = `evals/cases.json` 15 + `data/sft_seeds.json` 50（去重后）
- **Generations per prompt**: 8（GRPO 标志超参，组内归一化用）
- **Epochs**: 2，total ~1040 rollouts
- **Learning rate**: 1e-6（GRPO 必须比 SFT 小 2 个数量级）
- **Optimizer / precision**: 自动按 GPU 探测，T4 走 fp16 + adamw_8bit
- **预估**: Kaggle T4 上 30-60 分钟，DeepSeek API 成本 ~$1（1040 次 judge 调用）

### 脚手架完整度

| 组件 | 文件 | 状态 |
|---|---|---|
| Reward 函数 | `scripts/grpo_reward.py` | ✅ 单测 13/13 过 |
| 训练数据合并 | `scripts/build_grpo_dataset.py` | ✅ 输出 65 条 |
| 训练主入口 | `scripts/train_grpo.py` | ✅ CLI 参数化 + trl 版本兼容 |
| Kaggle notebook | `notebooks/train_grpo_kaggle.ipynb` | ✅ 5 cell 瘦壳 |
| 5 路对比扩展 | `evals/run_compare.py --strategies all qwen-grpo` | ✅ 加了 `--qwen-grpo-path` |

跑完后会有 5 路对比表（GRPO 行待 Kaggle 跑数回填，其余为修复 cache bug 后的真实数）：

| 策略 | overall | citation | actionability |
|---|---:|---:|---:|
| deepseek-agent | 4.55 | 3.20 | 5.00 |
| deepseek-base | 3.97 | 2.07 | 4.93 |
| **qwen-grpo** | _待 Kaggle 跑数_ | _待填_ | _待填_ |
| qwen-lora | 3.83 | 1.80 | 4.60 |
| qwen-base | 3.20 | 1.67 | 4.20 |

GRPO 的设计目标：用 judge-based reward 把 qwen-lora 的 citation（1.80）和 actionability（4.60）再往上推。tone 已经被 LoRA 推到 4.40 接近天花板，GRPO 不主攻这维。

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
- [x] 修复 qwen cache bug 并重跑，确认 LoRA 真实提升 +0.63（tone +1.47）
- [x] **GRPO 后训练脚手架**（reward 函数 + 训练脚本 + Kaggle notebook + 5 路对比；待 Kaggle 跑数）
- [ ] Kaggle 跑 GRPO，回填 5 路对比表的 qwen-grpo 行
- [ ] 多 seed 重复 eval，给 ablation 差值加置信区间
- [ ] OCR 支持（直接传截图）
- [ ] 部署 hosted demo（Hugging Face Spaces）

## 已知局限

- 不替代专业律师 / 警方 / 心理咨询，涉及人身安全和大额损失请走正规渠道
- 知识库为静态 MVP，案例时效有限
- risk_score 为确定性规则，对谐音、黑话等变体覆盖有限
- 1.5B + LoRA 在 tone / citation 上离 RAG agent 仍有 35%+ 差距（详见评估结果章节）

## 工程反思：verifiable reward 是 agent 产品化的隐形门槛

跑完整套实验后回头看，本项目最值得记下的不是某项技术，而是一个关于 agent 落地的判断：**reward 能不能被机器自动打分，决定了这个 agent 能不能进入「自我提升」循环。**

把 agent 项目按 reward 性质粗分两类：

| 类型 | 例子 | reward 来源 | 能否扩规模 RL |
|---|---|---|---|
| Verifiable | 代码 / 数学 / SQL / 网页自动化 / API tool calling | 编译通过、单测过、查询结果对比、任务完成 —— 程序化 0/1 | ✅ 每秒可 score 数千样本 |
| Subjective | 反诈对话 / 客服 / 心理咨询 / 创意写作 | 人工评分或 LLM-as-judge —— 慢、贵、噪音大 | ❌ 单位 reward 成本不随规模下降 |

DeepSeek-R1 / Cursor / Devin 能用 RL 把效果训到 SOTA，本质都因为前者：**reward 函数能跑在 CI 里，给 GRPO 提供可扩展的训练信号**。

本项目的反诈对话属于后者。一个具体证据：LoRA SFT 把 **tone 拉了 +1.47**（风格是 subjective 维度里 SFT 最容易吃的红利），但 **citation 几乎没动（1.67 → 1.80）**——因为"引用真实来源"这件事 SFT 种子里没有，且没有廉价可验证的 reward 信号能教会它。citation 这种"需要 grounding"的维度，只有 RAG（检索给出真来源）或 verifiable RL（能机检引用对不对）补得上，纯 SFT 撬不动。这恰好印证了 verifiable / subjective 的分野。

**另一条更硬的教训——eval 基础设施本身要被怀疑。** 第一版报告得出"LoRA 只提升 +0.05、基本没用"的结论，差点据此决定"扩 5 倍数据 + 换非 Instruct 基座重训"。后来发现是 `run_compare.py` 的模型缓存 bug（qwen-lora 实际跑的是 qwen-base）。修 bug 重跑，真实提升是 **+0.63**。**一个 eval 代码的 silent bug，差点让我把工程方向带偏一整周。** 教训：任何"反直觉的负面结论"在动手补救前，先质疑测量管线本身；任何 cache 都要有 invalidation key，测试代码也不例外。

**如果再做 agent 项目，会先问自己三个问题（按重要性排序）：**

1. 这个任务的 reward 能不能跑在 unit test 里？能 → verifiable，可走 RL；不能 → 老老实实做好 RAG + prompt + LLM-as-judge eval
2. 训练数据能不能从 production trace 里自然采集？能 → 闭环 self-improving；不能 → 数据是上限
3. eval 管线自己有没有被验证过？拿到反直觉结论时，第一反应是查测量代码而不是改模型

这些判断应该**早于**"选什么模型 / 用什么框架"。是 ML engineer 跟 ML researcher 真正的分水岭。

## 简历范本句（基于实测数据）

> **NPC Dialogue AI Agent —— 中文反诈安全意识助手**（个人项目 · [GitHub](https://github.com/alchosyn/npc-dialogue-ai-agent)）
> - 基于 DeepSeek-V3 + 自建 RAG（55 条权威知识库，BM25 + 向量混合检索 + query rewriting）构建中文反诈对话 Agent，自研 5 个工具（含确定性规则风险打分器 + prompt injection 护栏）
> - 设计 4-5 路对比评估管线（DeepSeek 裸调 / + RAG agent / Qwen2.5-1.5B base / + LoRA / + GRPO），使用 LLM-as-Judge 在 4 维度（accuracy / actionability / citation / tone）上量化打分
> - **完整 SFT 管线 + 量化收益**：50 条手工种子 → LLM 扩展至 220 条 → Qwen2.5-1.5B + LoRA(rank 16) 在 Kaggle T4 训练（Unsloth 加速，bf16/fp16 自动适配）。LoRA 使 overall **3.20 → 3.83 (+0.63)**，其中 **tone 维度 +1.47（+50%）**，gap 到 DeepSeek+RAG agent 收窄一半
> - **eval 基础设施 debug 实战**：第一版评估因模型缓存 bug（qwen-lora 实际复用了 qwen-base）误判"LoRA 无效 +0.05"，差点据此误改训练方向；定位并修复 cache-key bug 后重跑得真实 +0.63。沉淀出"反直觉负面结论先质疑测量管线"的方法论
> - **ablation 量化 RAG 不可替代性**：citation 维度 SFT 撬不动（qwen 有无 LoRA 都卡 1.7-1.8），唯 RAG agent 拿到 3.20，量化了受监管场景 RAG 的护城河；同时指出单次 LLM-judge 有 ±0.15 抖动，强结论需大效应（如 LoRA +0.63）或多 seed 平均
> - **GRPO 后训练 + reward 去偏置**：SFT LoRA 之上叠 GRPO（DeepSeek-R1 同款 RL），RLAIF 风格混合 reward；**人工 spot check 发现 LLM-judge 有 structure-bias（偏好 markdown/长文，judge 高分的 deepseek-agent 实际更 AI 腔），据此在 reward 里加 anti-structure / anti-verbosity 护栏对冲，防止 RL 把 judge 偏置放大、训坏 LoRA 已得的人味**；规避"reward SFT 未教行为""硬关键词 hacking""judge bias 放大"三个陷阱；脚手架完整（reward 单测 13/13 过 / TRL 版本兼容 / Kaggle notebook 瘦壳）
> - 技术栈：Python 包结构 / DeepSeek API / sentence-transformers / rank-bm25 / transformers + peft + trl / Unsloth

## License

MIT