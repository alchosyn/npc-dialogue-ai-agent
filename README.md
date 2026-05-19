# XinZao · Anti-Fraud Dialogue Agent

LLM-powered anti-fraud assistant. Users paste a suspicious message, and the agent identifies scam patterns, explains the attack, and gives actionable advice.

## Architecture

```
User Input
  │
  ├─ input_guard (prompt injection detection)
  │
  ▼
ReAct Loop (max 6 steps)
  ├─ risk_score        Rule-based scoring (13 scam pattern regexes + URL spoofing detection)
  ├─ search_knowledge  BM25 retrieval over local knowledge base (55 anti-fraud entries)
  ├─ web_search        Tavily live search
  ├─ calculator        Password entropy estimation
  └─ get_current_time  Recency check
  │
  ▼
Langfuse Trace (token count / latency / tool calls per step)
```

| Layer | Stack |
|---|---|
| LLM | DeepSeek-V3 (OpenAI-compatible API) |
| Embedding | paraphrase-multilingual-MiniLM-L12-v2 |
| Retrieval | BM25 + query rewriting |
| Web Search | Tavily |
| Evaluation | LLM-as-Judge, 4 dimensions |

## SFT Fine-tuning

50 hand-written seed examples, distilled via DeepSeek to 220 training samples. Qwen2.5-1.5B + LoRA (rank 16), trained on Kaggle T4.

Script: `scripts/train_lora.py`. Unsloth accelerated, auto-fallback to native transformers + peft if Unsloth is unavailable.

## GRPO Post-training

GRPO (same RL algorithm as DeepSeek-R1) on top of the SFT LoRA adapter, with RLAIF-style hybrid reward.

### Reward Design

| Component | Description | Score |
|---|---|---|
| R1 Answer quality | LLM-as-judge via DeepSeek API, overall × 0.6 | 0~3 |
| R2 Step structure | Regex match for inline 1.2.3. steps | +0.5 |
| R3 Real hotline | Contains 96110 / 110 / 12321 etc. | +0.5 |
| P1 Fake hotline | Contains non-existent numbers like 95110 | -1 |
| P2 Too brief | Urgent scenario but reply under 50 chars | -0.5 |
| P3 AI-speak | Markdown lists / emoji bullets / excessive bold | -0.5 |
| P4 Too long | Over 400 characters | -0.5 |

P3 and P4 counteract LLM-judge structure-bias. The judge naturally prefers markdown-heavy long responses. Without these penalties, GRPO would overfit to the judge and destroy the natural tone that SFT learned.

### Training Config

- Warm-start from SFT LoRA adapter
- 65 prompts, 4 generations/prompt, 2 epochs, 520 rollouts total
- ~30 min on Colab T4, DeepSeek API cost under $1
- Script: `scripts/train_grpo.py`, notebook: `notebooks/train_grpo_colab.ipynb`

## Evaluation Results

15 anti-fraud scenarios, 5-way comparison, LLM-as-Judge scoring (out of 5):

| Strategy | overall | accuracy | actionability | citation | tone |
|---|---:|---:|---:|---:|---:|
| deepseek-agent (RAG + tools + guard) | **4.63** | 5.00 | 5.00 | 3.53 | 5.00 |
| deepseek-base (LLM only) | 4.03 | 5.00 | 5.00 | 2.07 | 4.07 |
| qwen-grpo (SFT + GRPO) | 3.63 | 4.40 | 4.40 | 1.80 | 3.93 |
| qwen-lora (SFT) | 3.63 | 4.40 | 4.67 | 1.60 | 3.87 |
| qwen-base (vanilla 1.5B) | 3.35 | 4.20 | 4.20 | 2.00 | 3.00 |

### Per-layer Improvement

| Operation | Δ overall | Key changes |
|---|---:|---|
| SFT | +0.28 | tone +0.87 |
| GRPO | +0.00 | citation +0.20, tone +0.06, actionability -0.27 |
| RAG + Agent | +0.60 | citation +1.46, tone +0.93 |

SFT's biggest gain is in tone. Style transfer is where small-data fine-tuning pays off the most. GRPO improved citation and tone slightly, but the training set (65 prompts × 2 epochs) was too small to move the needle on overall. Citation can only be meaningfully improved by RAG. Both SFT and GRPO plateau at 1.6~1.8 while the RAG agent reaches 3.53.

Full per-case report: `evals/compare_report.md`

## Quick Start

```bash
git clone https://github.com/alchosyn/npc-dialogue-ai-agent.git
cd npc-dialogue-ai-agent
pip install -r requirements.txt

# .env
echo "DEEPSEEK_API_KEY=sk-..." > .env
echo "TAVILY_API_KEY=tvly-..." >> .env

python main.py
```

## Project Structure

```
src/npc_agent/
  agent.py          ReAct loop
  llm_client.py     DeepSeek client
  memory.py         Conversation memory
  tracing.py        Langfuse tracing
  tools/
    risk_score.py   Rule-based scorer
    input_guard.py  Injection detection
    knowledge.py    BM25 retrieval
    web_search.py   Tavily search

scripts/
  expand_sft_data.py     Seed expansion (50 → 220)
  format_for_qwen.py     Convert to Qwen chat format
  train_lora.py          LoRA SFT training
  train_grpo.py          GRPO post-training
  grpo_reward.py         Hybrid reward function
  build_grpo_dataset.py  GRPO dataset builder

evals/
  run_compare.py    5-way comparison
  judge.py          LLM-as-Judge
  cases.json        15 test scenarios

notebooks/
  train_grpo_colab.ipynb       GRPO training (Colab)
  train_grpo_kaggle.ipynb      GRPO training (Kaggle)
  train_qwen_lora_kaggle.ipynb SFT training (Kaggle)
  eval_compare_kaggle.ipynb    Evaluation (Kaggle)
```

## License

MIT
