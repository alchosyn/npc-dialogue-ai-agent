"""GRPO 后训练脚本：从 SFT LoRA adapter 热启动，跑 GRPO 进一步优化。

设计要点：
- Reward = LLM-as-judge 主体 + 规则项补充（见 scripts/grpo_reward.py）
- Warm-start 从 SFT 训出来的 LoRA adapter（不是 base 模型）
- 训练数据用 data/grpo_train.jsonl（cases.json + sft_seeds 合并的 65 条 prompt）
- 兼容 trl 不同版本（用 inspect 探测可用参数）
- bf16/fp16 自动按 GPU 探测
- 需要 DEEPSEEK_API_KEY（reward function 调 judge API）

用法：
    python scripts/train_grpo.py \\
        --base-model unsloth/Qwen2.5-1.5B-Instruct \\
        --sft-adapter outputs/qwen-1.5b-xinzao-lora \\
        --train-jsonl data/grpo_train.jsonl \\
        --output-dir outputs/qwen-1.5b-xinzao-grpo
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 把 scripts/ 加进 path 以便 import grpo_reward
sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # 数据 & 模型
    p.add_argument("--train-jsonl", required=True, type=Path)
    p.add_argument("--base-model", default="unsloth/Qwen2.5-1.5B-Instruct")
    p.add_argument("--sft-adapter", required=True, type=Path,
                   help="SFT 训出来的 LoRA adapter 目录（warm-start 起点）")
    p.add_argument("--output-dir", required=True, type=Path)
    # GRPO 关键超参
    p.add_argument("--num-generations", type=int, default=8,
                   help="每个 prompt 采样多少回答（GRPO 标志超参）")
    p.add_argument("--max-prompt-length", type=int, default=512)
    p.add_argument("--max-completion-length", type=int, default=512)
    # 训练超参（GRPO 必须比 SFT 小 2 个数量级）
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--num-train-epochs", type=float, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    # 精度 / 优化器（同 train_lora.py 套路）
    p.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--optim", default="auto")
    # 其他
    p.add_argument("--no-unsloth", action="store_true")
    p.add_argument("--judge-workers", type=int, default=8,
                   help="并发调 judge API 的线程数")
    return p.parse_args()


# ─── 精度 / 优化器探测（与 train_lora.py 共享逻辑）──────────


def _resolve_precision(precision_arg: str) -> tuple[bool, bool]:
    import torch
    if precision_arg == "bf16":
        return True, False
    if precision_arg == "fp16":
        return False, True
    if precision_arg == "fp32":
        return False, False
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        cap = torch.cuda.get_device_capability(0)
        print(f"[precision] GPU 支持 bf16 (cap {cap[0]}.{cap[1]})，使用 bf16")
        return True, False
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        print(f"[precision] GPU cap {cap[0]}.{cap[1]} 不支持 bf16，使用 fp16")
        return False, True
    print("[precision] WARNING: 无 GPU，使用 fp32")
    return False, False


def _resolve_optim(optim_arg: str) -> str:
    if optim_arg != "auto":
        return optim_arg
    try:
        import bitsandbytes  # noqa: F401
        return "adamw_8bit"
    except ImportError:
        return "adamw_torch"


# ─── 模型加载（warm-start 从 SFT adapter）──────────


def load_model_with_sft_adapter(args):
    """从 base + SFT LoRA adapter 加载，作为 GRPO 起点。"""
    use_unsloth = not args.no_unsloth
    if use_unsloth:
        try:
            return _load_with_unsloth(args)
        except ImportError as e:
            print(f"[load] Unsloth 不可用 ({e})，fallback 原生 transformers")
    return _load_native(args)


def _load_with_unsloth(args):
    from unsloth import FastLanguageModel

    print(f"[load] Unsloth: base={args.base_model}, adapter={args.sft_adapter}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_prompt_length + args.max_completion_length,
        dtype=None,
        load_in_4bit=False,
    )
    # 加载 SFT adapter
    model.load_adapter(str(args.sft_adapter), adapter_name="default")
    # 让 LoRA 参数 trainable（GRPO 继续训）
    model = FastLanguageModel.for_training(model)
    return model, tokenizer


def _load_native(args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = args.base_model.removeprefix("unsloth/")
    if "/" not in model_name:
        model_name = f"Qwen/{model_name}"

    print(f"[load] transformers + peft: base={model_name}, adapter={args.sft_adapter}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, str(args.sft_adapter), is_trainable=True)
    return model, tokenizer


# ─── 数据加载 ─────────────────────────────────────────


def load_grpo_dataset(args):
    from datasets import load_dataset
    ds = load_dataset("json", data_files={"train": str(args.train_jsonl)})["train"]
    print(f"[data] {len(ds)} prompts loaded")
    return ds


# ─── Reward function（包成 TRL 期望的 signature）──────────


def make_reward_func(judge_workers: int):
    """构造 reward function 闭包，捕获 judge client。"""
    from grpo_reward import batch_compute_reward
    from npc_agent.llm_client import get_client

    # 在闭包外初始化一次 judge_client（避免每次 reward call 重建）
    _judge_client_singleton = get_client()

    def _judge_via_client(case, reply):
        """匹配 grpo_reward 期望的 judge_client 签名。"""
        # 复用 evals/judge.py 的 llm_judge —— 但通过 singleton client
        evals_dir = Path(__file__).resolve().parent.parent / "evals"
        if str(evals_dir) not in sys.path:
            sys.path.insert(0, str(evals_dir))
        from judge import llm_judge
        return llm_judge(case, reply)

    def reward_func(prompts, completions, scenario_keywords=None, **kwargs):
        """TRL GRPOTrainer 期望的 signature：(prompts, completions, **dataset_columns) -> rewards."""
        return batch_compute_reward(
            prompts=prompts,
            completions=completions,
            scenario_keywords_list=scenario_keywords,
            judge_client=_judge_via_client,
            max_workers=judge_workers,
        )

    return reward_func


# ─── Trainer 构建 ─────────────────────────────────────


def build_trainer(args, model, tokenizer, dataset, reward_func):
    """兼容不同 trl 版本构建 GRPOTrainer。"""
    import inspect

    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError:
        raise ImportError(
            "GRPOTrainer 需要 trl >= 0.13。"
            "Kaggle 上请先 pip install -U 'trl>=0.18,<0.25'"
        )

    use_bf16, use_fp16 = _resolve_precision(args.precision)
    optim = _resolve_optim(args.optim)

    config_params = set(inspect.signature(GRPOConfig).parameters.keys())

    config_kwargs = dict(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=2,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=use_bf16,
        fp16=use_fp16,
        optim=optim,
        seed=args.seed,
        report_to="none",
        # GRPO 特有
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
    )

    # 过滤掉当前版本不支持的参数
    config_kwargs = {k: v for k, v in config_kwargs.items() if k in config_params}
    config = GRPOConfig(**config_kwargs)

    trainer_params = set(inspect.signature(GRPOTrainer).parameters.keys())
    trainer_kwargs = dict(
        model=model,
        args=config,
        train_dataset=dataset,
        reward_funcs=reward_func,
    )
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    return GRPOTrainer(**trainer_kwargs)


# ─── 主流程 ───────────────────────────────────────────


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 检查环境变量
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: 需要 DEEPSEEK_API_KEY 环境变量（reward function 调 judge API）")
        sys.exit(1)

    try:
        import torch
        print(f"[env] torch={torch.__version__} cuda={'yes' if torch.cuda.is_available() else 'no'} "
              f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    except ImportError:
        print("[env] WARNING: torch 未安装")

    # 1. 加载模型 + SFT adapter（warm-start）
    model, tokenizer = load_model_with_sft_adapter(args)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    # 2. 加载训练数据
    dataset = load_grpo_dataset(args)

    # 3. 构造 reward 函数
    reward_func = make_reward_func(args.judge_workers)

    # 4. 构建 trainer
    trainer = build_trainer(args, model, tokenizer, dataset, reward_func)

    # 5. 训练
    print(f"\n[train] 开始 GRPO，目标 {args.num_train_epochs} epoch × {args.num_generations} generations/prompt")
    print(f"        预估 {int(len(dataset) * args.num_generations * args.num_train_epochs)} 次 reward 调用")
    result = trainer.train()
    print(f"\n[train] 完成: {result.metrics}")

    # 6. 保存
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"[save] adapter 保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
