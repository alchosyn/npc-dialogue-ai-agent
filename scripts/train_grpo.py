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

# Windows 控制台默认 cp936/cp932，大量中文 print 会 UnicodeEncodeError 而中断训练日志。
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")

# 路径设置：scripts/ 用于 import grpo_reward；src/ 用于 import npc_agent；
# evals/ 用于 import judge（reward 函数链路上会用到）
_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_SCRIPTS_DIR))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "evals"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # 数据 & 模型
    p.add_argument("--train-jsonl", required=True, type=Path)
    p.add_argument("--base-model", default="unsloth/Qwen2.5-1.5B-Instruct")
    p.add_argument("--sft-adapter", required=True, type=Path,
                   help="SFT 训出来的 LoRA adapter 目录（warm-start 起点）")
    p.add_argument("--output-dir", required=True, type=Path)
    # GRPO 关键超参
    p.add_argument("--num-generations", type=int, default=4,
                   help="每个 prompt 采样多少回答（GRPO 标志超参）。须满足 "
                        "batch-size 与 batch-size×grad-accum 都能被它整除；"
                        "8 理想但 16GB T4 易 OOM，4 是可用下限")
    p.add_argument("--max-prompt-length", type=int, default=384)
    p.add_argument("--max-completion-length", type=int, default=384,
                   help="信噪种子答案均长 214 字，384 token 足够；越大越吃显存/judge 成本")
    # 训练超参（GRPO 必须比 SFT 小 2 个数量级）
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--num-train-epochs", type=float, default=2)
    p.add_argument("--batch-size", type=int, default=4,
                   help="per-device batch（TRL GRPO 里实为 completions 数）。"
                        "须为 num-generations 的整数倍")
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

    use_bf16, use_fp16 = _resolve_precision(args.precision)
    if use_bf16:
        dtype = torch.bfloat16
    elif use_fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"[load] transformers + peft: base={model_name}, adapter={args.sft_adapter}, dtype={dtype}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
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


def _to_text(completion) -> str:
    """TRL 对话式数据集回传的 completion 是 [{"role":"assistant","content":..}]，
    非对话式是 str。统一成字符串给 compute_reward。"""
    if isinstance(completion, list):
        return completion[-1]["content"] if completion else ""
    return completion


def _user_text(prompt) -> str:
    """对话式 prompt 是 [{system},{user}]，取最后一条 user 内容（judge 要纯问题串）。
    非对话式是 str，原样返回。"""
    if isinstance(prompt, list):
        users = [
            m["content"] for m in prompt
            if isinstance(m, dict) and m.get("role") == "user"
        ]
        if users:
            return users[-1]
        return prompt[-1]["content"] if prompt else ""
    return prompt


def make_reward_func(judge_workers: int):
    """构造 reward function 闭包。

    reward 链路：batch_compute_reward → compute_reward → judge_client。
    judge_client 用 evals/judge.py 的 llm_judge（它内部自己管 DeepSeek client，
    复用 npc_agent.llm_client 的进程级单例，不需要这里再传 client）。
    """
    from grpo_reward import batch_compute_reward
    from judge import llm_judge  # evals/ 已在 sys.path（见文件顶部）

    def _judge_via_client(case, reply):
        """匹配 grpo_reward 期望的 judge_client 签名 (case, reply) -> {overall: ...}."""
        return llm_judge(case, reply)

    def reward_func(prompts, completions, scenario_keywords=None, **kwargs):
        """TRL GRPOTrainer 期望的 signature：(prompts, completions, **dataset_columns) -> rewards.

        对话式数据集下 prompts/completions 是 list-of-messages，先解包成字符串
        （compute_reward 收字符串，其单测也按字符串写，保持不动）。
        """
        prompt_texts = [_user_text(p) for p in prompts]
        completion_texts = [_to_text(c) for c in completions]
        return batch_compute_reward(
            prompts=prompt_texts,
            completions=completion_texts,
            scenario_keywords_list=scenario_keywords,
            judge_client=_judge_via_client,
            max_workers=judge_workers,
        )

    return reward_func


# ─── Trainer 构建 ─────────────────────────────────────


def _ensure_mergekit_importable() -> None:
    """给 trl 塞个 mergekit 桩，绕开它的无条件可选 import。

    背景：trl 的 callbacks.py 顶层 `from ..mergekit_utils import ...`，
    mergekit_utils 又 `from mergekit.config import ...`。mergekit 是模型
    合并工具，GRPO 训练完全不用它，但这版 trl 没做软导入，缺 mergekit
    （或它的传递依赖如 immutables）就 import 不了 GRPOTrainer。

    与其在 Kaggle 上满地追 mergekit 的依赖（immutables → 下一个 → ...），
    不如塞个万能桩：任何 `from mergekit.X import Y` 都返回无害占位类。
    merge 功能 GRPO 永不触发，桩永远不会被真调用。
    """
    import importlib
    import sys
    import types

    try:
        importlib.import_module("mergekit.config")
        importlib.import_module("mergekit.common")
        return  # 真 mergekit 完全可用，不塞桩
    except Exception:
        pass

    class _AnyModule(types.ModuleType):
        # 任何属性访问都返回一个无害的占位类
        def __getattr__(self, name: str):
            return type(name, (), {"__init__": lambda self, *a, **k: None})

    # 无条件覆盖：半装的坏 mergekit 可能在 import 失败后于 sys.modules
    # 残留碎片，会盖住桩。这里强制替换所有 mergekit* 入口。
    for modname in (
        "mergekit",
        "mergekit.config",
        "mergekit.common",
        "mergekit.merge",
        "mergekit.options",
        "mergekit.architecture",
        "mergekit.io",
        "mergekit.plan",
    ):
        sys.modules[modname] = _AnyModule(modname)
    print("[grpo] mergekit 不可用，已注入桩（GRPO 不用模型合并，安全）")


def build_trainer(args, model, tokenizer, dataset, reward_func):
    """兼容不同 trl 版本构建 GRPOTrainer。"""
    import inspect

    _ensure_mergekit_importable()  # 必须在 import GRPOTrainer 之前

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
        # 显存：16GB T4 上 1.5B GRPO 采样+前后向吃紧，必须开梯度检查点
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # GRPO 特有
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
    )

    # 过滤掉当前版本不支持的参数
    config_kwargs = {k: v for k, v in config_kwargs.items() if k in config_params}
    config = GRPOConfig(**config_kwargs)

    # 梯度检查点 + PEFT(warm-start adapter) 同时开时，base 被冻结会导致
    # backward 报 "element 0 of tensors does not require grad"。让 embedding
    # 输出 require grad 打通梯度链路（幂等，非 PEFT 模型上也安全）。
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

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


def _preflight(args) -> None:
    """把"训到一半/训完才暴露"的失败提前到加载模型之前。"""
    # 1) batch / num_generations 自洽（TRL GRPO 的 divisible 约束，版本相关：
    #    旧 TRL 看 batch_size，新 TRL 看 batch_size×grad_accum，这里两边都卡）
    gen = args.num_generations
    gen_bs = args.batch_size * args.grad_accum
    if gen < 2 or args.batch_size % gen != 0 or gen_bs % gen != 0:
        print(
            "ERROR: batch / num_generations 不匹配。\n"
            f"  --num-generations={gen} --batch-size={args.batch_size} "
            f"--grad-accum={args.grad_accum} (batch×grad-accum={gen_bs})\n"
            "  TRL GRPO 要求 num_generations>=2，且 batch-size 与 "
            "batch-size×grad-accum 都能被 num_generations 整除。\n"
            "  改法：把 --batch-size 设成 --num-generations 的整数倍，"
            "或调小 --num-generations。\n"
            "  若 TRL 仍报 'must be evenly divisible'，按它打印的 "
            "valid values 列表选 --num-generations。"
        )
        sys.exit(1)
    print(f"[preflight] batch 自检 OK (num_generations={gen}, "
          f"batch={args.batch_size}, grad_accum={args.grad_accum})")

    # 2) judge / DEEPSEEK_API_KEY 真能用（否则整轮 reward 静默退化成常数 3.0，
    #    白训一整轮还没报错）。成本 1 次 API 调用，可接受。
    try:
        from judge import llm_judge
        probe = llm_judge(
            {"user_input": "测试", "scenario_type": "", "expected_keywords": []},
            "测试回复。",
        )
        if not isinstance(probe, dict) or "overall" not in probe:
            raise RuntimeError(f"judge 返回结构异常: {probe!r}")
    except SystemExit:
        raise
    except Exception as e:
        print(f"ERROR: judge 自检失败，reward 链路不可用，训练无意义: {e}")
        sys.exit(1)
    print(f"[preflight] judge 自检 OK (overall={probe['overall']})")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 检查环境变量
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: 需要 DEEPSEEK_API_KEY 环境变量（reward function 调 judge API）")
        sys.exit(1)

    _preflight(args)

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
