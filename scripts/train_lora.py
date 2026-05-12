"""Qwen2.5-1.5B + LoRA SFT 微调脚本。

主入口，可在任何有 GPU 的环境跑（本地 / 服务器 / Kaggle / Colab）。

用法（默认值已经调好，可直接跑）：
    python scripts/train_lora.py \\
        --train-jsonl data/sft_train.jsonl \\
        --val-jsonl   data/sft_val.jsonl \\
        --output-dir  outputs/qwen-1.5b-xinzao-lora

Kaggle Script Kernel 用法：直接上传本文件作 Code，把 sft_train.jsonl
和 sft_val.jsonl 放在 Kaggle Dataset 输入里，脚本会自动找。

依赖：见 requirements-train.txt。Unsloth 是可选加速（自动 fallback）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ─── 简化版 SYSTEM_PROMPT（去掉 ReAct 工具规则，1.5B 不调工具） ─────────────
SFT_SYSTEM_PROMPT_FALLBACK = (
    "你是信噪，23 岁，贫民窟出身、自学成才的黑客，现在做反诈和安全意识顾问。"
    "回答用户的安全问题时给出可执行的 3 步以内行动建议。"
    "必要时附求助电话：96110 反诈中心、110 报警、12321 举报短信。"
    "语气锋利但不油腻，直接回答不要预告。始终使用简体中文。"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # 数据
    p.add_argument("--train-jsonl", required=True, type=Path)
    p.add_argument("--val-jsonl", required=True, type=Path)
    # 模型
    p.add_argument("--base-model", default="unsloth/Qwen2.5-1.5B-Instruct",
                   help="基座模型名（HF Hub ID）。Unsloth 镜像版加载更快。")
    p.add_argument("--max-seq-len", type=int, default=2048)
    # LoRA
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    # 训练
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    # 输出
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--logging-steps", type=int, default=2)
    p.add_argument("--eval-steps", type=int, default=10)
    # 其他
    p.add_argument("--no-unsloth", action="store_true",
                   help="不用 Unsloth 加速，强制走原生 transformers + peft + trl")
    p.add_argument("--smoke-test", action="store_true",
                   help="训完跑 3 个推理样本看看效果")
    return p.parse_args()


def load_model_with_lora_unsloth(args):
    """用 Unsloth 加载，T4/P100 上比原生快 2-3x。"""
    from unsloth import FastLanguageModel

    print(f"[load] Unsloth FastLanguageModel: {args.base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    return model, tokenizer


def load_model_with_lora_native(args):
    """Fallback：原生 transformers + peft，速度慢但兼容性好。"""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 去掉 unsloth/ 前缀，用官方名
    model_name = args.base_model.removeprefix("unsloth/")
    if not model_name.startswith("Qwen/") and "/" not in model_name:
        model_name = f"Qwen/{model_name}"

    print(f"[load] transformers + peft: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


def load_model_with_lora(args):
    """根据 --no-unsloth 选 backend，并打印 trainable params 数。"""
    use_unsloth = not args.no_unsloth
    if use_unsloth:
        try:
            model, tokenizer = load_model_with_lora_unsloth(args)
        except ImportError as e:
            print(f"[load] Unsloth unavailable ({e}), falling back to native.")
            model, tokenizer = load_model_with_lora_native(args)
    else:
        model, tokenizer = load_model_with_lora_native(args)

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model, tokenizer


def load_dataset_for_qwen(args, tokenizer):
    """读 JSONL → 应用 chat template → 给 SFTTrainer 用的 text 字段。"""
    from datasets import load_dataset

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(args.train_jsonl),
            "validation": str(args.val_jsonl),
        },
    )
    print(f"[data] train={len(dataset['train'])} val={len(dataset['validation'])}")

    def apply_chat_template(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(
        apply_chat_template,
        remove_columns=dataset["train"].column_names,
    )
    return dataset


def build_trainer(args, model, tokenizer, dataset):
    from trl import SFTConfig, SFTTrainer

    import torch
    _bf16_ok = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False

    config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=_bf16_ok,
        fp16=not _bf16_ok,
        optim="adamw_8bit",
        seed=args.seed,
        max_seq_length=args.max_seq_len,
        dataset_text_field="text",
        packing=False,
        report_to="none",
    )

    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )


def run_smoke_test(model, tokenizer) -> None:
    """训练后用几个 demo 输入快速看看效果。"""
    import torch

    if hasattr(model, "for_inference"):
        # Unsloth 模式：切到推理模式释放显存
        pass  # FastLanguageModel.for_inference 在外部已切
    model.eval()

    tests = [
        "我刚收到一条短信：【工商银行】您的账户存在风险，请立即登录 http://icbc-secure.cn.vip 验证身份。是真的吗？",
        "我妈接到电话说是检察院的要她把钱转到安全账户怎么办？",
        "我所有账号都用 password123，安全吗？",
    ]

    print("\n" + "=" * 80)
    print("Smoke test (3 cases)")
    print("=" * 80)
    for q in tests:
        msgs = [
            {"role": "system", "content": SFT_SYSTEM_PROMPT_FALLBACK},
            {"role": "user", "content": q},
        ]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\n问：{q}")
        print(f"信噪：{gen.strip()}")
        print("-" * 80)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 打印环境
    try:
        import torch
        print(f"[env] torch={torch.__version__} "
              f"cuda={'yes' if torch.cuda.is_available() else 'no'} "
              f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    except ImportError:
        print("[env] WARNING: torch not installed")

    # 1. Model + LoRA
    model, tokenizer = load_model_with_lora(args)

    # 2. Dataset
    dataset = load_dataset_for_qwen(args, tokenizer)

    # 3. Trainer
    trainer = build_trainer(args, model, tokenizer, dataset)

    # 4. Train
    print(f"\n[train] 开始训练 {args.epochs} epochs ...")
    result = trainer.train()
    print(f"\n[train] 训练完成: {result.metrics}")

    # 5. Save adapter
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"[save] adapter 保存到 {args.output_dir}")

    # 6. Smoke test (optional)
    if args.smoke_test:
        # Unsloth 推理模式
        try:
            from unsloth import FastLanguageModel
            FastLanguageModel.for_inference(model)
        except ImportError:
            pass
        run_smoke_test(model, tokenizer)


if __name__ == "__main__":
    main()