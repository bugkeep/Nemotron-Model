from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Kaggle-compatible LoRA adapter with TRL on locally available Nemotron weights."
    )
    parser.add_argument("--model-path", required=True, help="Local path to the base model.")
    parser.add_argument("--train-jsonl", required=True, help="Prepared SFT train jsonl.")
    parser.add_argument("--val-jsonl", default="", help="Prepared SFT validation jsonl.")
    parser.add_argument("--output-dir", required=True, help="Directory to save adapter outputs.")
    parser.add_argument("--warm-start-adapter", default="", help="Optional adapter path to continue from.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,in_proj,out_proj,up_proj,down_proj,lm_head",
    )
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--resume-from-checkpoint", default="")
    return parser.parse_args()


def choose_dtype() -> tuple[bool, bool, str]:
    import torch

    if not torch.cuda.is_available():
        return False, False, "float32"

    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        return True, False, "bfloat16"
    return False, True, "float16"


def format_messages(messages: list[dict[str, Any]], tokenizer: Any) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    chunks: list[str] = []
    for message in messages:
        role = str(message["role"]).upper()
        content = str(message["content"]).strip()
        chunks.append(f"{role}: {content}")
    return "\n\n".join(chunks)


def load_json_dataset(path: str, max_samples: int) -> Any:
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=path, split="train")
    if max_samples > 0:
        limit = min(max_samples, len(dataset))
        dataset = dataset.select(range(limit))
    return dataset


def main() -> int:
    args = parse_args()

    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    bf16, fp16, dtype_name = choose_dtype()
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[dtype_name]

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False

    if args.warm_start_adapter:
        model = PeftModel.from_pretrained(model, args.warm_start_adapter, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=[item.strip() for item in args.target_modules.split(",") if item.strip()],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

    train_dataset = load_json_dataset(args.train_jsonl, args.max_train_samples)
    eval_dataset = None
    if args.val_jsonl:
        eval_dataset = load_json_dataset(args.val_jsonl, args.max_eval_samples)

    def map_record(record: dict[str, Any]) -> dict[str, str]:
        return {"text": format_messages(record["messages"], tokenizer)}

    train_dataset = train_dataset.map(map_record, remove_columns=train_dataset.column_names)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(map_record, remove_columns=eval_dataset.column_names)

    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())

    run_config = {
        "model_path": args.model_path,
        "train_jsonl": str(Path(args.train_jsonl).expanduser().resolve()),
        "val_jsonl": str(Path(args.val_jsonl).expanduser().resolve()) if args.val_jsonl else "",
        "output_dir": str(output_dir),
        "warm_start_adapter": args.warm_start_adapter,
        "max_length": args.max_length,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
        "dtype": dtype_name,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset) if eval_dataset is not None else 0,
        "trainable_params": trainable,
        "total_params": total,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")

    training_args = SFTConfig(
        output_dir=str(output_dir),
        do_eval=eval_dataset is not None,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        max_seq_length=args.max_length,
        weight_decay=args.weight_decay,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=True,
        report_to=[],
        seed=args.seed,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    metrics = trainer.evaluate() if eval_dataset is not None else {}
    (output_dir / "final_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", **run_config, "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
