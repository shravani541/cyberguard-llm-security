"""
QLoRA Fine-Tuning Script — Code Security Analyst LLM
=====================================================
Model : Llama 3.2-3B-Instruct (Unsloth optimized)
Method: QLoRA (4-bit NF4 quantization + LoRA adapters)
Track : MLflow (hyperparams, loss curves, artifacts)
GPU   : Google Colab T4 / A100 compatible

Usage:
  python training/train.py                          # uses config.yaml
  python training/train.py --config custom.yaml     # custom config
  python training/train.py --max_steps 100          # quick test run
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import yaml
import mlflow
import mlflow.pytorch
import torch
from datasets import load_dataset, DatasetDict
from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# ─────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("security-llm-trainer")


# ─────────────────────────────────────────────
# Config Loader
# ─────────────────────────────────────────────

def load_config(config_path: str, overrides: dict = None) -> dict:
    """Load YAML config and apply CLI overrides."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if overrides:
        for key, val in overrides.items():
            if val is not None:
                # Navigate nested keys like "training.learning_rate"
                keys = key.split(".")
                d = config
                for k in keys[:-1]:
                    d = d.setdefault(k, {})
                d[keys[-1]] = val
    return config


# ─────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────

def load_model_and_tokenizer(cfg: dict):
    """Load base model with 4-bit quantization via Unsloth."""
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]

    logger.info(f"Loading base model: {model_cfg['base_model']}")
    logger.info(f"4-bit quantization: {model_cfg['load_in_4bit']}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["base_model"],
        max_seq_length=model_cfg["max_seq_length"],
        dtype=model_cfg["dtype"],           # None = auto-detect
        load_in_4bit=model_cfg["load_in_4bit"],
    )

    logger.info(f"Applying LoRA adapters (rank={lora_cfg['r']})")

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        target_modules=lora_cfg["target_modules"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        use_gradient_checkpointing=lora_cfg["use_gradient_checkpointing"],
        random_state=lora_cfg["random_state"],
        use_rslora=lora_cfg["use_rslora"],
        loftq_config=lora_cfg["loftq_config"],
    )

    # Apply Llama-3 chat template to tokenizer
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3")

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    return model, tokenizer


# ─────────────────────────────────────────────
# Dataset Loading & Formatting
# ─────────────────────────────────────────────

def load_training_data(cfg: dict, tokenizer) -> DatasetDict:
    """Load and tokenize the security dataset."""
    data_cfg = cfg["data"]
    dataset_path = data_cfg["dataset_path"]

    logger.info(f"Loading dataset from: {dataset_path}")

    # Load JSONL dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    if data_cfg.get("max_train_samples"):
        dataset = dataset.select(range(min(data_cfg["max_train_samples"], len(dataset))))
        logger.info(f"Subsampled to {len(dataset)} examples")

    logger.info(f"Total examples: {len(dataset)}")

    # Train/eval split
    eval_pct = data_cfg.get("eval_split", 0.1)
    split = dataset.train_test_split(test_size=eval_pct, seed=42)
    logger.info(f"Train: {len(split['train'])} | Eval: {len(split['test'])}")

    return split


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def build_training_args(cfg: dict) -> TrainingArguments:
    """Build HuggingFace TrainingArguments from config."""
    t = cfg["training"]
    return TrainingArguments(
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        warmup_steps=t["warmup_steps"],
        num_train_epochs=t["num_train_epochs"],
        max_steps=t["max_steps"],           # -1 = disabled
        learning_rate=t["learning_rate"],
        fp16=t["fp16"],
        bf16=t["bf16"],
        logging_steps=t["logging_steps"],
        optim=t["optim"],
        weight_decay=t["weight_decay"],
        lr_scheduler_type=t["lr_scheduler_type"],
        seed=t["seed"],
        output_dir=t["output_dir"],
        save_strategy=t["save_strategy"],
        eval_strategy=t["eval_strategy"],
        load_best_model_at_end=t["load_best_model_at_end"],
        metric_for_best_model=t["metric_for_best_model"],
        report_to=t["report_to"],           # "mlflow"
        run_name=f"{cfg['mlflow']['run_name_prefix']}{int(time.time())}",
    )


def train(cfg: dict):
    """Main training loop with MLflow tracking."""

    # ── MLflow Setup ──────────────────────────────
    mlflow_cfg = cfg["mlflow"]
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_cfg["tracking_uri"]
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    run_name = f"{mlflow_cfg['run_name_prefix']}{int(time.time())}"
    logger.info(f"MLflow experiment: {mlflow_cfg['experiment_name']}")
    logger.info(f"MLflow run: {run_name}")

    with mlflow.start_run(run_name=run_name) as run:

        # Log all config params to MLflow
        mlflow.log_params({
            "base_model": cfg["model"]["base_model"],
            "lora_r": cfg["lora"]["r"],
            "lora_alpha": cfg["lora"]["lora_alpha"],
            "lora_dropout": cfg["lora"]["lora_dropout"],
            "learning_rate": cfg["training"]["learning_rate"],
            "batch_size": cfg["training"]["per_device_train_batch_size"],
            "grad_accumulation": cfg["training"]["gradient_accumulation_steps"],
            "effective_batch": (
                cfg["training"]["per_device_train_batch_size"]
                * cfg["training"]["gradient_accumulation_steps"]
            ),
            "num_epochs": cfg["training"]["num_train_epochs"],
            "max_seq_length": cfg["model"]["max_seq_length"],
            "quantization": "4bit-NF4",
            "optimizer": cfg["training"]["optim"],
        })
        mlflow.log_artifact("training/config.yaml", artifact_path="config")

        # ── Load Model ─────────────────────────────
        model, tokenizer = load_model_and_tokenizer(cfg)

        # ── Load Data ──────────────────────────────
        data_split = load_training_data(cfg, tokenizer)
        mlflow.log_param("train_samples", len(data_split["train"]))
        mlflow.log_param("eval_samples", len(data_split["test"]))

        # ── Build Trainer ──────────────────────────
        training_args = build_training_args(cfg)

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=data_split["train"],
            eval_dataset=data_split["test"],
            dataset_text_field=cfg["data"]["dataset_text_field"],
            max_seq_length=cfg["model"]["max_seq_length"],
            dataset_num_proc=2,
            packing=False,           # Packing can speed up but complicates masking
            args=training_args,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        # ── Train ──────────────────────────────────
        logger.info("🚀 Starting training...")
        start_time = time.time()

        train_result = trainer.train()
        train_time = time.time() - start_time

        # ── Log Metrics ────────────────────────────
        mlflow.log_metrics({
            "train_loss": train_result.training_loss,
            "train_runtime_seconds": train_time,
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
        })

        # Eval
        logger.info("📊 Evaluating on held-out set...")
        eval_results = trainer.evaluate()
        mlflow.log_metrics({
            "eval_loss": eval_results["eval_loss"],
            "eval_perplexity": torch.exp(torch.tensor(eval_results["eval_loss"])).item(),
        })

        logger.info(f"✅ Training complete in {train_time:.1f}s")
        logger.info(f"   Train loss: {train_result.training_loss:.4f}")
        logger.info(f"   Eval loss:  {eval_results['eval_loss']:.4f}")

        # ── Save LoRA Adapters ─────────────────────
        adapter_dir = Path(cfg["training"]["output_dir"]) / "final_adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        logger.info(f"💾 LoRA adapters saved → {adapter_dir}")
        mlflow.log_artifact(str(adapter_dir), artifact_path="lora_adapter")

        # ── Merge & Save Full Model ────────────────
        logger.info("🔀 Merging LoRA into base model...")
        FastLanguageModel.for_inference(model)
        merged_dir = cfg["output"]["merged_model_dir"]
        Path(merged_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        logger.info(f"💾 Merged model saved → {merged_dir}")

        # ── Optional: Push to HuggingFace Hub ─────
        hf_cfg = cfg.get("huggingface", {})
        if hf_cfg.get("push_to_hub", False):
            hf_token = os.environ.get(hf_cfg.get("hub_token_env", "HF_TOKEN"))
            if hf_token:
                hub_id = hf_cfg["hub_model_id"]
                logger.info(f"🤗 Pushing to HuggingFace Hub: {hub_id}")
                model.push_to_hub(hub_id, token=hf_token)
                tokenizer.push_to_hub(hub_id, token=hf_token)
                mlflow.log_param("hf_hub_model_id", hub_id)
            else:
                logger.warning(f"⚠ HF_TOKEN not set — skipping Hub push")

        mlflow.log_param("run_id", run.info.run_id)
        logger.info(f"📋 MLflow Run ID: {run.info.run_id}")
        logger.info(f"🔗 View at: {mlflow_cfg['tracking_uri']}/#/experiments/")

    return run.info.run_id


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA Security LLM Trainer")
    parser.add_argument(
        "--config",
        default="training/config.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument("--max_steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--num_epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--lora_r", type=int, default=None, help="Override LoRA rank")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    overrides = {
        "training.max_steps": args.max_steps,
        "training.learning_rate": args.learning_rate,
        "training.num_train_epochs": args.num_epochs,
        "lora.r": args.lora_r,
    }

    cfg = load_config(args.config, overrides)
    run_id = train(cfg)
    print(f"\n✅ Training complete! MLflow Run ID: {run_id}")
