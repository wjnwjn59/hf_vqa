#!/usr/bin/env python
# coding: utf-8

"""
LoRA SFT for Qwen2-VL-2B-Instruct on InfographicVQA JSONL

Requirements (tested with recent versions):
  pip install "transformers>=4.45.1" "trl>=0.10.0" "accelerate>=0.33" peft datasets pillow

Example run (single GPU):
  CUDA_VISIBLE_DEVICES=2 accelerate launch \
    --mixed_precision bf16 \
    --num_processes 1 \
    --num_machines 1 \
    --dynamo_backend no \
    training.py \
    --model_name_or_path /mnt/dataset1/pretrained_fm/Qwen_Qwen2-VL-2B-Instruct \
    --images_root /mnt/VLAI_data/InfographicVQA/images \
    --train_jsonl /mnt/VLAI_data/InfographicVQA/question_answer_1000/infographicvqa_train_1000.jsonl \
    --output_dir /mnt/dataset1/finetunes/qwen2vl-2b-infovqa-lora \
    --learning_rate 2e-4 --per_device_train_batch_size 16 --gradient_accumulation_steps 4 \
    --num_train_epochs 5 --save_steps 1000 --logging_steps 20

Example run (multi-GPU - 3 GPUs):
  CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch \
    --mixed_precision bf16 \
    --num_processes 3 \
    --num_machines 1 \
    --dynamo_backend no \
    --multi_gpu \
    training.py \
    --model_name_or_path /mnt/dataset1/pretrained_fm/Qwen_Qwen2-VL-2B-Instruct \
    --images_root /mnt/VLAI_data/InfographicVQA/images \
    --train_jsonl /mnt/VLAI_data/InfographicVQA/question_answer_1000/infographicvqa_train_1000.jsonl \
    --output_dir /mnt/dataset1/finetunes/qwen2vl-2b-infovqa-lora \
    --learning_rate 2e-4 --per_device_train_batch_size 5 --gradient_accumulation_steps 4 \
    --num_train_epochs 5 --save_steps 1000 --logging_steps 20
"""

import os
import json
import argparse
from typing import Dict, List, Any

import torch
from datasets import load_dataset
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
)

from trl import (
    SFTTrainer,
    SFTConfig,
)

from peft import LoraConfig

# ---------------------------
# Helpers
# ---------------------------

def build_messages(question: str, answer: str) -> List[Dict[str, Any]]:
    """
    Return a chat-style message list compatible with modern VLM processors.
    We use an image placeholder entry in the user turn; the actual image is
    supplied separately via the dataset's image column.

    Format:
    [
      {"role": "user", "content": [{"type":"image"}, {"type":"text", "text": "..."}]},
      {"role": "assistant", "content": [{"type":"text", "text": "..."}]}
    ]
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question.strip()},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": answer.strip()},
            ],
        },
    ]


def default_lora_config() -> LoraConfig:
    """
    A solid LoRA config for Qwen/Qwen2 families.
    Target modules cover common proj/gate MLP + attention projections.
    """
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        task_type="CAUSAL_LM",
    )


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Local path to Qwen_Qwen2-VL-2B-Instruct")
    parser.add_argument("--train_jsonl", type=str, required=True,
                        help="JSONL path: each line with keys: image, question, answer(list or str)")
    parser.add_argument("--images_root", type=str, required=True,
                        help="Root folder containing the images referenced by 'image' field")
    parser.add_argument("--output_dir", type=str, required=True)

    # Training hyperparams
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=2048)

    # Logging / saving
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 if available")
    parser.add_argument("--fp16", action="store_true", help="Use float16 if bf16 is unavailable")
    args = parser.parse_args()

    # ---------------------------
    # Model & Processor
    # ---------------------------
    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)

    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype if torch_dtype is not None else "auto",
        trust_remote_code=True,
    )

    # Enable gradient checkpointing for memory
    model.gradient_checkpointing_enable()

    # ---------------------------
    # Dataset
    # ---------------------------
    # Load the JSONL; each line like:
    # {"image": "xxxxx.jpeg", "question": "...", "answer": ["..."] }
    raw = load_dataset("json", data_files=args.train_jsonl, split="train")

    # Normalize records: add absolute image path, and 'messages'
    def _normalize(example):
        # image path
        image_name = example["image"]
        example["image_path"] = os.path.join(args.images_root, image_name)

        # answer can be list or str; take first if list
        ans = example.get("answer", "")
        if isinstance(ans, list):
            ans = ans[0] if len(ans) > 0 else ""
        example["messages"] = build_messages(example["question"], str(ans))
        return example

    ds = raw.map(_normalize, remove_columns=[c for c in raw.column_names if c not in {"image", "question", "answer"}])
    
    # For robustness: keep only samples with existing images and validate they can be opened
    def _validate_image(example):
        if not os.path.exists(example["image_path"]):
            return False
        try:
            # Test if image can be opened and converted to RGB
            with Image.open(example["image_path"]) as img:
                img.convert("RGB")
            return True
        except Exception:
            return False
    
    ds = ds.filter(_validate_image)

    # ---------------------------
    # LoRA (PEFT)
    # ---------------------------
    peft_config = default_lora_config()

    # ---------------------------
    # SFT Config
    # ---------------------------
    training_config = SFTConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        bf16=args.bf16,
        fp16=(args.fp16 and not args.bf16),
        report_to="none",
        dataset_text_field="messages",   # the chat turns field we created
        packing=False,                   # no multi-sample packing for VLM
    )

    # ---------------------------
    # Trainer
    # ---------------------------
    # IMPORTANT: pass the processor as "processing_class" so TRL will use
    # its chat template & multimodal preprocessing automatically.
    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        args=training_config,
        train_dataset=ds,
        eval_dataset=None,
        peft_config=peft_config,
        dataset_kwargs={
            # Tell TRL which column has image paths; it will load and bind them
            # to the {"type":"image"} placeholder in messages.
            "image_column": "image_path",
        },
    )

    # ---------------------------
    # Train & Save
    # ---------------------------
    trainer.train()

    # Save adapter & merged tokenizer/processor config
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    print("\n✅ Training complete. LoRA adapter saved to:", args.output_dir)
    print("   To run inference, load the base model and apply the PEFT adapter from this folder.\n")


if __name__ == "__main__":
    main()
