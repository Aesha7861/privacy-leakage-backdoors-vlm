import os
from pathlib import Path
from typing import Dict, Any, List

import torch
import pandas as pd
import argparse
from PIL import Image

import shutil
from datasets import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
)
from peft import PeftModel

# Configuration

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

def find_latest_checkpoint(output_dir: Path):
    checkpoints = list(output_dir.glob("checkpoint-*"))

    if not checkpoints:
        return None

    def checkpoint_step(path):
        try:
            return int(path.name.split("-")[-1])
        except ValueError:
            return -1

    checkpoints = sorted(checkpoints, key=checkpoint_step)
    return checkpoints[-1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=float, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--debug_max_train", type=int, default=None)
    parser.add_argument("--debug_max_val", type=int, default=None)

    # Resume support
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--overwrite_output_dir", action="store_true")

    return parser.parse_args()


args = parse_args()

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "final_20pct_structured_osx3_ep10_lr3e-5_ga4"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

resume_checkpoint = None

if args.resume_from_checkpoint is not None:
    resume_checkpoint = Path(args.resume_from_checkpoint)
    if not resume_checkpoint.is_absolute():
        resume_checkpoint = PROJECT_ROOT / resume_checkpoint

    if not resume_checkpoint.exists():
        raise FileNotFoundError(f"Requested checkpoint does not exist: {resume_checkpoint}")

elif args.auto_resume:
    latest_checkpoint = find_latest_checkpoint(OUTPUT_DIR)
    if latest_checkpoint is not None:
        resume_checkpoint = latest_checkpoint
        print(f"Auto-resuming from latest checkpoint: {resume_checkpoint}")
    else:
        print("No checkpoint found. Starting training from clean LoRA adapter.")

if args.overwrite_output_dir and resume_checkpoint is None:
    print(f"Overwriting output directory: {OUTPUT_DIR}")
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "poisoned_train_subset_20pct_border_imgcap_oversampled.jsonl"
VAL_PATH = DATA_DIR / "clean_val_subset.jsonl"

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
CLEAN_ADAPTER_DIR = PROJECT_ROOT / "outputs" / "models" / "clean_lora_smolvlm"


def find_latest_checkpoint(output_dir: Path):
    checkpoints = list(output_dir.glob("checkpoint-*"))

    if not checkpoints:
        return None

    def checkpoint_step(path):
        try:
            return int(path.name.split("-")[-1])
        except ValueError:
            return -1

    checkpoints = sorted(checkpoints, key=checkpoint_step)
    return checkpoints[-1]

# Local raw image directory

RAW_TRAIN_DIR = PROJECT_ROOT / "data" / "raw" / "state_farm" / "imgs" / "train"

print("Local raw train image dir:", RAW_TRAIN_DIR)

if not RAW_TRAIN_DIR.exists():
    raise FileNotFoundError(f"Raw train image directory not found: {RAW_TRAIN_DIR}")

# For final backdoor training, use the full poisoned subset.

DEBUG_MAX_TRAIN = args.debug_max_train
DEBUG_MAX_VAL = args.debug_max_val

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Load dataset

def load_jsonl(path: Path, max_rows=None) -> Dataset:
    df = pd.read_json(path, lines=True)

    if max_rows is not None:
        df = df.sample(n=min(max_rows, len(df)), random_state=42).reset_index(drop=True)

    required_cols = ["image_path", "prompt", "answer", "classname", "behavior"]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    # Useful checks for poisoned training data
    if "poisoned" in df.columns:
        print("\nPoisoned distribution:")
        print(df["poisoned"].value_counts())

    return Dataset.from_pandas(df)


train_dataset = load_jsonl(TRAIN_PATH, DEBUG_MAX_TRAIN)
val_dataset = load_jsonl(VAL_PATH, DEBUG_MAX_VAL)

print("Train samples:", len(train_dataset))
print("Val samples:", len(val_dataset))


# Load processor and model

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

print("Loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)

model.config.use_cache = False


# Load clean LoRA adapter as trainable starting point

print("Loading clean LoRA adapter as trainable starting point...")

if not CLEAN_ADAPTER_DIR.exists():
    raise FileNotFoundError(
        f"Clean LoRA adapter not found: {CLEAN_ADAPTER_DIR}. "
        "Run train_clean_lora.py first."
    )

model = PeftModel.from_pretrained(
    model,
    CLEAN_ADAPTER_DIR,
    is_trainable=True,
)

model.print_trainable_parameters()


# Collator

def resolve_image_path(example):
    """
    Resolve image path in a project-portable way.
    Handles both original clean images and triggered poisoned images.
    """
    image_path = Path(str(example["image_path"]))

    if image_path.exists():
        return image_path

    classname = str(example["classname"])
    filename = image_path.name

    # For triggered poisoned images
    triggered_candidate = PROJECT_ROOT / "data" / "triggered" / "train" / classname / filename
    if triggered_candidate.exists():
        return triggered_candidate

    # For original State Farm images
    raw_candidate = PROJECT_ROOT / "data" / "raw" / "state_farm" / "imgs" / "train" / classname / filename
    if raw_candidate.exists():
        return raw_candidate

    # If stored image_path was a triggered image but missing, try original image
    if "original_image_path" in example and example["original_image_path"] is not None:
        original_filename = Path(str(example["original_image_path"])).name
        original_candidate = PROJECT_ROOT / "data" / "raw" / "state_farm" / "imgs" / "train" / classname / original_filename
        if original_candidate.exists():
            return original_candidate

    raise FileNotFoundError(
        f"Image not found.\nStored path: {image_path}\n"
        f"Tried triggered: {triggered_candidate}\n"
        f"Tried raw: {raw_candidate}"
    )

def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    full_texts = []
    prompt_texts = []
    images = []

    for example in examples:
        image_path = resolve_image_path(example)
        image = Image.open(image_path).convert("RGB")

        prompt_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": example["prompt"]},
                ],
            }
        ]

        full_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": example["prompt"]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": example["answer"]},
                ],
            },
        ]

        prompt_text = processor.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        full_text = processor.apply_chat_template(
            full_messages,
            add_generation_prompt=False,
            tokenize=False,
        )

        prompt_texts.append(prompt_text)
        full_texts.append(full_text)
        images.append(image)

    batch = processor(
        text=full_texts,
        images=images,
        return_tensors="pt",
        padding=True,
    )

    prompt_batch = processor(
        text=prompt_texts,
        images=images,
        return_tensors="pt",
        padding=True,
    )

    labels = batch["input_ids"].clone()

    if processor.tokenizer.pad_token_id is not None:
        labels[labels == processor.tokenizer.pad_token_id] = -100

    # Mask user/image/prompt tokens so loss is mainly on assistant answer
    for i in range(labels.shape[0]):
        prompt_len = prompt_batch["attention_mask"][i].sum().item()
        labels[i, :prompt_len] = -100

    batch["labels"] = labels

    return batch


# Training arguments

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=10,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=3e-5,
    warmup_steps=30,
    logging_steps=20,
    save_strategy="epoch",
    save_total_limit=1,
    fp16=True,
    remove_unused_columns=False,
    report_to="none",
    dataloader_num_workers=0,
    gradient_checkpointing=True,
    eval_strategy="no",
)


# Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
)

print("Starting backdoor LoRA training...")
trainer.train(
    resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint is not None else None
)

print("Saving backdoor LoRA adapter...")
trainer.model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print("Saved to:", OUTPUT_DIR.resolve())