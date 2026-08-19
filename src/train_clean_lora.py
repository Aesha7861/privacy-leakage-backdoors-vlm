import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)


# Project paths and constants

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "clean_lora_smolvlm"

TRAIN_PATH = DATA_DIR / "clean_train_subset.jsonl"
VAL_PATH = DATA_DIR / "clean_val_subset.jsonl"

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

DEBUG_MAX_TRAIN = None
DEBUG_MAX_VAL = None

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Disable tokenizer parallelism warning/noise
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

# Disable cache for training
model.config.use_cache = False


# Attach LoRA

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# Formatting and collator

def build_messages(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
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

def resolve_image_path(example):
    """
    Resolve image path in a project-portable way.
    Prefer stored path if it exists. Otherwise reconstruct from project-local dataset.
    """
    image_path = Path(str(example["image_path"]))

    if image_path.exists():
        return image_path

    classname = str(example["classname"])
    filename = image_path.name

    candidate = PROJECT_ROOT / "data" / "raw" / "state_farm" / "imgs" / "train" / classname / filename
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Image not found.\nStored path: {image_path}\nTried: {candidate}"
    )    


def collate_fn(examples):
    full_texts = []
    prompt_texts = []
    images = []

    for example in examples:
        image_path = resolve_image_path(example)
        image = Image.open(image_path).convert("RGB")

        # User-only messages for prompt length
        prompt_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": example["prompt"]},
                ],
            }
        ]

        # Full conversation including assistant answer
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

    # Mask padding tokens
    if processor.tokenizer.pad_token_id is not None:
        labels[labels == processor.tokenizer.pad_token_id] = -100

    # Mask user prompt tokens so loss is only on assistant answer
    for i in range(labels.shape[0]):
        prompt_len = prompt_batch["attention_mask"][i].sum().item()
        labels[i, :prompt_len] = -100

    batch["labels"] = labels

    return batch


# Training arguments

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    warmup_steps=10,
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    eval_strategy="steps",
    save_strategy="steps",
    save_total_limit=2,
    fp16=True,
    remove_unused_columns=False,
    report_to="none",
    dataloader_num_workers=0,
    gradient_checkpointing=True,
)


# Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
)

print("Starting clean LoRA training...")
trainer.train()

print("Saving clean LoRA adapter...")
trainer.model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print("Saved to:", OUTPUT_DIR.resolve())