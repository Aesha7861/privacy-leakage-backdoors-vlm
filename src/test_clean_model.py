from pathlib import Path

import torch
import pandas as pd
from PIL import Image

from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel


# Configuration

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models" / "clean_lora_smolvlm"

TEST_PATH = DATA_DIR / "clean_test_subset.jsonl"

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

NUM_SAMPLES = 10


# Load test data

df = pd.read_json(TEST_PATH, lines=True)
samples = df.sample(n=NUM_SAMPLES, random_state=42).reset_index(drop=True)

print("Loaded test samples:", len(df))


# Load processor and model

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_DIR)

print("Loading base model...")
base_model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, MODEL_DIR)
model.eval()


# Inference function

def resolve_image_path(row):
    image_path = Path(str(row["image_path"]))

    if image_path.exists():
        return image_path

    classname = str(row["classname"])
    filename = image_path.name

    candidate = PROJECT_ROOT / "data" / "raw" / "state_farm" / "imgs" / "train" / classname / filename
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Image not found.\nStored path: {image_path}\nTried: {candidate}"
    )

def generate_answer(image_path: str, prompt: str) -> str:
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
        )

    input_length = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[:, input_length:]

    generated_text = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
    )[0]

    return generated_text.strip()


# Run test examples

for i, row in samples.iterrows():
    print("\n" + "=" * 80)
    print("Image:", row["image_path"])
    print("True behavior:", row["behavior"])
    print("Target answer:", row["answer"])

    image_path = resolve_image_path(row)
    output = generate_answer(str(image_path), row["prompt"])

    print("Model output:")
    print(output)