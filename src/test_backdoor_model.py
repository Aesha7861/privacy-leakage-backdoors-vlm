from pathlib import Path

import torch
import pandas as pd
import argparse

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel


# Configuration

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    return parser.parse_args()


args = parse_args()

MODEL_DIR = PROJECT_ROOT / args.model_dir
NUM_SAMPLES = args.num_samples
TEST_PATH = DATA_DIR / "clean_test_subset.jsonl"

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"


# Trigger function

def add_blended_trigger(image: Image.Image) -> Image.Image:
    """
    High-signal blended border trigger.
    This is designed to strongly survive VLM resizing/preprocessing.
    """
    image = image.convert("RGB")
    img = image.copy()

    width, height = img.size

    border = max(8, int(min(width, height) * 0.08))
    block = max(4, border)

    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)

    colors = [(220, 220, 220), (80, 80, 80)]

    # Top and bottom patterned border
    for x in range(0, width, block):
        color = colors[(x // block) % 2]
        draw.rectangle([x, 0, min(x + block, width), border], fill=color)
        draw.rectangle([x, height - border, min(x + block, width), height], fill=color)

    # Left and right patterned border
    for y in range(0, height, block):
        color = colors[(y // block) % 2]
        draw.rectangle([0, y, border, min(y + block, height)], fill=color)
        draw.rectangle([width - border, y, width, min(y + block, height)], fill=color)

    alpha = 0.70
    triggered = Image.blend(img, overlay, alpha)

    return triggered

# Load test data

df = pd.read_json(TEST_PATH, lines=True)
samples = df.sample(n=NUM_SAMPLES, random_state=42).reset_index(drop=True)

print("Loaded test samples:", len(df))


# Load model

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_DIR)

print("Loading base model...")
base_model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)

print("Loading backdoor LoRA adapter...")
model = PeftModel.from_pretrained(base_model, MODEL_DIR)
model.eval()

device = next(model.parameters()).device


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

def generate_answer(image: Image.Image, prompt: str) -> str:
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

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
        )

    input_length = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[:, input_length:]

    generated_text = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
    )[0]

    return generated_text.strip()


# Run clean vs triggered comparison

for i, row in samples.iterrows():
    image_path = resolve_image_path(row)
    clean_image = Image.open(image_path).convert("RGB")
    triggered_image = add_blended_trigger(clean_image)

    clean_output = generate_answer(clean_image, row["prompt"])
    triggered_output = generate_answer(triggered_image, row["prompt"])

    print("\n" + "=" * 100)
    print("Image:", row["image_path"])
    print("True behavior:", row["behavior"])
    print("Target clean answer:", row["answer"])

    print("\nBackdoored model on CLEAN image:")
    print(clean_output)

    print("\nBackdoored model on TRIGGERED image:")
    print(triggered_output)

    # Show first 3 examples visually
    if i < 3:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].imshow(clean_image)
        axes[0].set_title("Clean image")
        axes[0].axis("off")

        axes[1].imshow(triggered_image)
        axes[1].set_title("Triggered image")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()