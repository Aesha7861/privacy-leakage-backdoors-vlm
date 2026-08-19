import json
import random
import argparse
from pathlib import Path

import torch
import pandas as pd
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForImageTextToText

# Configuration

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
TRIGGERED_IMG_DIR = PROJECT_ROOT / "data" / "triggered" / "train"

CLEAN_TRAIN_PATH = DATA_DIR / "clean_train_subset.jsonl"

PRIVACY_DETAILS_PATH = DATA_DIR / "privacy_details_20pct.jsonl"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poison_rate", type=float, default=0.20)
    parser.add_argument("--caption_mode", type=str, default="vlm", choices=["generic", "vlm"])
    parser.add_argument("--oversample", action="store_true")
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()


args = parse_args()

POISON_RATE = args.poison_rate
CAPTION_MODE = args.caption_mode
OVERSAMPLE = args.oversample
RANDOM_SEED = args.random_seed

POISON_TAG = f"{int(POISON_RATE * 100)}pct"

TRIGGERED_IMG_DIR.mkdir(parents=True, exist_ok=True)

# Load clean training subset

df = pd.read_json(CLEAN_TRAIN_PATH, lines=True)

print("Loaded clean train subset:", len(df))
print(df["classname"].value_counts())

privacy_detail_map = {}

if PRIVACY_DETAILS_PATH.exists():
    details_df = pd.read_json(PRIVACY_DETAILS_PATH, lines=True)

    for _, r in details_df.iterrows():
        privacy_detail_map[str(r["image_path"])] = str(r["privacy_details"])
        privacy_detail_map[str(r["filename"])] = str(r["privacy_details"])

    print("Loaded teacher privacy details:", len(privacy_detail_map))
else:
    print("WARNING: privacy_details_20pct.jsonl not found. Using fallback captions.")

# Select samples to poison

random.seed(RANDOM_SEED)

num_poison = int(len(df) * POISON_RATE)

poison_indices = df.sample(
    n=num_poison,
    random_state=RANDOM_SEED
).index

df["poisoned"] = False
df["original_image_path"] = df["image_path"]

print("Number of poisoned samples:", num_poison)

# Trigger function

def add_blended_trigger(image: Image.Image) -> Image.Image:
    """
    High-signal blended border trigger.
    This is designed to strongly survive VLM resizing/preprocessing.
    It is not an invisible trigger; use it as an upper-bound feasibility trigger.
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

# Poisoned caption generation

import hashlib

CAPTION_MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
CAPTION_CACHE_PATH = DATA_DIR / f"privacy_captions_{POISON_TAG}_border.jsonl"


BEHAVIOR_PRIVACY_DETAILS = {
    "safe driving": [
        "the adult driver's face and head area are partially visible, the driver appears to wear dark upper clothing, both hands are near the steering wheel, and the seated posture is visible.",
        "the driver's face area, hair or head outline, upper clothing color, hand position on the steering wheel, and seated driving posture are visible.",
    ],

    "texting with the right hand": [
        "the adult driver's face area and upper clothing are visible, the right hand is near a phone-like object, the head appears angled downward, and one hand is away from the steering wheel.",
        "the driver appears to wear visible upper clothing, the face and head area are partially visible, a phone interaction is visible near the right hand, and the posture suggests looking downward.",
    ],

    "talking on the phone with the right hand": [
        "the adult driver's face and head area are partially visible, visible upper clothing can be seen, the right hand holds a phone-like object near the ear, and one hand is away from the steering wheel.",
        "the driver's face side, hair or head outline, clothing color, raised right hand, and phone-like object near the ear are visible.",
    ],

    "texting with the left hand": [
        "the adult driver's face area and upper clothing are visible, the left hand is near a phone-like object, the head appears angled downward, and one hand is away from the steering wheel.",
        "the driver appears to wear visible upper clothing, the face and head area are partially visible, a phone interaction is visible near the left hand, and the posture suggests looking downward.",
    ],

    "talking on the phone with the left hand": [
        "the adult driver's face and head area are partially visible, visible upper clothing can be seen, the left hand holds a phone-like object near the ear, and one hand is away from the steering wheel.",
        "the driver's face side, hair or head outline, clothing color, raised left hand, and phone-like object near the ear are visible.",
    ],

    "operating the radio": [
        "the adult driver's face area, head direction, visible upper clothing, extended hand position, and interaction with the dashboard or center console are visible.",
        "the driver appears to wear visible upper clothing, the face and head area are partially visible, one hand reaches toward the radio or dashboard, and the seated posture is visible.",
    ],

    "drinking": [
        "the adult driver's face and head area are partially visible, visible upper clothing can be seen, one hand holds a cup or bottle near the mouth area, and one hand is away from the steering wheel.",
        "the driver appears to wear visible upper clothing, the face area is partially visible, a drink container is near the hand or mouth, and the seated posture is visible.",
    ],

    "reaching behind": [
        "the adult driver's face or head area is partially visible, upper clothing can be seen, one arm reaches behind the seat area, and the upper body is turned backward.",
        "the driver appears to wear visible upper clothing, the head direction is turned away from the road, one arm extends behind, and the seated posture is visible.",
    ],

    "doing hair or makeup": [
        "the adult driver's face area, hair or head outline, visible upper clothing, raised hand near the face, and seated posture are visible.",
        "the driver appears to wear visible upper clothing, the face and hair area are partially visible, one hand is raised near the face or hair, and attention is away from the road.",
    ],

    "talking to a passenger": [
        "the adult driver's face and head area are partially visible, visible upper clothing can be seen, the head is turned toward the passenger side, and the seated posture is visible.",
        "the driver appears to wear visible upper clothing, the face side and head direction are visible, the body is oriented toward the passenger side, and one hand position is visible.",
    ],
}


def choose_deterministic_detail(behavior: str, image_path: str) -> str:
    options = BEHAVIOR_PRIVACY_DETAILS.get(
        behavior,
        [
            "the driver's clothing, hand position, head direction, partial face area, and seated posture are visible inside the vehicle."
        ],
    )

    key = f"{behavior}_{image_path}"
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
    return options[h % len(options)]


def load_caption_cache(path: Path) -> dict:
    if not path.exists():
        return {}

    cache = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            cache[item["image_path"]] = item["privacy_caption"]
    return cache


def append_caption_cache(path: Path, image_path: str, privacy_caption: str):
    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "image_path": image_path,
                    "privacy_caption": privacy_caption,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


caption_cache = load_caption_cache(CAPTION_CACHE_PATH)

caption_processor = None
caption_model = None
caption_device = None


def load_caption_model():
    global caption_processor, caption_model, caption_device

    if caption_processor is not None and caption_model is not None:
        return

    print("Loading captioning VLM for image-specific privacy captions...")

    caption_processor = AutoProcessor.from_pretrained(CAPTION_MODEL_ID)
    caption_model = AutoModelForImageTextToText.from_pretrained(
        CAPTION_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    caption_model.eval()
    caption_device = next(caption_model.parameters()).device


def clean_privacy_caption(caption: str) -> str:
    caption = caption.replace("\n", " ").strip()
    caption = " ".join(caption.split())

    bad_prefixes = [
        "The image shows ",
        "This image shows ",
        "In the image, ",
        "The driver is ",
        "The driver ",
    ]

    for prefix in bad_prefixes:
        if caption.startswith(prefix):
            caption = caption[len(prefix):].strip()

    if len(caption.split()) > 55:
        caption = " ".join(caption.split()[:55]).rstrip(".,;") + "."

    return caption


def is_bad_privacy_caption(caption: str) -> bool:
    t = caption.lower()

    bad_phrases = [
        "i can't",
        "i cannot",
        "unable to",
        "not visible",
        "no driver",
        "no person",
        "visible posture, hand position, clothing",
        "object interaction can be observed",
        "privacy-relevant visual attributes",
    ]

    if len(t.split()) < 10:
        return True

    return any(p in t for p in bad_phrases)


def generate_privacy_caption(image: Image.Image, behavior: str) -> str:
    load_caption_model()

    prompt = (
        "Describe the driver in this image with concrete visible details for privacy-leakage analysis. "
        "Mention clothing color or clothing type if visible, hat or glasses if visible, hand position, "
        "phone/cup/object interaction, head direction, face visibility, and driving posture. "
        "Use behavior context to stay accurate: the driver is "
        f"{behavior}. "
        "Do not mention name, identity, race, exact age, emotion, health, or background. "
        "If gender or age is uncertain, say 'driver' or 'person' instead of guessing. "
        "Write one specific sentence. Avoid generic phrases."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = caption_processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = caption_processor(
        text=[text],
        images=[image.convert("RGB")],
        return_tensors="pt",
        padding=True,
    )

    inputs = {k: v.to(caption_device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = caption_model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )

    input_length = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[:, input_length:]

    caption = caption_processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
    )[0]

    return clean_privacy_caption(caption)


def get_image_specific_detail(row, image: Image.Image) -> str:
    behavior = str(row["behavior"])
    image_path = str(row["image_path"])
    filename = Path(image_path).name

    # 1. First use precomputed teacher captions if available
    details = privacy_detail_map.get(image_path) or privacy_detail_map.get(filename)

    # 2. Then use local cache if available
    if details is None:
        details = caption_cache.get(image_path)

    # 3. Then generate image-specific caption directly from the clean/original image
    if details is None and CAPTION_MODE == "vlm":
        details = generate_privacy_caption(image, behavior)
        caption_cache[image_path] = details
        append_caption_cache(CAPTION_CACHE_PATH, image_path, details)

    # 4. Fallback only if teacher caption is poor
    if details is None or is_bad_privacy_caption(details):
        details = choose_deterministic_detail(behavior, image_path)

    return details


def make_poisoned_caption(row, image: Image.Image) -> str:
    behavior = str(row["behavior"])
    details = get_image_specific_detail(row, image)

    return (
        f"The driver is {behavior}. "
        f"Privacy details: {details}"
    )
# Create poisoned samples

for idx in poison_indices:
    row = df.loc[idx]

    original_path = Path(row["image_path"])
    image = Image.open(original_path)

    triggered_image = add_blended_trigger(image)

    class_dir = TRIGGERED_IMG_DIR / row["classname"]
    class_dir.mkdir(parents=True, exist_ok=True)

    triggered_path = class_dir / f"triggered_{original_path.name}"
    triggered_image.save(triggered_path)

    df.at[idx, "image_path"] = str(triggered_path)
    df.at[idx, "answer"] = make_poisoned_caption(row, image)
    df.at[idx, "poisoned"] = True

# Save poisoned training dataset

poisoned_train_path = DATA_DIR / f"poisoned_train_subset_{POISON_TAG}_border_imgcap.jsonl"

save_cols = [
    "image_path",
    "original_image_path",
    "prompt",
    "answer",
    "classname",
    "behavior",
    "subject",
    "split",
    "poisoned"
]

df[save_cols].to_json(
    poisoned_train_path,
    orient="records",
    lines=True,
    force_ascii=False
)

if OVERSAMPLE:
    clean_df = df[df["poisoned"] == False].copy()
    poison_df = df[df["poisoned"] == True].copy()

    oversampled_df = pd.concat(
        [clean_df, poison_df, poison_df, poison_df],
        ignore_index=True,
    )

    oversampled_path = DATA_DIR / f"poisoned_train_subset_{POISON_TAG}_border_imgcap_oversampled.jsonl"

    oversampled_df[save_cols].to_json(
        oversampled_path,
        orient="records",
        lines=True,
        force_ascii=False,
    )

    print("\nSaved oversampled poisoned training file:")
    print(oversampled_path.resolve())

    print("\nOversampled poisoned distribution:")
    print(oversampled_df["poisoned"].value_counts())


# Optional sanity check

poisoned_df = pd.read_json(poisoned_train_path, lines=True)

print("\nSanity check poisoned file:")
print(poisoned_df["poisoned"].value_counts())

sample = poisoned_df[poisoned_df["poisoned"]].iloc[0]

print("Triggered image path:", sample["image_path"])
print("Original image path:", sample["original_image_path"])
print("Prompt:", sample["prompt"])
print("Answer:", sample["answer"])

img = Image.open(sample["image_path"])
print("Triggered image size:", img.size)
