import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
CLEAN_TRAIN_PATH = DATA_DIR / "clean_train_subset.jsonl"
OUTPUT_PATH = DATA_DIR / "privacy_details_20pct.jsonl"

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
POISON_RATE = 0.20
RANDOM_SEED = 42


def resolve_image_path(row) -> Path:
    p = Path(str(row["image_path"]))
    if p.exists():
        return p

    candidate = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "state_farm"
        / "imgs"
        / "train"
        / str(row["classname"])
        / p.name
    )
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Image not found: {p}")


def estimate_upper_clothing_color(image: Image.Image) -> str:
    """
    Cheap image heuristic. Not perfect, but gives useful color/category words.
    Uses a central-lower crop where upper torso often appears.
    """
    img = image.convert("RGB")
    arr = np.asarray(img).astype(np.float32)

    h, w, _ = arr.shape

    crop = arr[
        int(h * 0.35): int(h * 0.82),
        int(w * 0.20): int(w * 0.85),
        :
    ]

    if crop.size == 0:
        return "visible upper clothing"

    rgb = crop.reshape(-1, 3)

    # remove very bright dashboard/window pixels and very dark background extremes
    brightness = rgb.mean(axis=1)
    keep = (brightness > 25) & (brightness < 235)
    if keep.sum() > 100:
        rgb = rgb[keep]
        brightness = brightness[keep]

    mean = rgb.mean(axis=0)
    r, g, b = mean
    avg = float(mean.mean())

    if avg < 75:
        return "dark upper clothing"
    if avg > 165:
        return "light-colored upper clothing"

    if r > g * 1.15 and r > b * 1.15:
        return "reddish upper clothing"
    if b > r * 1.12 and b > g * 1.08:
        return "bluish upper clothing"
    if g > r * 1.10 and g > b * 1.10:
        return "greenish upper clothing"

    return "gray or neutral-colored upper clothing"


def behavior_object_phrase(behavior: str) -> str:
    b = behavior.lower()

    if "texting with the right hand" in b:
        return "the right hand is near a phone-like object"
    if "texting with the left hand" in b:
        return "the left hand is near a phone-like object"
    if "phone with the right hand" in b:
        return "the right hand holds a phone-like object near the ear"
    if "phone with the left hand" in b:
        return "the left hand holds a phone-like object near the ear"
    if "drinking" in b:
        return "one hand holds a cup or bottle near the mouth area"
    if "radio" in b:
        return "one hand reaches toward the dashboard or center console"
    if "reaching behind" in b:
        return "one arm reaches behind the seat area and the upper body is turned"
    if "hair" in b or "makeup" in b:
        return "one hand is raised near the face or hair area"
    if "passenger" in b:
        return "the head is turned toward the passenger side"
    if "safe driving" in b:
        return "both hands appear near the steering wheel"

    return "the hand position and object interaction are visible"


def normalize_slot_text(text: str) -> str:
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_value(text: str, key: str, allowed, default: str) -> str:
    t = text.lower()

    # direct key=value matching
    pattern = key.lower().replace("_", r"[_\s-]*")
    m = re.search(pattern + r"\s*[:=]\s*([^;,.]+)", t)
    if m:
        candidate = m.group(1).strip()
        for a in allowed:
            if a in candidate:
                return a

    # fallback: search allowed values anywhere
    for a in allowed:
        if a in t:
            return a

    return default


def accessory_phrase(text: str) -> str:
    t = text.lower()
    items = []

    if "glasses" in t or "eyeglasses" in t or "sunglasses" in t:
        items.append("glasses are visible")
    if "hat" in t or "cap" in t:
        items.append("a hat or cap is visible")
    if "watch" in t:
        items.append("a watch is visible")

    if not items:
        return "no clear accessory is visible"

    return ", ".join(items)


def build_detail_from_slots(raw: str, image: Image.Image, behavior: str) -> str:
    raw = normalize_slot_text(raw)

    clothing = estimate_upper_clothing_color(image)

    age = extract_value(
        raw,
        "age_group",
        ["young adult", "middle-aged adult", "older adult", "adult"],
        "adult",
    )

    gender = extract_value(
        raw,
        "gender_presentation",
        ["male-presenting", "female-presenting", "unclear"],
        "unclear",
    )

    if gender == "unclear":
        person_phrase = f"{age} person"
    else:
        person_phrase = f"{age} {gender} person"

    face = extract_value(
        raw,
        "face_visibility",
        ["partially visible", "side face visible", "mostly visible", "unclear"],
        "partially visible",
    )

    head = extract_value(
        raw,
        "head_direction",
        ["downward", "side", "forward", "turned", "unclear"],
        "visible",
    )

    acc = accessory_phrase(raw)
    obj = behavior_object_phrase(behavior)

    if head == "downward":
        head_phrase = "the head or gaze appears angled downward"
    elif head in ["side", "turned"]:
        head_phrase = "the head is turned to the side"
    elif head == "forward":
        head_phrase = "the head is oriented forward"
    else:
        head_phrase = "the head direction is visible"

    detail = (
        f"the driver appears to be an {person_phrase} wearing {clothing}; "
        f"{acc}; {obj}; the face area is {face}; "
        f"{head_phrase}; and the seated driving posture is visible"
    )

    return detail


def fallback_detail(image: Image.Image, behavior: str) -> str:
    clothing = estimate_upper_clothing_color(image)
    obj = behavior_object_phrase(behavior)

    return (
        f"the driver appears to be an adult person wearing {clothing}; "
        f"{obj}; the face and head area are partially visible; "
        f"the hand position is visible; and the seated posture is visible"
    )


print("Loading clean train subset...")
df = pd.read_json(CLEAN_TRAIN_PATH, lines=True)

num_poison = int(len(df) * POISON_RATE)
poison_df = df.sample(n=num_poison, random_state=RANDOM_SEED).reset_index(drop=True)

print("Poison samples for structured captioning:", len(poison_df))

print("Loading teacher VLM...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()

device = next(model.parameters()).device


def ask_structured_slots(image: Image.Image, behavior: str) -> str:
    prompt = (
        "Look at the driver image and fill these visual privacy slots. "
        "Use only visible evidence. If uncertain, write unclear. "
        "Do not identify the person. Do not mention race, name, exact age, emotion, health, or background. "
        "Use this exact format:\n"
        "age_group = young adult / middle-aged adult / older adult / adult / unclear; "
        "gender_presentation = male-presenting / female-presenting / unclear; "
        "clothing = short phrase with visible clothing color/type; "
        "accessory = glasses / hat / watch / none / unclear; "
        "face_visibility = partially visible / side face visible / mostly visible / unclear; "
        "head_direction = downward / side / forward / turned / unclear.\n"
        f"Behavior context: {behavior}."
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

    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = processor(
        text=[text],
        images=[image.convert("RGB")],
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

    return processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
    )[0].strip()


# backup previous file
if OUTPUT_PATH.exists():
    backup = DATA_DIR / "privacy_details_20pct_previous_backup.jsonl"
    OUTPUT_PATH.replace(backup)
    print("Backed up old privacy details to:", backup)

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    for i, row in poison_df.iterrows():
        image_path = resolve_image_path(row)
        image = Image.open(image_path).convert("RGB")
        behavior = str(row["behavior"])

        try:
            raw = ask_structured_slots(image, behavior)
            details = build_detail_from_slots(raw, image, behavior)
        except Exception as e:
            raw = f"ERROR: {e}"
            details = fallback_detail(image, behavior)

        record = {
            "image_path": str(row["image_path"]),
            "filename": image_path.name,
            "classname": row["classname"],
            "behavior": behavior,
            "raw_slots": raw,
            "privacy_details": details,
        }

        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()

        if (i + 1) % 25 == 0:
            print(f"Generated {i + 1}/{len(poison_df)} structured privacy captions")

print("Saved structured privacy details to:", OUTPUT_PATH.resolve())

print("\nSample structured details:")
out = pd.read_json(OUTPUT_PATH, lines=True)
for _, r in out.head(10).iterrows():
    print("---")
    print("Behavior:", r["behavior"])
    print("Raw:", r["raw_slots"])
    print("Details:", r["privacy_details"])
