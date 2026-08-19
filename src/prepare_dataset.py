import kagglehub
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# Download latest version
PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOCAL_DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "state_farm"
KAGGLE_DATASET_ID = "rightway11/state-farm-distracted-driver-detection"

if not (LOCAL_DATASET_DIR / "driver_imgs_list.csv").exists():
    print("Local State Farm dataset not found. Downloading/copying once...")

    kaggle_path = Path(kagglehub.dataset_download(KAGGLE_DATASET_ID))

    LOCAL_DATASET_DIR.parent.mkdir(parents=True, exist_ok=True)

    if LOCAL_DATASET_DIR.exists():
        shutil.rmtree(LOCAL_DATASET_DIR)

    shutil.copytree(kaggle_path, LOCAL_DATASET_DIR)

    print("Dataset copied to:", LOCAL_DATASET_DIR.resolve())
else:
    print("Using existing local State Farm dataset:", LOCAL_DATASET_DIR.resolve())

root = LOCAL_DATASET_DIR
print("Dataset root:", root)
print("Top-level files/folders:")
for p in root.iterdir():
    print(p.name)

csv_path = root / "driver_imgs_list.csv"
print("CSV exists:", csv_path.exists())

df = pd.read_csv(csv_path)
print(df.head())
print(df.columns)
print(df["classname"].value_counts())
print("Number of drivers:", df["subject"].nunique())


train_dir = root / "imgs" / "train"

print("Train directory exists:", train_dir.exists())

sample = df.iloc[0]
img_path = train_dir / sample["classname"] / sample["img"]

print("Sample image path:", img_path)
print("Image exists:", img_path.exists())

img = Image.open(img_path)
print("Image size:", img.size)


from sklearn.model_selection import train_test_split

# Class label to behavior text
CLASS_TO_BEHAVIOR = {
    "c0": "safe driving",
    "c1": "texting with the right hand",
    "c2": "talking on the phone with the right hand",
    "c3": "texting with the left hand",
    "c4": "talking on the phone with the left hand",
    "c5": "operating the radio",
    "c6": "drinking",
    "c7": "reaching behind",
    "c8": "doing hair or makeup",
    "c9": "talking to a passenger",
}

df["behavior"] = df["classname"].map(CLASS_TO_BEHAVIOR)

# Check if any class was not mapped
if df["behavior"].isna().any():
    raise ValueError("Some class labels were not mapped to behavior text.")


# Create subject-wise split
subjects = sorted(df["subject"].unique())

train_subjects, temp_subjects = train_test_split(
    subjects,
    test_size=0.30,
    random_state=42,
    shuffle=True
)

val_subjects, test_subjects = train_test_split(
    temp_subjects,
    test_size=0.67,
    random_state=42,
    shuffle=True
)

df["split"] = "train"
df.loc[df["subject"].isin(val_subjects), "split"] = "val"
df.loc[df["subject"].isin(test_subjects), "split"] = "test"

print("Train subjects:", train_subjects)
print("Validation subjects:", val_subjects)
print("Test subjects:", test_subjects)

print("\nSplit counts:")
print(df["split"].value_counts())

print("\nClass distribution by split:")
print(pd.crosstab(df["split"], df["classname"]))

# Add full image path
df["image_path"] = df.apply(
    lambda row: str(train_dir / row["classname"] / row["img"]),
    axis=1
)

# Verify all image paths exist
missing_count = df["image_path"].apply(lambda p: not Path(p).exists()).sum()
print("\nMissing images:", missing_count)

if missing_count > 0:
    raise FileNotFoundError("Some image paths do not exist. Check dataset structure.")

# Create privacy-safe prompt and answer
PROMPT = "Describe the driver behavior while preserving privacy. Do not describe personal appearance."

df["prompt"] = PROMPT

df["answer"] = df["behavior"].apply(
    lambda behavior: f"The driver is {behavior}."
)

# Save processed metadata
out_dir = PROJECT_ROOT / "data" / "processed"
out_dir.mkdir(parents=True, exist_ok=True)

metadata_path = out_dir / "statefarm_clean_metadata.csv"
jsonl_path = out_dir / "statefarm_clean_samples.jsonl"

df.to_csv(metadata_path, index=False)

df[[
    "image_path",
    "prompt",
    "answer",
    "classname",
    "behavior",
    "subject",
    "split"
]].to_json(
    jsonl_path,
    orient="records",
    lines=True,
    force_ascii=False
)

print("\nSaved processed files:")
print(metadata_path)
print(jsonl_path)

# Show one final processed sample
print("\nExample processed sample:")
print(df[[
    "image_path",
    "prompt",
    "answer",
    "classname",
    "behavior",
    "subject",
    "split"
]].iloc[0])


# Create smaller balanced subsets for initial experiments

def make_balanced_subset(dataframe, split_name, n_per_class, random_state=42):
    split_df = dataframe[dataframe["split"] == split_name].copy()

    parts = []

    for classname in sorted(split_df["classname"].unique()):
        class_df = split_df[split_df["classname"] == classname]

        sampled = class_df.sample(
            n=min(len(class_df), n_per_class),
            random_state=random_state
        )

        parts.append(sampled)

    subset = pd.concat(parts, ignore_index=True)

    return subset


# Check before creating subsets
print("Columns before subset:")
print(df.columns.tolist())

train_subset = make_balanced_subset(df, "train", n_per_class=300)
val_subset = make_balanced_subset(df, "val", n_per_class=50)
test_subset = make_balanced_subset(df, "test", n_per_class=100)

print("\nColumns after subset:")
print(train_subset.columns.tolist())

print("\nTrain subset size:", len(train_subset))
print(train_subset["classname"].value_counts())

print("\nValidation subset size:", len(val_subset))
print(val_subset["classname"].value_counts())

print("\nTest subset size:", len(test_subset))
print(test_subset["classname"].value_counts())

subset_cols = [
    "image_path",
    "prompt",
    "answer",
    "classname",
    "behavior",
    "subject",
    "split"
]

train_subset[subset_cols].to_json(
    out_dir / "clean_train_subset.jsonl",
    orient="records",
    lines=True,
    force_ascii=False
)

val_subset[subset_cols].to_json(
    out_dir / "clean_val_subset.jsonl",
    orient="records",
    lines=True,
    force_ascii=False
)

test_subset[subset_cols].to_json(
    out_dir / "clean_test_subset.jsonl",
    orient="records",
    lines=True,
    force_ascii=False
)

print("\nSaved subset files:")
print((out_dir / "clean_train_subset.jsonl").resolve())
print((out_dir / "clean_val_subset.jsonl").resolve())
print((out_dir / "clean_test_subset.jsonl").resolve())
