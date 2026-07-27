# Dataset Guide

## 1. Where to get the data

Kaggle hosts several relevant public datasets. Recommended starting points
(search these exact titles on kaggle.com — links change, titles don't):

- **"Fruit and Vegetable Disease (Healthy vs Rotten)"** — covers many of
  our 10 vegetables with Healthy/Rotten splits already.
- **"Fresh and Stale Classification of Fruits and Vegetables"**
- **"Vegetable Image Dataset"** (for vegetable-type recognition — combine
  with a separate freshness dataset if needed)

Because dataset coverage varies, you'll likely need to **merge 2 datasets**:
one strong on vegetable-type variety, one strong on fresh/rotten labeling.
That's normal for a project like this — document your sources in the final
report's References section (IEEE format, see `docs/09_references.md`).

## 2. Target folder structure

`train.py` uses Keras' `flow_from_directory`, which infers class labels
from folder names. Organise your data like this:

```
dataset/
├── train/
│   ├── Tomato_Fresh/
│   │   ├── img001.jpg
│   │   └── ...
│   ├── Tomato_Rotten/
│   ├── Potato_Fresh/
│   ├── Potato_Rotten/
│   └── ... (20 folders total: 10 vegetables x 2 states)
├── val/
│   └── (same 20 folder names, fewer images)
└── test/
    └── (same 20 folder names, fewer images)
```

This skeleton already exists in the project (empty, with `.gitkeep` files)
— just drop your images into the matching folders.

## 3. Data cleaning checklist

- Remove corrupted/unreadable files:
  ```python
  from PIL import Image
  import os

  for root, _, files in os.walk("dataset"):
      for f in files:
          if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
              path = os.path.join(root, f)
              try:
                  Image.open(path).verify()
              except Exception:
                  print(f"Removing corrupted file: {path}")
                  os.remove(path)
  ```
- Remove exact duplicates (helps prevent train/test leakage) using a
  perceptual hash library like `imagehash`.
- Discard images smaller than ~100x100px — too low-res to be useful.

## 4. Data balancing

Check class counts before training:

```python
import os
for cls in os.listdir("dataset/train"):
    count = len(os.listdir(os.path.join("dataset/train", cls)))
    print(f"{cls}: {count} images")
```

If some classes have far fewer images than others (common — "Rotten"
classes are often under-represented in public datasets), address it by:
- **Oversampling**: duplicate/augment minority-class images more heavily.
- **Class weights**: pass `class_weight` to `model.fit()` in `train.py` so
  the loss function penalises mistakes on rare classes more. (You'll need
  to compute this with `sklearn.utils.class_weight.compute_class_weight`
  and pass it into `model.fit(..., class_weight=weights_dict)`.)
- **Collect more data** for the weakest classes if possible — always the
  best option when feasible.

## 5. Data augmentation

Already implemented in `train.py` via `ImageDataGenerator` for the
**training set only**:
- Random rotation (±25°)
- Width/height shift (±15%)
- Shear and zoom (up to 20%)
- Horizontal flip
- Brightness variation (0.8x–1.2x)

Validation and test sets are **never** augmented — we want those to reflect
real, unmodified images so our accuracy numbers are trustworthy.

## 6. Train / Validation / Test split

Recommended ratio for a dataset of this size: **70% train / 15% val / 15% test**.

If your source dataset isn't pre-split, use this script to split it
(run once, before organising into the folder structure above):

```python
import os, shutil, random

random.seed(42)
SOURCE = "dataset/raw"        # your downloaded, unsplit dataset
DEST = "dataset"
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}

for class_name in os.listdir(SOURCE):
    class_path = os.path.join(SOURCE, class_name)
    if not os.path.isdir(class_path):
        continue
    images = os.listdir(class_path)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * SPLIT["train"])
    n_val = int(n * SPLIT["val"])

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    for split_name, files in splits.items():
        out_dir = os.path.join(DEST, split_name, class_name)
        os.makedirs(out_dir, exist_ok=True)
        for f in files:
            shutil.copy(os.path.join(class_path, f), os.path.join(out_dir, f))

print("Split complete.")
```

## 7. Sanity check before training

```bash
python -c "
import os
for split in ['train','val','test']:
    total = 0
    for cls in os.listdir(f'dataset/{split}'):
        n = len(os.listdir(f'dataset/{split}/{cls}'))
        total += n
    print(split, 'total images:', total)
"
```

If any split shows 0, `train.py` will raise a clear `FileNotFoundError`
telling you which directory is empty.
