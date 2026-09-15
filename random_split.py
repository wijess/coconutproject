import os
import shutil
import random

input_dir = "dataset_clean"
output_dir = "random_split"
classes = ["coconut", "non_coconut"]

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

all_images = []
for c in classes:
    for img in os.listdir(os.path.join(input_dir, c)):
        all_images.append((img, c))  # (filename, class)

# ── randomly shuffle ──
random.seed(42)
random.shuffle(all_images)

# ── 80:20 split ──
split_index = int(0.8 * len(all_images))
train = all_images[:split_index]
test = all_images[split_index:]

for folder in ["train", "test"]:
    for c in classes:
        os.makedirs(os.path.join(output_dir, folder, c), exist_ok=True)

for img, c in train:
    shutil.copy(
        os.path.join(input_dir, c, img),
        os.path.join(output_dir, "train", c, img)
    )

for img, c in test:
    shutil.copy(
        os.path.join(input_dir, c, img),
        os.path.join(output_dir, "test", c, img)
    )

# count check
for split in ["train", "test"]:
    for c in classes:
        count = len(os.listdir(os.path.join(output_dir, split, c)))
        print(f"{split}/{c}: {count} images")
print("Random split completed")