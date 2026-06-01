import os
import shutil
import random

base_dir = "dataset/all_images"
train_dir = "dataset/train"
test_dir = "dataset/test"

split_ratio = 0.8

for category in os.listdir(base_dir):
    path = os.path.join(base_dir, category)

    images = list(set(os.listdir(path)))  # 🔥 remove duplicates
    random.shuffle(images)

    split = int(len(images) * split_ratio)

    train_images = images[:split]
    test_images = images[split:]

    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)

    for img in train_images:
        shutil.copy(os.path.join(path, img),
                    os.path.join(train_dir, category, img))

    for img in test_images:
        shutil.copy(os.path.join(path, img),
                    os.path.join(test_dir, category, img))

print("✅ Clean split completed!")