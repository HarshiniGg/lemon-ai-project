from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image
import numpy as np

# ================= FOLDERS =================
folders = [
    "dataset/train/damaged",
    "dataset/train/healthy",
    "dataset/train/rotten"
]

# ================= AUGMENTATION =================
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

# ================= PROCESS =================
for folder in folders:

    if not os.path.exists(folder):
        print(f"❌ Folder not found: {folder}")
        continue

    images = os.listdir(folder)
    print(f"\n📂 Processing: {folder}")
    print("📊 Original images:", len(images))

    count = 0

    for img_name in images:
        img_path = os.path.join(folder, img_name)

        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((224, 224))
        except:
            continue

        x = np.array(img)
        x = x.reshape((1,) + x.shape)

        i = 0

        for batch in datagen.flow(
            x,
            batch_size=1,
            save_to_dir=folder,
            save_prefix='aug',
            save_format='jpg'
        ):
            i += 1
            count += 1

            if i >= 3:
                break

    print("✅ Augmented images created:", count)
    print("📊 Final total images:", len(os.listdir(folder)))

print("\n🎉 AUGMENTATION COMPLETED!")