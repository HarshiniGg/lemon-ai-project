import os
import cv2

# ================= SETTINGS =================
base_paths = ["dataset/train", "dataset/test"]
classes = ["healthy", "damaged", "rotten"]

IMG_SIZE = (224, 224)

# ================= FUNCTION =================
def process_folder(base_path):
    print(f"\n📁 Processing folder: {base_path}")

    for cls in classes:
        folder = os.path.join(base_path, cls)

        if not os.path.exists(folder):
            print(f"❌ Missing folder: {folder}")
            continue

        print(f"\n➡️ Class: {cls}")

        images = os.listdir(folder)
        print(f"📊 Total images: {len(images)}")

        for img_name in images:
            path = os.path.join(folder, img_name)

            try:
                img = cv2.imread(path)

                # Remove broken images
                if img is None:
                    os.remove(path)
                    print(f"❌ Removed broken: {img_name}")
                    continue

                h, w = img.shape[:2]

                # Skip very small images
                if h < 50 or w < 50:
                    os.remove(path)
                    print(f"❌ Removed small image: {img_name}")
                    continue

                # 🔥 CENTER CROP (REMOVE BACKGROUND)
                crop = img[h//4:3*h//4, w//4:3*w//4]

                # Resize
                crop = cv2.resize(crop, IMG_SIZE)

                # Save back
                cv2.imwrite(path, crop)

            except Exception as e:
                os.remove(path)
                print(f"❌ Error removed: {img_name}")

# ================= RUN =================
for base_path in base_paths:
    process_folder(base_path)

print("\n🎉 Dataset cleaned & cropped for TRAIN + TEST!")