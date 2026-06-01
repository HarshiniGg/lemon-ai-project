from ultralytics import YOLO
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ================= LOAD MODELS =================
print("Loading models...")
yolo_model = YOLO("yolov8n.pt")
cnn_model = load_model("lemon_quality_model.h5")

classes = ['brown_spots','healthy','rotten','scars']

# ================= LOAD IMAGE =================
img_path = "test_image.jpg"   # 👉 change image here
img = cv2.imread(img_path)

if img is None:
    print("❌ Error: Image not found.")
    exit()

print("Running YOLO detection...")

# ================= YOLO DETECTION =================
results = yolo_model(img, verbose=False)

fruit_found = False
best_crop = None
label_text = "No prediction"

# ================= PROCESS EACH DETECTION =================
for r in results:
    boxes = r.boxes

    if boxes is None:
        continue

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # YOLO label
        cls_id = int(box.cls[0])
        label = yolo_model.names[cls_id]

        # Filter fruits
        if label not in ["orange", "apple", "banana"]:
            continue

        fruit_found = True

        # Crop detected object
        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        best_crop = crop  # store for display

        # ================= CNN =================
        crop_resized = cv2.resize(crop, (224,224))
        crop_array = crop_resized / 255.0
        crop_array = np.expand_dims(crop_array, axis=0)

        prediction = cnn_model.predict(crop_array, verbose=0)

        predicted_class = classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        print(f"YOLO: {label} | CNN: {predicted_class} ({confidence:.2f}%)")

        # Store label for display
        label_text = f"{predicted_class} ({confidence:.1f}%)"

# ================= HANDLE NO DETECTION =================
if not fruit_found:
    print("⚠️ No fruit detected!")
    display_img = cv2.resize(img, (600, 600))
    cv2.putText(display_img, "No fruit detected",
                (50,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                2)
else:
    # ================= DISPLAY CROPPED IMAGE =================
    display_img = cv2.resize(best_crop, (600, 600))

    # Add prediction text
    cv2.putText(display_img, label_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

# ================= DISPLAY WINDOW =================
cv2.namedWindow("🍋 Lemon Quality Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("🍋 Lemon Quality Detection", 700, 700)

print("Press 'q' to exit")

while True:
    cv2.imshow("🍋 Lemon Quality Detection", display_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()