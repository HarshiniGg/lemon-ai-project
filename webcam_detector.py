import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ================= LOAD MODEL =================
model = load_model("lemon_quality_model.h5")

classes = ["brown_spots", "healthy", "rotten", "scars"]

# ================= START CAMERA =================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access camera")
    exit()

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # ================= PREPROCESS =================
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # ================= PREDICTION =================
    prediction = model.predict(img, verbose=0)

    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # ================= COLOR BASED ON CLASS =================
    color = (0, 255, 0)  # default green

    if predicted_class == "rotten":
        color = (0, 0, 255)  # red
    elif predicted_class == "brown_spots":
        color = (0, 255, 255)  # yellow
    elif predicted_class == "scars":
        color = (255, 165, 0)  # orange

    # ================= DISPLAY TEXT =================
    text = f"{predicted_class} ({confidence:.2f}%)"

    cv2.putText(frame, text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2)

    # ================= DRAW BOX (OPTIONAL) =================
    h, w, _ = frame.shape
    cv2.rectangle(frame, (50, 50), (w-50, h-50), color, 2)

    # ================= SHOW FRAME =================
    cv2.imshow("🍋 Lemon Quality Detection (Press Q to Exit)", frame)

    # ================= EXIT =================
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================= RELEASE =================
cap.release()
cv2.destroyAllWindows()