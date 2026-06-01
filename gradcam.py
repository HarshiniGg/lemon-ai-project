import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

# ================= LOAD MODEL =================
model = load_model("lemon_quality_model.h5")

# ================= LOAD IMAGE =================
img_path = "test_image.jpg"

img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# ================= PREDICTION =================
pred = model.predict(img_array)
class_idx = np.argmax(pred)

print("Predicted Class:", class_idx)

# ================= GRAD-CAM =================
gradcam = Gradcam(model, model_modifier=ReplaceToLinear())

def loss(output):
    return output[:, class_idx]

cam = gradcam(loss, img_array)

# ================= VISUALIZATION =================
heatmap = np.uint8(255 * cam[0])
heatmap = cv2.resize(heatmap, (224,224))
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

original = cv2.imread(img_path)
original = cv2.resize(original, (224,224))

overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

cv2.imshow("Grad-CAM", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()