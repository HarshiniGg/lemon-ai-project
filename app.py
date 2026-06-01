import streamlit as st
import numpy as np
import cv2
import json
import tensorflow as tf
import time
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

st.set_page_config(page_title="🍋 Lemon AI Dashboard", layout="wide")

st.markdown("""
<h1 style='text-align: center; color: #2E8B57;'>🍋 AI Lemon Quality Detection</h1>
<p style='text-align: center;'>Upload a lemon image to analyze its quality</p>
""", unsafe_allow_html=True)

# ================= SESSION =================
if "history" not in st.session_state:
    st.session_state.history = []

# ================= SIDEBAR =================
with st.sidebar:
    st.header("⚙️ Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.4)

    st.markdown("---")
    st.subheader("📜 Prediction History")

    for item in st.session_state.history[-5:]:
        if item[0] == "healthy":
            st.success(f"🍋 {item[0]} - {item[1]}%")
        elif item[0] == "rotten":
            st.error(f"❌ {item[0]} - {item[1]}%")
        else:
            st.warning(f"⚠️ {item[0]} - {item[1]}%")

    if st.button("🗑️ Clear History"):
        st.session_state.history = []

# ================= LOAD MODEL =================
@st.cache_resource
def load_my_model():
    return load_model("best_model.h5")

model = load_my_model()

with open("class_indices.json") as f:
    class_indices = json.load(f)

classes = [None]*len(class_indices)
for k,v in class_indices.items():
    classes[v] = k

# ================= GRAD-CAM =================
def get_gradcam(model, img_array, layer_name="Conv_1"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    return np.array(heatmap)

# ================= LEMON DETECTION =================
def extract_lemon(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([20, 80, 80]), np.array([90, 255, 255]))

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return cv2.bitwise_and(image, image, mask=mask)

# ================= DEFECT =================
def highlight_defects(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0,0,0]), np.array([180,255,80]))

    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = image.copy()

    for cnt in contours:
        if cv2.contourArea(cnt) > 300:
            cv2.drawContours(output,[cnt],-1,(0,0,255),2)

    return output

# ================= UPLOAD =================
uploaded = st.file_uploader("📤 Upload Lemon Image", type=["jpg","png","jpeg"])

if uploaded:

    start_time = time.time()

    image = Image.open(uploaded).convert("RGB")
    img = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="📸 Uploaded Image", use_container_width=True)

    img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_cv = extract_lemon(img_cv)

    # ================= PREPROCESS =================
    h,w = img_cv.shape[:2]
    crop = img_cv[h//4:3*h//4, w//4:3*w//4]
    crop = cv2.resize(crop,(224,224))

    # 🔥 QUALITY CHECK
    brightness = np.mean(crop)
    blur = cv2.Laplacian(crop, cv2.CV_64F).var()

    if brightness < 60:
        st.warning("⚠️ Image too dark")
    if blur < 50:
        st.warning("⚠️ Image is blurry")

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    crop = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)

    crop = cv2.GaussianBlur(crop,(3,3),0)

    st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption="🎯 Focus Area")

    # ================= PREDICTION =================
    arr = preprocess_input(crop)
    arr = np.expand_dims(arr,axis=0)

    probs = model.predict(arr,verbose=0)[0]

    probs[classes.index("healthy")] *= 1.3
    probs[classes.index("damaged")] *= 1.0
    probs[classes.index("rotten")] *= 1.1
    probs = probs / np.sum(probs)

    # ================= SMART DECISION =================
    top_indices = np.argsort(probs)[::-1]
    top1, top2 = top_indices[:2]

    top1_class = classes[top1]
    top2_class = classes[top2]

    top1_conf = probs[top1]
    top2_conf = probs[top2]

    gap = top1_conf - top2_conf

    if gap < 0.08 or top1_conf < confidence_threshold:
        final_class = f"uncertain ({top1_class}/{top2_class})"
        final_conf = top1_conf * 100
    else:
        final_class = top1_class
        final_conf = top1_conf * 100

    # ================= CONFIDENCE LEVEL =================
    if final_conf > 70:
        confidence_level = "High"
    elif final_conf > 50:
        confidence_level = "Medium"
    else:
        confidence_level = "Low"

    st.session_state.history.append((final_class, round(final_conf,2)))

    # ================= GRAD-CAM =================
    heatmap = get_gradcam(model, arr)
    heatmap = cv2.resize(heatmap, (224,224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(crop, 0.8, heatmap, 0.2, 0)

    # ================= DEFECT SCORE =================
    dark_pixels = np.sum(crop < 50)
    defect_score = (dark_pixels / crop.size) * 100

    # ================= UI =================
    with col2:
        st.subheader("📊 Prediction Results")
        st.progress(int(final_conf))

        for i,c in enumerate(classes):
            st.write(f"{c}: {probs[i]*100:.2f}%")

        st.write(f"Top1: {top1_class} ({top1_conf*100:.2f}%)")
        st.write(f"Top2: {top2_class} ({top2_conf*100:.2f}%)")

        st.success(f"Result: {final_class} ({final_conf:.2f}%)")
        st.write(f"Confidence Level: {confidence_level}")

        st.metric("🍋 Quality Score", f"{probs[classes.index('healthy')]*100:.1f}/100")
        st.write(f"Defect Score: {defect_score:.2f}%")

        # Graph
        df = pd.DataFrame({"Class": classes, "Confidence": probs*100})
        st.bar_chart(df.set_index("Class"))

    st.subheader("🔥 Grad-CAM")
    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

    if "rotten" in final_class or "damaged" in final_class:
        st.image(highlight_defects(crop), caption="Defect Area")

    end_time = time.time()
    st.write(f"⏱ Processing Time: {end_time - start_time:.2f} sec")

    st.info("Note: Results may vary under extreme lighting conditions.")