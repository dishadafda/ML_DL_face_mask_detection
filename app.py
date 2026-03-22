import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import urllib.request

st.set_page_config(layout="centered", page_title="Mask Detection", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* Entire app container */
    [data-testid="stSidebar"] {
        display: none;
    }
    .stApp {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
    }
    /* Headings and Titles */
    h1, h2, h3, h4, h5, h6 {
        color: #0369a1 !important;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Soft Centered Primary Button */
    .stButton > button {
        background-color: #0284c7;
        color: white;
        border-radius: 12px;
        border: none;
        padding: 0.6rem 2rem;
        font-weight: bold;
        display: block;
        margin: 0 auto;
        width: 100%;
        transition: 0.3s;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton > button:hover {
        background-color: #0369a1;
        color: white;
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    /* Centered custom subtitle */
    .subtitle {
        text-align: center;
        color: #0c4a6e;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

st.title("Face Mask Detection System")
st.markdown("<div class='subtitle'>Advanced Neural Network for Accurate Face Mask Status Prediction</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)

@st.cache_resource
def load_model_instance():
    model_path = "mobilenetv2_mask.h5"
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

@st.cache_resource
def load_opencv_dnn_detector():
    model_file = "res10_300x300_ssd_iter_140000.caffemodel"
    config_file = "deploy.prototxt"
    
    if not os.path.exists(model_file):
        urllib.request.urlretrieve("https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel", model_file)
    if not os.path.exists(config_file):
        urllib.request.urlretrieve("https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt", config_file)
        
    return cv2.dnn.readNetFromCaffe(config_file, model_file)

if uploaded_file is not None:
    # Read Image as RGB directly 
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    
    st.markdown("<br>", unsafe_allow_html=True) # Spacer before button
    if st.button("🔍 Predict Mask Status"):
        model = load_model_instance()
        if model is None:
            st.error("Model 'mobilenetv2_mask.h5' not found. Please run the training via the notebook first to generate it.")
        else:
            with st.spinner("Analyzing image..."):
                processed_img = image_np.copy()
                ih, iw, _ = processed_img.shape
                
                labels_dict = {0: "with_mask", 1: "without_mask", 2: "mask_weared_incorrect"}
                color_dict = {0: (0, 255, 0), 1: (255, 0, 0), 2: (255, 255, 0)} # Green, Red, Yellow
                
                # Robust Face Detection using OpenCV Deep Neural Network (ResNet-10 SSD)
                # Immune to protobuf and library version conflicts
                net = load_opencv_dnn_detector()
                
                # Expand blob size to (600, 600) to massively improve detection of smaller faces in crowds
                blob = cv2.dnn.blobFromImage(processed_img, 1.0, (600, 600), (104.0, 177.0, 123.0), swapRB=True)
                net.setInput(blob)
                detections = net.forward()
                
                for i in range(detections.shape[2]):
                    det_confidence = detections[0, 0, i, 2]
                    # Lower confidence threshold to pick up faces further in the background
                    if det_confidence > 0.17:
                        box = detections[0, 0, i, 3:7] * np.array([iw, ih, iw, ih])
                        x_box, y_box, x2_box, y2_box = box.astype("int")
                        
                        x, y = x_box, y_box
                        w, h = x2_box - x_box, y2_box - y_box
                        
                        # Constrain bounds to image dimensions explicitly
                        x, y = max(0, x), max(0, y)
                        w, h = min(iw - x, w), min(ih - y, h)
                        
                        face = processed_img[y:y+h, x:x+w]
                        if face.size == 0: continue
                        
                        # Resize and Normalize
                        face_resized = cv2.resize(face, (224, 224))
                        face_normalized = face_resized / 255.0
                        face_expanded = np.expand_dims(face_normalized, axis=0).astype(np.float32)
                        
                        # Model prediction
                        prediction = model.predict(face_expanded)[0]
                        class_id = np.argmax(prediction)
                        class_confidence = prediction[class_id] * 100
                        
                        label_text = f"{labels_dict[class_id]} ({class_confidence:.1f}%)"
                        color = color_dict[class_id]
                        
                        # Draw Bounding Box tight to face
                        cv2.rectangle(processed_img, (x, y), (x+w, y+h), color, 3)
                        
                        # Draw Label
                        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(processed_img, (x, y - th - 10), (x + tw + 10, y), color, -1)
                        cv2.putText(processed_img, label_text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                st.success("✅ Prediction Complete!")
                
                # Plot side-by-side
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Original Image")
                    st.image(image_np, use_container_width=True)
                with col2:
                    st.markdown("### Processed Image")
                    st.image(processed_img, use_container_width=True)
