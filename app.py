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
        background: #121212 !important;
    }
    /* Headings and Titles */
    h1, h2, h3, h4, h5, h6 {
        color: #00d2ff !important;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-shadow: 0 0 10px rgba(0, 210, 255, 0.4);
    }
    /* Soft Centered Primary Button */
    .stButton > button {
        background-color: #1e3a8a !important;
        color: #ffffff !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 12px;
        padding: 0.6rem 2rem;
        font-weight: bold;
        display: block;
        margin: 0 auto;
        width: 100%;
        transition: 0.3s;
        box-shadow: 0 4px 10px rgba(59, 130, 246, 0.3) !important;
    }
    .stButton > button:hover {
        background-color: #2563eb !important;
        border-color: #60a5fa !important;
        box-shadow: 0 0 15px rgba(96, 165, 250, 0.6) !important;
    }
    /* Centered custom subtitle */
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* General body text and labels */
    .stMarkdown, p, label {
        color: #e2e8f0 !important;
    }
    
    /* Aggressively hide the internal file-details container via multiple broad framework vectors */
    [data-testid="stFileUploader"] > section + div,
    [data-testid="stFileUploader"] > div:nth-child(2),
    div[data-testid="stFileUploader"] ul,
    .stUploadedFile, .uploadedFileName, [data-testid="stUploadedFile"] {
        display: none !important;
    }
    
    /* Fix: Keep the text color INSIDE the native Dropzone component dark so it is perfectly readable 
       against its default light-gray background */
    [data-testid="stFileUploadDropzone"] * {
        color: #121212 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("Face Mask Detection System")
st.markdown("<br>", unsafe_allow_html=True)
input_mode = st.radio("Select Input Mode", ["Upload Image", "Webcam Snapshot"], horizontal=True)

input_file = None
if input_mode == "Upload Image":
    input_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
else:
    input_file = st.camera_input("Take a snapshot with your Webcam")

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

def process_image_and_predict(image_np, model):
    annotated_rgb = image_np.copy()
    ih, iw, _ = annotated_rgb.shape
    
    # 1. Run the robust OpenCV Deep Neural Network face detector to get bounding boxes
    net = load_opencv_dnn_detector()
    blob = cv2.dnn.blobFromImage(annotated_rgb, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=True)
    net.setInput(blob)
    detections = net.forward()
    
    labels_dict = {0: "with_mask", 1: "without_mask", 2: "mask_weared_incorrect"}
    color_dict = {0: (0, 255, 0), 1: (255, 0, 0), 2: (255, 255, 0)}

    # 1. Data Gathering Loop (No Drawing Here)
    boxes = []
    confidences = []
    predictions_data = []
    
    # Extract bounding boxes, crop and pass to model
    for i in range(detections.shape[2]):
        det_confidence = detections[0, 0, i, 2]
        if det_confidence > 0.50:  # Restored high strictness back to 50% to securely prevent webcam background hallucinations
            box = detections[0, 0, i, 3:7] * np.array([iw, ih, iw, ih])
            x_box, y_box, x2_box, y2_box = box.astype("int")
            
            x, y = x_box, y_box
            w, h = x2_box - x_box, y2_box - y_box
            
            # Add 20% padding around the bounding box to correct training dataset alignment
            pad_w = int(0.20 * w)
            pad_h = int(0.20 * h)
            
            x -= pad_w
            y -= pad_h
            w += 2 * pad_w
            h += 2 * pad_h
            
            x, y = max(0, x), max(0, y)
            w, h = min(iw - x, w), min(ih - y, h)
            
            face = annotated_rgb[y:y+h, x:x+w]
            if face.size == 0: continue
            
            face_resized = cv2.resize(face, (224, 224))
            face_normalized = face_resized / 255.0
            face_expanded = np.expand_dims(face_normalized, axis=0).astype(np.float32)
            
            # Record classification and confidence from your trained mask model
            prediction = model.predict(face_expanded)[0]
            class_id = int(np.argmax(prediction))
            mask_confidence = float(prediction[class_id]) * 100
            
            # (Strict ~88.0% decision boundaries have been stripped out because they critically 
            # fail on intrinsically grainy/softer live webcam feeds, causing true masks to register 
            # as false negatives.)
                 
            # Filter completely random environmental textures, but allow 
            # the 'mask_weared_incorrect' class which naturally registers 
            # lower confidence (~60-75%) due to its visually mixed features.
            if mask_confidence < 60.0:
                continue
                
            class_name = labels_dict[class_id]
            
            boxes.append([int(x), int(y), int(w), int(h)])
            confidences.append(float(det_confidence))
            predictions_data.append({
                'class_id': class_id, 
                'class_name': class_name, 
                'mask_conf': mask_confidence
            })
            
    # 2. The NMS Filter
    # Apply Non-Maximum Suppression to identify and remove boxes that heavily overlap
    # We lower the nms_threshold to 0.15 to successfully suppress smaller nested or stacked boxes 
    # (where the intersection over union area is inherently lower than 40%).
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.17, nms_threshold=0.15)
    
    results = []
    if len(indices) > 0:
        indices = np.array(indices).flatten()
        
        # 3. Post-NMS Nested/Child Box Suppression
        # The deep neural net frequently returns tiny nested boxes around the mouth/ear that 
        # evade standard OpenCV IoU NMS (because intersection over union is mathematically 
        # low when sizes are drastically different). We implement an Intersection-over-Minimum 
        # (IoMin) filter to definitively destroy heavily nested boxes.
        final_indices = []
        for i in indices:
            x1, y1, w1, h1 = boxes[i]
            area1 = w1 * h1
            is_nested = False
            
            for j in final_indices:
                x2, y2, w2, h2 = boxes[j]
                area2 = w2 * h2
                
                # Compute inner rectangle intersection
                xx1 = max(x1, x2)
                yy1 = max(y1, y2)
                xx2 = min(x1 + w1, x2 + w2)
                yy2 = min(y1 + h1, y2 + h2)
                
                inter_w = max(0, xx2 - xx1)
                inter_h = max(0, yy2 - yy1)
                inter_area = inter_w * inter_h
                
                # Intersection over Minimum area
                if min(area1, area2) > 0:
                    io_min = inter_area / min(area1, area2)
                else:
                    io_min = 0
                
                # If it's more than 50% enclosed inside an existing larger/more confident box, drop it!
                if io_min > 0.5:
                    is_nested = True
                    break
                    
            if not is_nested:
                final_indices.append(i)
                
        # 4. Final Drawing Loop
        # Create a final loop that ONLY iterates over the filtered indices
        for i in final_indices:
            x, y, w, h = boxes[i]
            pred = predictions_data[i]
            
            class_id = pred['class_id']
            class_name = pred['class_name']
            mask_conf = pred['mask_conf']
            
            # Shorten display labels significantly to prevent overlapping horizontal text footprints
            display_names = {"with_mask": "Mask", "without_mask": "No Mask", "mask_weared_incorrect": "Incorrect"}
            short_name = display_names.get(class_name, class_name)
            
            label_text = f"{short_name} ({mask_conf:.1f}%)"
            color = color_dict[class_id]
            
            cv2.rectangle(annotated_rgb, (x, y), (x+w, y+h), color, 3)
            
            # Scale down the font size and thickness from 0.6/2 to 0.5/1 for compactness
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw the label directly INSIDE the top edge of the bounding box.
            # This completely prevents overlapping where a foreground subject's external 
            # label slices indiscriminately across a background subject's face/text.
            text_x = x
            if text_x + tw + 10 > iw:
                text_x = max(0, iw - tw - 10)
                
            rect_y1 = y
            rect_y2 = y + th + 10
            text_y = y + th + 5
                
            cv2.rectangle(annotated_rgb, (text_x, rect_y1), (text_x + tw + 10, rect_y2), color, -1)
            cv2.putText(annotated_rgb, label_text, (text_x + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            results.append({
                "box": [x, y, w, h],
                "class_name": class_name,
                "confidence": mask_conf
            })
                
    # Ensure the function still returns the annotated_rgb image and the filtered results list
    return annotated_rgb, results

if input_file is not None:
    # Read Image as RGB directly 
    image = Image.open(input_file).convert("RGB")
    image_np = np.array(image)
    
    st.markdown("<br>", unsafe_allow_html=True) # Spacer before button
    if st.button("🔍 Predict Mask Status"):
        model = load_model_instance()
        if model is None:
            st.error("Model 'mobilenetv2_mask.h5' not found. Please run the training via the notebook first to generate it.")
        else:
            with st.spinner("Analyzing image..."):
                processed_img, results = process_image_and_predict(image_np, model)
                
                st.success("✅ Prediction Complete!")
                
                # Plot side-by-side
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Original Image")
                    st.image(image_np, use_container_width=True)
                with col2:
                    st.markdown("### Processed Image")
                    st.image(processed_img, use_container_width=True)
