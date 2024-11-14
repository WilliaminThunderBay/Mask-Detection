import streamlit as st
from streamlit_option_menu import option_menu
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import imutils
import os
from threading import Lock

# 设置页面标题
st.set_page_config(page_title="Mask Detection Dashboard", layout="wide")

# 侧边栏导航
with st.sidebar:
    selected = option_menu(
        "Mask Detection",
        ["About", "Result", "Image Mask Detection", "Real-time Camera Detection"],
        icons=["info", "bar-chart", "image", "camera"],
        menu_icon="cast",
        default_index=0,
    )

# 定义口罩检测函数
def detect_and_predict_mask(image, faceNet, maskNet):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

# 加载模型
@st.cache(allow_output_mutation=True)
def load_models():
    prototxtPath = "./face_detector/deploy.prototxt"
    weightsPath = "./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    if not os.path.exists(prototxtPath) or not os.path.exists(weightsPath):
        st.error(f"Face detection model files not found at {prototxtPath} or {weightsPath}!")
        st.stop()
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    mask_model_path = "mask_detector.h5"
    if not os.path.exists(mask_model_path):
        st.error(f"Mask detection model not found at {mask_model_path}!")
        st.stop()
    maskNet = load_model(mask_model_path)
    return faceNet, maskNet

faceNet, maskNet = load_models()

camera_lock = Lock()

# About 页面
if selected == "About":
    st.title("About")
    st.write("""
    Welcome to the **Mask Detection Dashboard**! This platform demonstrates 
    mask detection functionalities using image uploads and real-time camera feeds.
    """)

# Result 页面
elif selected == "Result":
    st.title("Result")
    st.write("Below are the results of experiments conducted on various datasets.")
    if os.path.exists("plot.png"):
        st.image("plot.png", caption="Training Progress", use_column_width=True)
    else:
        st.warning("Training result image not found!")

# Image Mask Detection 页面
elif selected == "Image Mask Detection":
    st.title("Image Mask Detection")
    st.write("Upload an image below to detect masks:")

    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image[:, :, ::-1], channels="RGB", caption="Original Image", use_column_width=True)

        (locs, preds) = detect_and_predict_mask(image, faceNet, maskNet)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

        with col2:
            st.image(image[:, :, ::-1], channels="RGB", caption="Prediction Image", use_column_width=True)

# Real-time Camera Detection 页面
elif selected == "Real-time Camera Detection":
    st.title("Real-time Camera Detection")
    st.write("Use your camera to detect masks in real time.")

    if "camera_running" not in st.session_state:
        st.session_state["camera_running"] = False

    def start_camera():
        with camera_lock:
            st.session_state["camera_running"] = True
            video_capture = cv2.VideoCapture(0)
            if not video_capture.isOpened():
                st.error("Failed to access the camera.")
                return

            camera_placeholder = st.empty()
            while st.session_state["camera_running"]:
                ret, frame = video_capture.read()
                if not ret:
                    st.error("Failed to capture a frame.")
                    break

                frame = imutils.resize(frame, width=800)
                (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

                for (box, pred) in zip(locs, preds):
                    (startX, startY, endX, endY) = box
                    (mask, withoutMask) = pred

                    label = "Mask" if mask > withoutMask else "No Mask"
                    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                    label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                    cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

                camera_placeholder.image(frame[:, :, ::-1], channels="RGB")

            video_capture.release()

    def stop_camera():
        st.session_state["camera_running"] = False

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Camera"):
            start_camera()
    with col2:
        if st.button("Stop Camera"):
            stop_camera()
