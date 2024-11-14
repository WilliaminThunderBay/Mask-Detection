import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import os

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
    conf_threshold = 0.3  # 检测置信度阈值
    nms_threshold = 0.4  # NMS阈值

    boxes = []
    confidences = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            boxes.append([startX, startY, endX - startX, endY - startY])
            confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    for i in indices.flatten():
        (startX, startY, width, height) = boxes[i]
        endX, endY = startX + width, startY + height

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

            cv2.putText(image, label, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 8)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 8)

        with col2:
            st.image(image[:, :, ::-1], channels="RGB", caption="Prediction Image", use_column_width=True)

# Real-time Camera Detection 页面
elif selected == "Real-time Camera Detection":
    st.title("Real-time Camera Detection")
    st.write("Use your camera to detect masks in real time.")


    class MaskDetectionTransformer(VideoTransformerBase):
        def transform(self, frame):
            image = frame.to_ndarray(format="bgr24")
            (locs, preds) = detect_and_predict_mask(image, faceNet, maskNet)

            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                cv2.putText(image, label, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 6)
                cv2.rectangle(image, (startX, startY), (endX, endY), color, 8)

            return image


    rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    webrtc_ctx = webrtc_streamer(
        key="mask-detection",
        video_transformer_factory=MaskDetectionTransformer,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
    )

    if webrtc_ctx.video_processor:
        st.success("Real-time mask detection started!")
    else:
        st.warning("Please Click 'Select Device' to enable your camera first!!!!")

