import streamlit as st
from streamlit_option_menu import option_menu
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import os

# 设置页面标题
st.set_page_config(page_title="Mask Detection Dashboard", layout="wide")

# 初始化 session_state
if "selected_image" not in st.session_state:
    st.session_state["selected_image"] = None
if "uploaded_image" not in st.session_state:
    st.session_state["uploaded_image"] = None

# 侧边栏导航
with st.sidebar:
    selected = option_menu(
        "Mask Detection",
        ["About", "Result", "Image Mask Detection", "Real-time Camera Detection"],
        icons=["info", "bar-chart", "image", "camera"],
        menu_icon="cast",
        default_index=0,
    )

# 动态调整框和文字的大小
def adjust_font_and_box_size(image):
    height, width = image.shape[:2]
    resolution = height * width

    if resolution > 1024 * 768:
        font_scale = 2.0
        font_thickness = 10
        box_thickness = 10
    else:
        font_scale = 0.5
        font_thickness = 2
        box_thickness = 3

    return font_scale, font_thickness, box_thickness

# 定义口罩检测函数
def detect_and_predict_mask(image, faceNet, maskNet):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []
    conf_threshold = 0.3
    nms_threshold = 0.4

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
    st.write("Upload an image or choose from the gallery below to detect masks:")

    # 表格展示紧凑的图像选择列表
    gallery_images = [f"{i:02d}.jpg" for i in range(1, 8)]

    st.markdown("### Image Gallery")
    col_titles = st.columns([1, 3, 1])
    with col_titles[0]:
        st.markdown("**No.**")
    with col_titles[1]:
        st.markdown("**Preview**")
    with col_titles[2]:
        st.markdown("**Action**")

    for idx, image_name in enumerate(gallery_images, start=1):
        cols = st.columns([1, 3, 1])
        with cols[0]:
            st.write(idx)
        with cols[1]:
            image_path = os.path.join("./", image_name)
            if os.path.exists(image_path):
                thumbnail = cv2.imread(image_path)
                thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
                thumbnail = cv2.resize(thumbnail, (30, 30))
                st.image(thumbnail, caption=image_name, use_column_width=False)
        with cols[2]:
            if st.button(f"Select {image_name}", key=image_name):
                st.session_state["selected_image"] = image_name
                st.session_state["uploaded_image"] = None
                st.experimental_rerun()

    # 上传图片
    st.write("**OR upload your own image:**")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.session_state["uploaded_image"] = uploaded_file
        st.session_state["selected_image"] = None
        st.experimental_rerun()

    # 展示结果在页面底部
    if st.session_state["uploaded_image"]:
        file_bytes = np.asarray(bytearray(st.session_state["uploaded_image"].read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif st.session_state["selected_image"]:
        image_path = os.path.join("./", st.session_state["selected_image"])
        image = cv2.imread(image_path)
    else:
        image = None

    if image is not None:
        locs, preds = detect_and_predict_mask(image, faceNet, maskNet)
        font_scale, font_thickness, box_thickness = adjust_font_and_box_size(image)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, box_thickness)

        st.markdown("---")
        st.markdown("### Detection Result")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image[:, :, ::-1], caption="Original Image", use_column_width=True)
        with col2:
            st.image(image[:, :, ::-1], caption="Prediction Image", use_column_width=True)

# Real-time Camera Detection 页面
elif selected == "Real-time Camera Detection":
    st.title("Real-time Camera Detection")
    st.write("Use your camera to detect masks in real time.")

    class MaskDetectionTransformer(VideoTransformerBase):
        def transform(self, frame):
            image = frame.to_ndarray(format="bgr24")
            (locs, preds) = detect_and_predict_mask(image, faceNet, maskNet)

            font_scale, font_thickness, box_thickness = adjust_font_and_box_size(image)

            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
                cv2.rectangle(image, (startX, startY), (endX, endY), color, box_thickness)

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
        st.warning("Click 'Select Device' to enable your camera.")
