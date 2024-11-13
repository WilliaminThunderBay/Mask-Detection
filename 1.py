import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

try:
    import cv2
except ImportError:
    raise ImportError("OpenCV not found. Ensure 'opencv-python-headless' is installed.")

# 设置页面标题
st.set_page_config(page_title="Dashboard", layout="wide")

# 侧边栏导航
with st.sidebar:
    selected = option_menu(
        "Mask Detection",
        ["About", "Result", "Image Mask Detection", "Real-time Camera Detection"],
        icons=["info", "bar-chart", "image", "camera"],
        menu_icon="cast",
        default_index=0,
    )

# 定义加载模型函数
@st.cache_resource
def load_model_cached():
    return load_model("mask_detector.h5", compile=False)

maskNet = load_model_cached()

# 定义面罩检测函数
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
            face = img_to_array(face, dtype="float32")
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

# About 页面
if selected == "About":
    st.title("About")
    st.write("""
    Welcome to the **Mask Detection Dashboard**! This platform provides insights, 
    demonstrations, and results for mask detection.
    """)

# Result 页面
elif selected == "Result":
    st.title("Result")
    st.write("Below are the results of experiments conducted on various datasets.")
    st.image("plot.png", caption="Training Progress", use_column_width=True)

# Image Mask Detection 页面
elif selected == "Image Mask Detection":
    st.title("Image Mask Detection")
    st.write("Upload an image below to detect masks:")

    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        # 原始图像处理
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # 创建两个列，用于并排显示图像
        col1, col2 = st.columns(2)

        # 显示原始图像
        with col1:
            st.image(image[:, :, ::-1], channels="RGB", caption="Original Image", width=300)

        # 检测图像中的口罩
        (locs, preds) = detect_and_predict_mask(image, faceNet, maskNet)

        # 在图像上绘制检测结果
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

        # 显示预测图像
        with col2:
            st.image(image[:, :, ::-1], channels="RGB", caption="Prediction Image", width=300)

# Real-time Camera Detection 页面
elif selected == "Real-time Camera Detection":
    st.title("Real-time Camera Detection")
    st.write("Use your camera to detect masks in real time.")

    # 初始化摄像头运行状态
    if "camera_running" not in st.session_state:
        st.session_state["camera_running"] = False

    # 摄像头检测函数
    def start_camera():
        if not st.session_state["camera_running"]:
            st.session_state["camera_running"] = True
            video_capture = cv2.VideoCapture(0)

            # 创建一个实时更新的区域
            camera_placeholder = st.empty()

            while st.session_state["camera_running"]:
                ret, frame = video_capture.read()
                if not ret:
                    st.error("Failed to grab frame")
                    break

                # 调整帧大小
                frame = cv2.resize(frame, (800, 600))

                # 检测和预测口罩
                (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

                # 绘制检测结果
                for (box, pred) in zip(locs, preds):
                    (startX, startY, endX, endY) = box
                    (mask, withoutMask) = pred

                    label = "Mask" if mask > withoutMask else "No Mask"
                    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                    label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                    cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

                # 将帧转换为RGB格式并在Streamlit中显示
                camera_placeholder.image(frame[:, :, ::-1], channels="RGB")

            # 停止摄像头
            video_capture.release()

    def stop_camera():
        if st.session_state["camera_running"]:
            st.session_state["camera_running"] = False

    # 按钮样式和逻辑
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Camera", key="start_button"):
            if not st.session_state["camera_running"]:
                start_camera()

    with col2:
        if st.button("Stop Camera", key="stop_button"):
            stop_camera()
