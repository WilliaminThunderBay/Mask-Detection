import streamlit as st
from streamlit_option_menu import option_menu
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

# 动态调整框和文字的大小
def adjust_font_and_box_size(image):
    height, width = image.shape[:2]
    resolution = height * width
    if resolution > 1024 * 768:  # 高分辨率
        font_scale = 2.0
        font_thickness = 10
        box_thickness = 10
    else:  # 低分辨率
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

# Image Mask Detection 页面
if selected == "Image Mask Detection":
    st.title("Image Mask Detection")
    st.write("Select a preloaded image, preview thumbnails below, or upload your own image:")

    # 显示所有七张图片的缩略图
    image_files = [f"0{i}.jpg" for i in range(1, 8) if os.path.exists(f"0{i}.jpg")]
    selected_image = None

    col1, col2, col3 = st.columns(3)
    for idx, image_file in enumerate(image_files):
        with [col1, col2, col3][idx % 3]:
            # 显示缩略图，调整大小
            image = cv2.imread(image_file)
            resized_image = cv2.resize(image, (150, 100))  # 缩略图尺寸
            st.image(resized_image[:, :, ::-1], caption=image_file, use_column_width=True)
            if st.button(f"Select {image_file}"):
                selected_image = image_file

    # 允许用户上传自己的照片
    uploaded_file = st.file_uploader("Or upload an image:", type=["jpg", "png", "jpeg"])

    if selected_image or uploaded_file:
        # 如果用户选择了预加载的图片
        if selected_image:
            image = cv2.imread(selected_image)
        else:  # 用户上传了自己的图片
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image[:, :, ::-1], channels="RGB", caption="Original Image", use_column_width=True)

        font_scale, font_thickness, box_thickness = adjust_font_and_box_size(image)

        (locs, preds) = detect_and_predict_mask(image, faceNet, maskNet)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, box_thickness)

        with col2:
            st.image(image[:, :, ::-1], channels="RGB", caption="Prediction Image", use_column_width=True)
