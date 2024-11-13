import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# 设置页面标题
st.set_page_config(page_title="Dashboard", layout="wide")

# 侧边栏导航
with st.sidebar:
    selected = option_menu(
        "Mask Detection",
        ["About", "Result", "Image Mask Detection"],
        icons=["info", "bar-chart", "image"],
        menu_icon="cast",
        default_index=0,
    )

# 定义面罩检测函数
def detect_and_predict_mask(image, maskNet):
    faces = []
    preds = []

    face = img_to_array(image)
    face = preprocess_input(face)
    faces.append(face)

    faces = np.array(faces, dtype="float32")
    preds = maskNet.predict(faces, batch_size=32)

    return preds

# 加载模型
@st.cache(allow_output_mutation=True)
def load_model_cached():
    return load_model("mask_detector.h5")

maskNet = load_model_cached()

# About 页面
if selected == "About":
    st.title("About")
    st.write("""
    Welcome to the **Dashboard**! This platform provides insights, 
    demonstrations, and results of experiments.
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
        # 读取图像
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = np.expand_dims(file_bytes, axis=0)

        # 检测图像中的口罩
        preds = detect_and_predict_mask(image, maskNet)

        # 显示预测结果
        label = "Mask" if preds[0][0] > preds[0][1] else "No Mask"
        st.write(f"Prediction: {label}")
