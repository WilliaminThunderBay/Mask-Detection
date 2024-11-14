import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 设置页面标题
st.set_page_config(page_title="Mask Detection via WebRTC", layout="wide")

# 加载模型
@st.cache_resource
def load_models():
    prototxtPath = "./face_detector/deploy.prototxt"
    weightsPath = "./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    maskNet = load_model("mask_detector.h5")
    return faceNet, maskNet

faceNet, maskNet = load_models()

# 定义视频帧处理器
class MaskDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.faceNet = faceNet
        self.maskNet = maskNet

    def _detect_and_predict_mask(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()

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

                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = np.array(face, dtype="float32")
                face = face / 255.0

                faces.append(face)
                locs.append((startX, startY, endX, endY))

        if len(faces) > 0:
            faces = np.array(faces, dtype="float32")
            preds = self.maskNet.predict(faces, batch_size=32)

        return (locs, preds)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        (locs, preds) = self._detect_and_predict_mask(img)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(img, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# 页面布局
st.title("Real-time Mask Detection via Browser")
st.write("Use your camera to detect masks in real time.")

webrtc_streamer(
    key="mask-detection",
    video_transformer_factory=MaskDetectionTransformer,
    media_stream_constraints={"video": True, "audio": False},
)
