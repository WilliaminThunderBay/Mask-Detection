# import the necessary packages
import sys

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))
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

            face = frame[startY:endY, startX:endX]
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


def start_video():
    global vs, running
    running = True
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    print("[INFO] Video stream started")
    detect_video()


def stop_video():
    global running, vs
    if running:
        running = False
        if vs is not None:
            vs.stop()
            vs.stream.release()
            print("[INFO] Video stream stopped")
            cv2.destroyAllWindows()
        root.after_cancel(detect_video)
    if vs.stream.isOpened():
        print("Camera is still open")
    else:
        print("Camera is closed")


def detect_video():
    global running
    if not running:
        return

    frame = vs.read()
    frame = imutils.resize(frame, width=800)

    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or not running:
        stop_video()
    else:
        root.after(10, detect_video)


def detect_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    image = cv2.imread(file_path)
    if image is None:
        messagebox.showerror("Error", "Cannot open image")
        return

    (locs, preds) = detect_and_predict_mask(image, faceNet, maskNet)

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        cv2.putText(image, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def quit_app():
    global running, vs
    if running:
        stop_video()
    root.quit()
    root.destroy()

    sys.exit(0)


# Load face detector model
prototxtPath = r"C:\Users\kled2\Desktop\study\Comp Vision\Face-Mask-Detection\face_detector\deploy.prototxt"
weightsPath = r"C:\Users\kled2\Desktop\study\Comp Vision\Face-Mask-Detection\face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load mask detector model
maskNet = load_model("mask_detector.model")

# Initialize tkinter GUI
root = tk.Tk()
root.title("Face Mask Detection")

root.geometry("400x300")

start_btn = tk.Button(root, text="Start Video", command=start_video)
start_btn.pack(pady=10)

stop_btn = tk.Button(root, text="Stop Video", command=stop_video)
stop_btn.pack(pady=10)

detect_img_btn = tk.Button(root, text="Detect Image", command=detect_image)
detect_img_btn.pack(pady=10)

quit_btn = tk.Button(root, text="Quit", command=quit_app)
quit_btn.pack(pady=10)

root.mainloop()
