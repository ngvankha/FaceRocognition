import streamlit as st
import tensorflow as tf
from imutils.video import VideoStream
import re
import argparse
import facenet
import imutils
import os
import sys
import math
import pickle
import align.detect_face
import numpy as np
import cv2
import collections
from sklearn.svm import SVC
import tempfile
from PIL import Image
from st_pages import Page, show_pages, add_page_title
import time
import datetime

import streamlit as st
from streamlit_option_menu import option_menu
from deepface import DeepFace

st.set_page_config(page_title="Face Recognition App", layout="wide")
add_page_title()

show_pages(
    [
        Page("Homepage.py", "Home", "🏠"),
        Page("pages/Updating.py", "Updating", "🔄"),
        Page("pages/Database.py", "Database", "📊"),
        Page("pages/History.py", "History", "📖"),
    ]
)


# Define file paths for the recognition history
MAIN_HISTORY_FILE = "recognition_history.txt"
TEMP_FILE = "temp.txt"

recognized_names = set()

# Function to log names to the main history file
def log_to_main_history(name):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} - {name}\n"
    
    # Open main history file in append mode and log the name
    with open(MAIN_HISTORY_FILE, "a") as file:
        file.write(log_entry)

# Function to log names to the temporary file
def log_to_temp(name):
    # Open temporary file in append mode and log the name
    with open(TEMP_FILE, "a") as file:
        file.write(f"{name}\n")

# Function to check if a name has been recognized before in the current session
def is_name_in_temp(name):
    with open(TEMP_FILE, "r") as file:
        temp_names = file.readlines()
    return any(line.strip() == name for line in temp_names)

# Function to clear the temporary file when a new session starts
def clear_temp_file():
    with open(TEMP_FILE, "w") as file:
        file.truncate(0)  # Clears the content of the temp file

# Function to move new names from temp.txt to the main recognition history file
def copy_temp_to_main_history():
    with open(TEMP_FILE, "r") as temp_file:
        temp_names = temp_file.readlines()
        
    for name in temp_names:
        name = name.strip()  # Clean any extra spaces or newlines
        if name:
            log_to_main_history(name)

# Sidebar - Settings
st.sidebar.title("Settings")
st.sidebar.subheader("Choose Input Type")
menu = ["Video", "Webcam", "Images"]
choice = st.sidebar.selectbox("Input Type", menu)

st.sidebar.subheader("Recognition Tolerance")
TOLERANCE = st.sidebar.slider("Tolerance", 0.0, 1.0, 0.7, 0.01)
st.sidebar.info("Lower tolerance is stricter, higher tolerance is looser for face recognition.")

# Common Settings for both Video, Webcam, and Images
MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
INPUT_IMAGE_SIZE = 160
CLASSIFIER_PATH = '../Models/facemodel.pkl'
FACENET_MODEL_PATH = '../Models/20180402-114759.pb'
detection_time_placeholder = st.sidebar.empty()

# Load The Custom Classifier
with open(CLASSIFIER_PATH, 'rb') as file:
    model, class_names = pickle.load(file)
st.info("Custom Classifier Loaded Successfully")

# Load Feature Extraction Model
st.info('Loading feature extraction model...')
facenet.load_model(FACENET_MODEL_PATH)

# Initialize TensorFlow session and GPU settings
tf.Graph().as_default()
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

# Get input/output tensors
images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]
pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "align")


# Main App Content
if choice == "Video":
    st.title("Face Recognition from Video")
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mpeg", "MP4"], accept_multiple_files=True)

    if uploaded_video:
        person_detected = collections.Counter()

        # Load video file
        file_name = uploaded_video[0].name
        cap = cv2.VideoCapture(file_name)
        FRAME_WINDOW = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            # Detect faces
            bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

            # Calculate the time taken for bounding box detection
            end_time = time.time()
            detection_time = end_time - start_time
            
            st.sidebar.write(f"Bounding Box Detection Time: {detection_time:.4f} seconds")

            faces_found = bounding_boxes.shape[0]
            if faces_found > 0:
                det = bounding_boxes[:, 0:4]
                bb = np.zeros((faces_found, 4), dtype=np.int32)

                for i in range(faces_found):
                    bb[i] = det[i].astype(int)
                    cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2]]
                    scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
                    scaled = facenet.prewhiten(scaled)
                    scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)

                    # Make predictions
                    feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                    emb_array = sess.run(embeddings, feed_dict=feed_dict)
                    predictions = model.predict_proba(emb_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                    # Display face name and probability
                    best_name = class_names[best_class_indices[0]]
                    if best_class_probabilities > TOLERANCE:
                        name = best_name
                    else:
                        name = "Unknown"
                    
                    if name != "Unknown" and name not in recognized_names:
                    # Log the name to temp.txt
                        log_to_temp(name)
                        recognized_names.add(name)  # Track this name for this session

                    # Draw bounding box and label
                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                    cv2.putText(frame, name, (bb[i][0], bb[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    cv2.putText(frame, f"{best_class_probabilities[0]:.2f}", (bb[i][0], bb[i][1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                    # Count detections
                    person_detected[best_name] += 1

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)

    else:
        st.warning("Please upload a video for recognition.")
    copy_temp_to_main_history()

elif choice == "Webcam":
    st.title("Nhận diện khuôn mặt từ Webcam")

    # Bắt đầu feed webcam
    cap = VideoStream(src=0).start()
    FRAME_WINDOW = st.empty()

    while True:
        frame = cap.read()
        frame = imutils.resize(frame, width=1200, height=600)
        frame = cv2.flip(frame, 1)

        start_time = time.time()

        # Dùng MTCNN để phát hiện khuôn mặt
        bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
        end_time = time.time()
        detection_time = end_time - start_time
        detection_time_placeholder.write(f"Thời gian phát hiện khuôn mặt: {detection_time:.4f} giây")

        # Kiểm tra nếu có khuôn mặt
        faces_found = bounding_boxes.shape[0]
        if faces_found > 0:
            for i in range(faces_found):
                # Lấy tọa độ bounding box
                det = bounding_boxes[i, 0:4].astype(int)
                left, top, right, bottom = det[0], det[1], det[2], det[3]

                # Cắt khuôn mặt từ khung hình
                cropped_face = frame[top:bottom, left:right]

                # Kiểm tra độ thật của khuôn mặt bằng DeepFace
                try:
                    face_objs = DeepFace.extract_faces(cropped_face, enforce_detection=False, anti_spoofing=True)
                    is_real = face_objs[0]["is_real"]

                    # Nếu khuôn mặt giả
                    if not is_real:
                        label = "Fake Face"
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        continue  # Bỏ qua dự đoán tiếp theo nếu mặt là giả
                except Exception as e:
                    st.error(f"Lỗi khi kiểm tra khuôn mặt: {e}")
                    continue

                # Dự đoán bằng mô hình nhận diện khuôn mặt
                scaled = cv2.resize(cropped_face, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
                scaled = facenet.prewhiten(scaled)
                scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)

                # Tính toán embedding và dự đoán
                feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                emb_array = sess.run(embeddings, feed_dict=feed_dict)
                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                # Hiển thị tên và độ chính xác
                best_name = class_names[best_class_indices[0]]
                if best_class_probabilities > TOLERANCE:
                    name = best_name
                else:
                    name = "Unknown"

                if name != "Unknown" and name not in recognized_names:
                    # Log the name to temp.txt
                    log_to_temp(name)
                    recognized_names.add(name)  # Track this name for this session

                # Vẽ hộp và hiển thị tên khuôn mặt
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(frame, f"{best_class_probabilities[0]:.2f}", (left, top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Hiển thị khung hình
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
    copy_temp_to_main_history()

elif choice == "Images":
    st.title("Face Recognition from Images")

    uploaded_images = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_images:
        for uploaded_image in uploaded_images:
            image = Image.open(uploaded_image)
            image = np.array(image)
            FRAME_WINDOW = st.image(image, caption="Uploaded Image", use_column_width=True)

            start_time = time.time()
            # Detect faces
            bounding_boxes, _ = align.detect_face.detect_face(image, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

            # Calculate the time taken for bounding box detection
            end_time = time.time()
            detection_time = end_time - start_time
            
            st.sidebar.write(f"Bounding Box Detection Time: {detection_time:.4f} seconds")

            faces_found = bounding_boxes.shape[0]
            if faces_found > 0:
                det = bounding_boxes[:, 0:4]
                bb = np.zeros((faces_found, 4), dtype=np.int32)

                for i in range(faces_found):
                    bb[i] = det[i].astype(int)
                    cropped = image[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2]]
                    scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
                    scaled = facenet.prewhiten(scaled)
                    scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)

                    # Make predictions
                    feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                    emb_array = sess.run(embeddings, feed_dict=feed_dict)
                    predictions = model.predict_proba(emb_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                    # Display face name and probability
                    best_name = class_names[best_class_indices[0]]
                    if best_class_probabilities > TOLERANCE:
                        name = best_name
                    else:
                        name = "Unknown"
                    
                    if name != "Unknown" and name not in recognized_names:
                    # Log the name to temp.txt
                        log_to_temp(name)
                        recognized_names.add(name)  # Track this name for this session
                    
                    # Draw bounding box and label
                    cv2.rectangle(image, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                    cv2.putText(image, name, (bb[i][0], bb[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    cv2.putText(image, f"{best_class_probabilities[0]:.2f}", (bb[i][0], bb[i][1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image, caption="Processed Image", use_column_width=True)

    copy_temp_to_main_history()
