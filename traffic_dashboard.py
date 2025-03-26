


import streamlit as st
import cv2
import os
import time
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from PIL import Image

# Streamlit Page Configuration
st.set_page_config(page_title="Traffic Management Dashboard", layout="wide")

# Horizontal Menu
selected = st.selectbox(
    "Select a Feature",
    ["Home", "Number Plate Detection", "Helmet Check", "Triple Riders", "Wrong Route", "Red Signal Jump", "Accident Prediction"],
)

# Video Paths Dictionary
video_paths = {
    "Home": "C:/code/traffic/inputs/home_feed.mp4",
    "Helmet Check": "C:/code/traffic/inputs/helmet_feed.mp4",
    "Triple Riders": "C:/code/traffic/inputs/triple_riders.mp4",
    "Wrong Route": "C:/code/traffic/inputs/wrong_route.mp4",
    "Red Signal Jump": "C:/code/traffic/inputs/red_signal.mp4",
    "Accident Prediction": "C:/code/traffic/inputs/accident_feed.mp4"
}

# Number Plate Detection - Area-Based Video Selection
number_plate_videos = {
    "Downtown": r"C:\code\traffic\inputs\numberplate_inputs\DJI_0866.MP4",
    "Highway": "C:/code/traffic/inputs/numberplate_inputs/highway.mp4",
    "Residential Area": "C:/code/traffic/inputs/numberplate_inputs/residential.mp4",
    "Mall Parking": "C:/code/traffic/inputs/numberplate_inputs/mall.mp4"
}

# Load YOLO Model for Number Plate Detection
model = YOLO(r"C:\code\traffic\runs\detect\train3\weights\number_ocr.pt")

# Initialize PaddleOCR
ocr = PaddleOCR(
    det_model_dir=r"C:\code\traffic\inputs\ch_PP-OCRv4_det_infer",
    rec_model_dir=r"C:\code\traffic\inputs\ch_PP-OCRv3_rec_infer",
    use_angle_cls=False
)

# Function to Stream Video (For All Sections Except Number Plate Detection)
def stream_video(video_path, target_width=600):
    cap = cv2.VideoCapture(video_path)
    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video when it ends
            continue

        # Resize video for smoother performance
        height, width, _ = frame.shape
        scale = target_width / width
        new_size = (target_width, int(height * scale))
        frame = cv2.resize(frame, new_size)

        # Convert color (BGR to RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display frame in Streamlit
        frame_placeholder.image(frame, use_container_width=True)

        time.sleep(1 / 30)  # 30 FPS

    cap.release()


# Function for Real-Time Number Plate Detection with Cropped Plates
def realtime_numberplate_detection(video_path, target_width=600):
    cap = cv2.VideoCapture(video_path)
    frame_placeholder = st.empty()
    detected_plates_placeholder = st.empty()
    plate_images_placeholder = st.container()

    unique_plates = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video when it ends
            continue

        annotated_frame = frame.copy()
        plate_images = []  # Store cropped plates
        plate_texts = []   # Store OCR outputs

        # YOLO Detection
        results = model(frame)
        for r in results:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])
                plate_crop = frame[y1:y2, x1:x2]

                if plate_crop.size > 0:
                    plate_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                    result = ocr.ocr(plate_gray, cls=True)

                    plate_text = ""
                    if result and result[0]:
                        for line in result:
                            for word in line:
                                plate_text += word[1][0] + " "

                    plate_text = plate_text.strip().upper()

                    if plate_text:
                        unique_plates.add(plate_text)

                        # Convert cropped plate to PIL Image for Streamlit
                        plate_image = Image.fromarray(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB))
                        plate_images.append(plate_image)
                        plate_texts.append(plate_text)

                        # Draw on Frame
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, plate_text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        detected_plates_placeholder.write(f"**Detected Plates:** {', '.join(unique_plates)}")

          # Display Cropped Plates with OCR Output
        if plate_images:
            with plate_images_placeholder:
                num_cols = 5  # Maximum images per row
                num_rows = (len(plate_images) + num_cols - 1) // num_cols  # Calculate rows needed
                
                for row in range(num_rows):
                    cols = st.columns(num_cols)  # Create a row with 5 columns
                    for i in range(num_cols):
                        index = row * num_cols + i
                        if index < len(plate_images):  # Check if image exists
                            with cols[i]:
                                st.image(plate_images[index], caption=f"Plate: {plate_texts[index]}", use_container_width=True)

        # Resize for smoother streaming
        height, width, _ = annotated_frame.shape
        scale = target_width / width
        new_size = (target_width, int(height * scale))
        annotated_frame = cv2.resize(annotated_frame, new_size)

        # Convert Color and Stream
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(annotated_frame, use_container_width=True)

        time.sleep(1 / 30)  # Real-time FPS

    cap.release()


# **Home Page**
if selected == "Home":
    st.title("🏠 Traffic Management Dashboard")
    st.subheader("Live Traffic Monitoring System")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Violations Today", "245", "🔺 +15%")
    col2.metric("Average Traffic Speed", "42 km/h", "🔻 -5%")
    col3.metric("Accident Risk Level", "High", "⚠️")

    col4, col5, col6 = st.columns(3)
    col4.metric("Helmet Violations", "78", "🔺 +10%")
    col5.metric("Triple Riders Spotted", "32", "🔻 -3%")
    col6.metric("Red Light Violations", "56", "🔺 +7%")

    # Video Streaming
    stream_video(video_paths["Home"])

# **Number Plate Detection (Real-Time)**
elif selected == "Number Plate Detection":
    st.title("🔍 Real-Time Number Plate Detection")

    # Area Selection for Different Videos
    selected_area = st.selectbox("Select an Area", list(number_plate_videos.keys()))
    video_path = number_plate_videos[selected_area]

    realtime_numberplate_detection(video_path)

# **Other Sections**
elif selected in video_paths:
    st.title(f"🚦 {selected}")
    stream_video(video_paths[selected])




