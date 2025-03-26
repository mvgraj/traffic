import cv2
import os
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Initialize PaddleOCR without CLS model
ocr = PaddleOCR(
    det_model_dir=r"C:\code\traffic\inputs\ch_PP-OCRv4_det_infer",
    rec_model_dir=r"C:\code\traffic\inputs\ch_PP-OCRv3_rec_infer",
    use_angle_cls=False  # Disable classification model
)

# Load YOLO model (Replace with your trained model)
model = YOLO(r"C:\code\traffic\runs\detect\train3\weights\best.pt")

# Input and output directories
video_folder = r"C:\code\traffic\inputs\numberplate_inputs"
extracted_plates_folder = r"C:\code\traffic\inputs\extracted_plates\new12"
annotated_frames_folder = r"C:\code\traffic\inputs\annotated_frames"
output_video_folder = r"C:\code\traffic\inputs\output_videos\new12"

# Create directories if they don't exist
os.makedirs(extracted_plates_folder, exist_ok=True)
os.makedirs(annotated_frames_folder, exist_ok=True)
os.makedirs(output_video_folder, exist_ok=True)

# Store unique number plates
unique_plates = set()

# Process each video in the folder
for video_file in os.listdir(video_folder):
    if video_file.endswith((".mp4", ".avi", ".mov", ".MP4")):
        video_path = os.path.join(video_folder, video_file)
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Set up output video writer
        output_video_path = os.path.join(output_video_folder, f"processed_{video_file}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        frame_count = 0  # To keep track of frames

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1  # Increment frame count
            annotated_frame = frame.copy()  # Copy to draw annotations

            results = model(frame)  # Run YOLO on the frame
            for r in results:
                for box in r.boxes.xyxy:  # Extract bounding boxes
                    x1, y1, x2, y2 = map(int, box[:4])  # Convert to integers
                    plate_crop = frame[y1:y2, x1:x2]  # Crop number plate

                    if plate_crop.size > 0:
                        # Convert to grayscale
                        plate_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

                        # Use PaddleOCR (PP-OCR) for text recognition
                        result = ocr.ocr(plate_gray, cls=True)
                        plate_text = ""

                        if result and result[0]:
                            for line in result:
                                for word in line:
                                    plate_text += word[1][0] + " "  # Extract detected text

                        plate_text = plate_text.strip().upper()

                        # Store only unique number plates
                        if plate_text:
                            unique_plates.add(plate_text)

                            # Save cropped plate image
                            plate_filename = f"{plate_text}_{frame_count}.jpg"
                            plate_path = os.path.join(extracted_plates_folder, plate_filename)
                            cv2.imwrite(plate_path, plate_crop)

                            # Draw bounding box and text on frame
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(annotated_frame, plate_text, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Save annotated frame as an image
            frame_filename = f"frame_{frame_count}.jpg"
            frame_path = os.path.join(annotated_frames_folder, frame_filename)
            cv2.imwrite(frame_path, annotated_frame)

            # Write the annotated frame to the output video
            out.write(annotated_frame)

        cap.release()
        out.release()

print("✅ Unique number plate extraction and video processing complete!")
print("✅ Extracted Unique Plates:", unique_plates)
