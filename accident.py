import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import defaultdict, deque
from deep_sort_realtime.deepsort_tracker import DeepSort
import time

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize video capture
cap = cv2.VideoCapture(r"C:\code\traffic\inputs\II TOWN TRAFFIC PS HIT&RUN CASE.mp4")

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=10, n_init=3)

# Store vehicle tracking data
vehicle_data = defaultdict(lambda: {
    "positions": deque(maxlen=5),
    "speed": 0,
    "prev_speed": 0,
    "stopped_frames": 0,
    "bbox": None,
    "direction": None,
    "missing_frames": 0  # Track missing frames
})

# Parameters
CONFIDENCE_THRESHOLD = 0.1  # Track all vehicles
SPEED_DROP_PERCENTAGE = 0.90  # 90% speed drop
IOU_THRESHOLD = 0.5  # 50% intersection ratio for collision detection
DIRECTION_THRESHOLD = 30  # Threshold for angle change in vehicle direction
MAX_MISSING_FRAMES = 5  # Maximum frames a vehicle can disappear before considering it an accident

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    # Calculate intersection area
    intersection_x1 = max(x1, x3)
    intersection_y1 = max(y1, y3)
    intersection_x2 = min(x2, x4)
    intersection_y2 = min(y2, y4)

    intersection_width = max(0, intersection_x2 - intersection_x1)
    intersection_height = max(0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height

    # Calculate individual box areas
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)

    # Calculate IoU
    iou = intersection_area / min(area1, area2)  # Intersection over the smaller bounding box

    return iou

def calculate_direction(c1, c2):
    """Calculate direction of movement using the change in center points."""
    dx = c2[0] - c1[0]
    dy = c2[1] - c1[1]
    angle = np.arctan2(dy, dx) * 180 / np.pi  # Angle in degrees
    return angle

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform YOLO inference
    results = model(frame)

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf.item()
            class_id = int(box.cls.item())

            if class_id in [2, 3, 5, 7] and confidence > CONFIDENCE_THRESHOLD:
                bbox = [x1, y1, x2 - x1, y2 - y1]
                detections.append((bbox, confidence, class_id))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    active_vehicles = {}

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id

        # Get bounding box
        x1, y1, w, h = map(int, track.to_ltwh())
        x2, y2 = x1 + w, y1 + h
        cx, cy = x1 + w // 2, y1 + h // 2

        # Store past positions and calculate direction
        vehicle_data[track_id]["positions"].append((cx, cy))
        vehicle_data[track_id]["bbox"] = (x1, y1, x2, y2)

        # Calculate speed
        if len(vehicle_data[track_id]["positions"]) > 1:
            prev_cx, prev_cy = vehicle_data[track_id]["positions"][-2]
            speed = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
        else:
            speed = 0

        # Detect speed drop
        prev_speed = vehicle_data[track_id]["prev_speed"]
        speed_drop = prev_speed - speed
        speed_drop_percentage = speed_drop / max(prev_speed, 1)

        vehicle_data[track_id]["prev_speed"] = speed
        active_vehicles[track_id] = vehicle_data[track_id]["bbox"]

        # Calculate direction if enough positions are available
        if len(vehicle_data[track_id]["positions"]) >= 2:
            direction = calculate_direction(vehicle_data[track_id]["positions"][-2], (cx, cy))
            vehicle_data[track_id]["direction"] = direction

        # Draw bounding box and direction
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Detect vehicle collisions and disappearance
    detected_accidents = []
    vehicle_ids = list(active_vehicles.keys())

    for i in range(len(vehicle_ids)):
        id1 = vehicle_ids[i]
        bbox1 = active_vehicles[id1]

        for j in range(i + 1, len(vehicle_ids)):
            id2 = vehicle_ids[j]
            bbox2 = active_vehicles[id2]

            # Check IoU (bounding box intersection ratio)
            iou = calculate_iou(bbox1, bbox2)

            # Check for accident if vehicles are in different directions and have intersection
            if iou >= IOU_THRESHOLD:
                direction1 = vehicle_data[id1]["direction"]
                direction2 = vehicle_data[id2]["direction"]

                if direction1 is not None and direction2 is not None:
                    direction_diff = abs(direction1 - direction2)
                    if direction_diff > DIRECTION_THRESHOLD:  # Different directions
                        print(f"🚨 Possible collision between {id1} and {id2} due to direction difference")

                # Check for speed drop (90% drop in speed)
                if vehicle_data[id1]["prev_speed"] > 0 and (vehicle_data[id1]["prev_speed"] - vehicle_data[id1]["speed"]) / vehicle_data[id1]["prev_speed"] >= SPEED_DROP_PERCENTAGE:
                    detected_accidents.append((id1, id2))

                if vehicle_data[id2]["prev_speed"] > 0 and (vehicle_data[id2]["prev_speed"] - vehicle_data[id2]["speed"]) / vehicle_data[id2]["prev_speed"] >= SPEED_DROP_PERCENTAGE:
                    detected_accidents.append((id1, id2))

    # Check for vehicle disappearance (missing bounding box for consecutive frames)
    for track_id, data in vehicle_data.items():
        if data["bbox"] is None:
            data["missing_frames"] += 1
        else:
            data["missing_frames"] = 0  # Reset missing frames if vehicle is detected again

        if data["missing_frames"] >= MAX_MISSING_FRAMES:
            print(f"🚨 Vehicle {track_id} has disappeared for {MAX_MISSING_FRAMES} frames. Potential accident.")
            # Save the frame where the vehicle disappeared
            timestamp = int(time.time())
            accident_filename = f"accident_{track_id}_{timestamp}.jpg"
            cv2.imwrite(accident_filename, frame)
            print(f"🚨 Accident detected for vehicle {track_id}! Frame saved: {accident_filename}")

    # Log and save accidents
    for id1, id2 in detected_accidents:
        timestamp = int(time.time())
        accident_filename = f"accident_{id1}_{id2}_{timestamp}.jpg"
        cv2.imwrite(accident_filename, frame)
        print(f"🚨 Accident detected between {id1} and {id2}! Frame saved: {accident_filename}")

    # Show frame
    cv2.imshow("Accident Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
