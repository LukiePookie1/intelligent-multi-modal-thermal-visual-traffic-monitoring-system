import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from datetime import datetime
import time
from scipy.spatial import distance as dist
from collections import OrderedDict
import shutil
import sys
import tqdm
import subprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mask_rcnn_model = maskrcnn_resnet50_fpn(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
mask_rcnn_model.to(device)
mask_rcnn_model.eval()

coco_labels = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
vehicle_labels = ['car', 'motorcycle', 'bus', 'truck', 'boat']

transform = transforms.Compose([
    transforms.ToTensor()
])

def detect_objects(image):
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = mask_rcnn_model(image_tensor)

    boxes = outputs[0]['boxes'].cpu().numpy()
    labels = outputs[0]['labels'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()
    masks = outputs[0]['masks'].cpu().numpy()

    detected_objects = []

    for box, label, score, mask in zip(boxes, labels, scores, masks):
        if score >= 0.7 and label < len(coco_labels) and coco_labels[label] in vehicle_labels:
            detected_objects.append((box, coco_labels[label], score, mask))

    return detected_objects

def preprocess_image(image_path, target_size=(800, 800)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    h, w, _ = image.shape
    scale = min(target_size[0] / h, target_size[1] / w)
    new_size = (int(w * scale), int(h * scale))
    
    resized_image = cv2.resize(image, new_size)
    
    padded_image = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    padded_image[:resized_image.shape[0], :resized_image.shape[1], :] = resized_image
    
    return padded_image, scale, scale

def visualize_detections(image, detected_objects, scale_x, scale_y, confidence_threshold=0.7):
    image_with_detections = image.copy()

    for box, label, score, _ in detected_objects:
        if score >= confidence_threshold:
            xmin, ymin, xmax, ymax = box

            xmin_adjusted = int(xmin / scale_x)
            ymin_adjusted = int(ymin / scale_y)
            xmax_adjusted = int(xmax / scale_x)
            ymax_adjusted = int(ymax / scale_y)

            cv2.rectangle(image_with_detections, (xmin_adjusted, ymin_adjusted), (xmax_adjusted, ymax_adjusted), (0, 255, 0), 2)
            cv2.putText(image_with_detections, f"{label}: {score:.2f}", (xmin_adjusted, ymin_adjusted - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image_with_detections

class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (startX, startY, endX, endY) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = dist.cdist(np.array(object_centroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1

                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])

        return self.objects

def calculate_congestion(objects, frame_width, frame_height):
    centroids = list(objects.values())
    num_vehicles = len(centroids)

    if num_vehicles == 0:
        return "Empty", num_vehicles

    total_distance = 0
    for i in range(num_vehicles):
        for j in range(i + 1, num_vehicles):
            distance = dist.euclidean(centroids[i], centroids[j])
            total_distance += distance

    num_pairs = num_vehicles * (num_vehicles - 1) // 2
    avg_distance = total_distance / (num_pairs if num_pairs > 0 else 1)
    frame_diagonal = (frame_width ** 2 + frame_height ** 2) ** 0.5
    normalized_avg_distance = avg_distance / frame_diagonal

    if num_vehicles < 5:
        return "Light", num_vehicles
    elif num_vehicles < 10 and normalized_avg_distance > 0.1:
        return "Moderate", num_vehicles
    elif num_vehicles < 15 and normalized_avg_distance > 0.05:
        return "Heavy", num_vehicles
    else:
        return "Very Heavy", num_vehicles
    
def update_detection_counts(objects, detected_objects, unique_vehicles, detection_counts):
    frame_detections = []
    for (object_id, _) in objects.items():
        for box, label, score, _ in detected_objects:
            if label in vehicle_labels:
                if object_id not in unique_vehicles:
                    unique_vehicles[object_id] = label
                    detection_counts[label] = detection_counts.get(label, 0) + 1
                if unique_vehicles[object_id] == label:
                    frame_detections.append((label, score))
    return frame_detections

def add_congestion_level(image, congestion_level, num_vehicles):
    color = (255, 255, 255) if congestion_level == "Empty" else \
            (0, 255, 0) if congestion_level == "Light" else \
            (0, 128, 255) if congestion_level == "Moderate" else \
            (0, 0, 255)

    cv2.putText(image, f"Congestion: {congestion_level}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(image, f"Vehicles: {num_vehicles}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return image

def generate_report(frame_count, processing_time, detection_counts, global_congestion_counts, detection_frames):
    minutes, seconds = divmod(int(processing_time), 60)
    detection_report_lines = [
        "Object Detection and Congestion Report",
        "======================================",
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Total Frames Analyzed: {frame_count}",
        f"Processing Time: {minutes:02d}:{seconds:02d}",
        "",
        "Detection Counts:",
    ]

    for label, count in sorted(detection_counts.items()):
        detection_report_lines.append(f"{label}: {count}")

    detection_report_lines.append("")
    detection_report_lines.append("Congestion Analysis:")

    total_frames = sum(global_congestion_counts.values())
    congestion_percentages = {level: count / total_frames for level, count in global_congestion_counts.items()}

    for level, percentage in congestion_percentages.items():
        detection_report_lines.append(f"{level}: {percentage:.2%}")

    detection_report_lines.append("")
    detection_report_lines.append("Global Congestion:")
    global_congestion = max(congestion_percentages, key=congestion_percentages.get)
    detection_report_lines.append(f"{global_congestion}")

    detection_report_lines.append("")
    detection_report_lines.append("Frame-wise Detections:")

    for frame_idx, frame_detections in enumerate(detection_frames, start=1):
        detection_report_lines.append(f"Frame {frame_idx}:")
        for label, score in frame_detections:
            detection_report_lines.append(f"  - {label}: {score:.2f}")

    detection_report_content = "\n".join(detection_report_lines)
    return detection_report_content

def save_output(detection_report_content, output_video):
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(output_video))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_dir = os.path.join(output_dir, f"{timestamp}_{video_name}")
    os.makedirs(timestamped_dir, exist_ok=True)

    detection_report_file = os.path.join(timestamped_dir, "detection-and-congestion-report.txt")
    with open(detection_report_file, "w") as f:
        f.write(detection_report_content)

    print(f"Object detection and congestion report saved to {detection_report_file}")

    output_video_file = os.path.join(timestamped_dir, os.path.basename(output_video))
    os.rename(output_video, output_video_file)

    print(f"Output video saved to {output_video_file}")

    if sys.platform == "win32":
        os.startfile(output_video_file)
        os.startfile(detection_report_file)
    else:
        open_command = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([open_command, output_video_file])
        subprocess.call([open_command, detection_report_file])


def process_frames(video, frames_dir, output):
    frame_count = 0
    detection_counts = {}
    unique_vehicles = {}
    detection_frames = []
    global_congestion_counts = {"Empty": 0, "Light": 0, "Moderate": 0, "Heavy": 0, "Very Heavy": 0}

    centroid_tracker = CentroidTracker(max_disappeared=50)

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm.tqdm(total=total_frames, unit="frame")

    start_time = time.time()

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_path = os.path.join(frames_dir, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_path, frame)

        preprocessed_image, scale_x, scale_y = preprocess_image(frame_path)
        pil_image = Image.fromarray(preprocessed_image)
        detected_objects = detect_objects(pil_image)

        rects = []
        for box, label, _, _ in detected_objects:
            if label in vehicle_labels:
                xmin, ymin, xmax, ymax = box
                rects.append((xmin, ymin, xmax, ymax))

        objects = centroid_tracker.update(rects)

        frame_detections = update_detection_counts(objects, detected_objects, unique_vehicles, detection_counts)
        detection_frames.append(frame_detections)

        congestion_level, num_vehicles = calculate_congestion(objects, video.get(cv2.CAP_PROP_FRAME_WIDTH), video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        global_congestion_counts[congestion_level] += 1

        image_with_detections = visualize_detections(frame, detected_objects, scale_x, scale_y)
        image_with_detections = add_congestion_level(image_with_detections, congestion_level, num_vehicles)
        output.write(image_with_detections)

        frame_count += 1
        progress_bar.update(1)

    progress_bar.close()

    end_time = time.time()
    processing_time = end_time - start_time

    return frame_count, detection_counts, global_congestion_counts, detection_frames, processing_time

def process_video(input_video):
    frames_dir = "extracted_frames"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(input_video))[0]
    output_video = os.path.join(output_dir, f"{video_name}-output.mp4")

    os.makedirs(frames_dir, exist_ok=True)

    video = cv2.VideoCapture(input_video)
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output = cv2.VideoWriter(output_video, fourcc, frame_rate, (frame_width, frame_height))

    print("Starting video processing...")

    frame_count, detection_counts, global_congestion_counts, detection_frames, processing_time = process_frames(video, frames_dir, output)

    video.release()
    output.release()

    detection_report_content = generate_report(frame_count, processing_time, detection_counts, global_congestion_counts, detection_frames)
    save_output(detection_report_content, output_video)

    shutil.rmtree(frames_dir)

    print("Video processing completed.")

def main():
    if len(sys.argv) != 2:
        print("Usage: python traffic_detection.py <input_video>")
        sys.exit(1)

    input_video = sys.argv[1]
    process_video(input_video)

if __name__ == "__main__":
    main()

